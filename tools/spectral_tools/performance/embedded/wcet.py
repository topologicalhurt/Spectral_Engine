"""P5 synthesis: parametric per-block WCET bound from the validated stack.

Composes the three layers (M7_PERF_MODEL_PLAN):
  L2  per-kernel steady-state cycles  [modeled: mca/CortexM7Model, body
      throughput validated <=1% on the P3 microbench set]
  L1  measured per-block instruction profile of the real kernel under QEMU
      [measured: qemu-tcg]
  L3  memory-stall serial bounds under the CHOSEN hardware configuration
      [modeled: P4, constants parsed from daisy_seed_sdram.h]

into a per-block worst-case cycle bound:

  WCET_block(active, scan_segments, block, cold) =
      T1  synthesis loops:  active * block * worst_cycles_per_voice_sample
    + T2  non-loop residual: residual_insns * CPI_BOUND
                             + scan_segments * BRANCH_MISPREDICT_CYCLES
    + T3  memory (cold):    (code+data working-set lines + scan lines)
                             * row-miss linefill serial cost

Every term carries its provenance; the result is an upper BOUND, not an
expectation — real-time scheduling needs "never worse than", and a bound
presented as a prediction would repeat the bug class P5 just retired.
WCET excludes init/load (not real-time) and assumes DTCM placement for hot
data per the Daisy contract (zero-wait [TRM/AN4891]).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .codegen import codegen_report
from .counts import measure
from .fixture import default_fixture
from .memory_model import ModelConstants, load_constants
from .toolchain import Toolchain

# Voice-samples produced per loop iteration of each hot kernel.
# [structure: codegen-verified — GCC re-unrolls the 4x source body to 16
# samples/iter in the sustain loop (pass 239, re-checked on xPack 15.2.1);
# the SMLALD pair loop advances two voices one sample (2 voice-samples/iter);
# tail and fade are 1 sample/iter. Re-verify on a toolchain change: a drift
# makes cycles/iter jump far outside the guard band below.]
SAMPLES_PER_ITER = {
    "synth_core_m7/.L452": 16.0,   # main sustain loop
    "synth_core_pair_m7/.L534": 2.0,
    "synth_core_m7/.L454": 1.0,    # scalar tail
    "synth_fade_m7/.L624": 1.0,
}
# Guard band for cycles/voice-sample: if a kernel falls outside, the unroll
# map has drifted and the WCET must not be trusted silently.
CYC_PER_VS_GUARD = (5.0, 80.0)

# Upper-bound CPI for the non-loop residual instructions.
# [bound: in-order dual-issue with zero-wait DTCM data sustains CPI ~<=1.3
# (loads <=1/3 of the stream, <=1 load-use bubble each [jnk0le]); branch
# mispredicts are charged separately. 2.0 is a deliberate margin above that.]
RESIDUAL_CPI_BOUND = 2.0

# Per-segment scan-check instruction bound.
# [bound: the activation scan body (start compare, bounds check, advance) is
# ~10 instructions in the pass-239 codegen; 20 is a 2x margin.]
SCAN_INSNS_PER_CHECK_BOUND = 20.0

# [measured-community: jnk0le STM32H743 — misprediction penalty 8 cycles
# (6 with early flag forwarding); WCET takes the unforwarded worst case.]
BRANCH_MISPREDICT_CYCLES = 8.0

# Back-edge upper bias per loop iteration on top of mca body throughput
# [P3 finding: real M7 folds predicted-taken back-edges (~0-1 cyc); the hand
# range upper bound is 2 — charged here as worst case].
BACKEDGE_CYCLES_PER_ITER = 2.0


class WcetError(RuntimeError):
    pass


DAISY_CONFIG_RELPATH = "api/daisy_seed/daisy_seed_config.h"


def parse_daisy_app_config(repo_root: Path) -> dict[str, int]:
    """C-truth: Daisy app-level constants (sample rate, callback block,
    active cap). Fails loudly if the contract moves."""
    path = repo_root / DAISY_CONFIG_RELPATH
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WcetError(f"cannot read {DAISY_CONFIG_RELPATH}: {exc}") from exc
    out: dict[str, int] = {}
    for key in ("DAISY_SAMPLE_RATE", "DAISY_AUDIO_BLOCK_SIZE", "DAISY_MAX_ACTIVE"):
        m = re.search(rf"#define\s+{key}\s+(\d+)", text)
        if m is None:
            raise WcetError(
                f"{key} not found in {DAISY_CONFIG_RELPATH} — the app config "
                "contract moved; fix the parse, do not hardcode"
            )
        out[key] = int(m.group(1))
    return out


@dataclass(frozen=True, slots=True)
class WcetInputs:
    """Live stack outputs the bound is built from."""
    worst_cyc_per_voice_sample: float        # L2 + back-edge bias, worst kernel
    worst_kernel: str
    sustain_insns_per_voice_sample: float    # L2: the CHEAPEST loop's rate, so
                                             # subtracting it OVER-estimates the
                                             # residual (safe for a bound)
    measured_insns_per_block: float          # L1, fixture
    measured_voice_samples_per_block: float  # fixture structure
    fixture_peak_active: int                 # fixture structure (overlap)
    code_lines: int                          # L1 working set (32B lines)
    data_lines: int


@dataclass(frozen=True, slots=True)
class WcetReport:
    inputs: WcetInputs
    constants: ModelConstants
    active: int
    scan_segments: int
    block: int
    sample_rate: int = 48000  # overridden by wcet() from the C SSOT

    @property
    def t1_synth_cycles(self) -> float:
        return self.active * self.block * self.inputs.worst_cyc_per_voice_sample

    @property
    def residual_insns(self) -> float:
        """Per-block non-loop instructions: measured fixture total minus the
        loop part at the SUSTAIN rate (the cheapest insns/voice-sample of the
        four kernels, so the subtraction over-estimates the residual — safe
        for a bound), then scaled by the active-voice ratio (per-active
        overhead such as fade windows and bounds checks scales with active;
        scaling the fixed part too is conservative)."""
        fixture_loop = (self.inputs.measured_voice_samples_per_block
                        * self.inputs.sustain_insns_per_voice_sample)
        fixture_residual = max(0.0, self.inputs.measured_insns_per_block - fixture_loop)
        scale = max(1.0, self.active / max(1, self.inputs.fixture_peak_active))
        return fixture_residual * scale

    @property
    def t2_residual_cycles(self) -> float:
        """Residual at the CPI bound, plus the scan charged explicitly: a
        bounded instruction count per check plus a worst-case mispredict per
        check (the scan's branches are data-dependent)."""
        scan_cycles = self.scan_segments * (
            SCAN_INSNS_PER_CHECK_BOUND * RESIDUAL_CPI_BOUND + BRANCH_MISPREDICT_CYCLES)
        return self.residual_insns * RESIDUAL_CPI_BOUND + scan_cycles

    @property
    def t3_cold_memory_cycles(self) -> float:
        """Cold-cache serial bound: every working-set line plus the scan's
        segment lines refilled at the row-miss price. The SDRAM row-miss fill
        upper-bounds the flash fill (flash on AXI@200MHz with 2WS is strictly
        faster than FMC SDRAM with a full row open), so one constant bounds
        both I-side and D-side cold traffic."""
        c = self.constants
        seg_bytes = 16  # SpectralSegmentQ15 upper size (14 packed / 16 chirp)
        scan_lines = (self.scan_segments * seg_bytes + c.line_bytes - 1) // c.line_bytes
        lines = self.inputs.code_lines + self.inputs.data_lines + scan_lines
        return lines * c.line_fill_cycles(False)

    @property
    def wcet_cycles(self) -> float:
        return self.t1_synth_cycles + self.t2_residual_cycles + self.t3_cold_memory_cycles

    def as_dict(self) -> dict[str, Any]:
        c = self.constants
        cpu_hz = c.defines["SPECTRAL_DAISY_CPU_HZ"]
        sr = self.sample_rate
        budget = cpu_hz / sr * self.block
        return {
            "provenance": "upper BOUND composed from [modeled: mca validated <=1%] + "
                          "[measured: qemu-tcg] + [modeled: P4 serial memory bounds]; "
                          "excludes init/load; assumes Daisy DTCM placement contract",
            "parameters": {"active": self.active, "scan_segments": self.scan_segments,
                           "block": self.block},
            "terms": {
                "t1_synth_loops": {
                    "cycles": round(self.t1_synth_cycles, 0),
                    "worst_kernel": self.inputs.worst_kernel,
                    "cyc_per_voice_sample": round(self.inputs.worst_cyc_per_voice_sample, 2),
                    "provenance": "modeled: mca/CortexM7Model (P3-validated) + "
                                  f"back-edge bias {BACKEDGE_CYCLES_PER_ITER}/iter worst case",
                },
                "t2_nonloop_residual": {
                    "cycles": round(self.t2_residual_cycles, 0),
                    "residual_insns": round(self.residual_insns, 0),
                    "cpi_bound": RESIDUAL_CPI_BOUND,
                    "mispredict_cycles_per_scan_check": BRANCH_MISPREDICT_CYCLES,
                    "provenance": "measured insns [qemu-tcg] x CPI bound [bound: see "
                                  "wcet.py] + jnk0le-measured mispredict worst case",
                },
                "t3_cold_memory": {
                    "cycles": round(self.t3_cold_memory_cycles, 0),
                    "provenance": "modeled: P4 row-miss serial fill (chosen timing set) "
                                  "over measured working-set lines; SDRAM price "
                                  "upper-bounds flash",
                },
            },
            "wcet_block_cycles": round(self.wcet_cycles, 0),
            "block_budget_cycles": round(budget, 0),
            "sample_rate": sr,
            "budget_fraction": round(self.wcet_cycles / budget, 3),
            "constants_source": c.header_relpath,
        }


def derive_inputs(tc: Toolchain, *, out_dir: Path) -> WcetInputs:
    """Pull live numbers from the stack (no cached copies)."""
    report = codegen_report(tc, out_dir=out_dir)
    if report.failed_regions:
        raise WcetError(f"mca regions failed: {report.failed_regions}")

    worst_cpv = 0.0
    worst_kernel = ""
    min_ipv = float("inf")
    for loop in report.loops:
        key = f"{loop.kernel}/{loop.label}"
        spi = SAMPLES_PER_ITER.get(key)
        if spi is None:
            continue
        cpv = (loop.cycles_per_iter + BACKEDGE_CYCLES_PER_ITER) / spi
        if not (CYC_PER_VS_GUARD[0] <= cpv <= CYC_PER_VS_GUARD[1]):
            raise WcetError(
                f"{key}: {cpv:.1f} cyc/voice-sample outside guard band "
                f"{CYC_PER_VS_GUARD} — the unroll map (SAMPLES_PER_ITER) has "
                "drifted from codegen; re-verify before trusting WCET"
            )
        if cpv > worst_cpv:
            worst_cpv = cpv
            worst_kernel = key
        min_ipv = min(min_ipv, loop.instructions_per_iter / spi)
    if worst_cpv <= 0.0 or min_ipv == float("inf"):
        raise WcetError("no known hot loops found — kernel extraction drifted")

    counts = measure(tc, out_dir=out_dir, verify_reproducible=False)
    process = counts.range("spectral_arm32_process")
    if process is None:
        raise WcetError("spectral_arm32_process missing from counts ranges")

    spec = default_fixture()
    n_blocks = spec.total_samples // spec.block_samples
    voice_samples = sum(v.length for v in spec.voices)
    # Fixture peak concurrency: max interval overlap of the voices.
    events = sorted(
        [(v.start, 1) for v in spec.voices]
        + [(v.start + v.length, -1) for v in spec.voices])
    peak = cur = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)

    return WcetInputs(
        worst_cyc_per_voice_sample=worst_cpv,
        worst_kernel=worst_kernel,
        sustain_insns_per_voice_sample=min_ipv,
        measured_insns_per_block=process.insns / n_blocks,
        measured_voice_samples_per_block=voice_samples / n_blocks,
        fixture_peak_active=max(1, peak),
        code_lines=counts.region_lines32.get("code", 0),
        data_lines=(counts.region_lines32.get("ssram23", 0)
                    + counts.region_lines32.get("bulk60", 0)),
    )


def wcet(
    tc: Toolchain, *, out_dir: Path,
    active: int = 64, scan_segments: int = 1024, block: int = 256,
    constants: ModelConstants | None = None,
) -> WcetReport:
    if active <= 0 or scan_segments < 0 or block <= 0:
        raise WcetError("active/block must be positive, scan_segments >= 0")
    return WcetReport(
        inputs=derive_inputs(tc, out_dir=out_dir),
        constants=constants or load_constants(tc.repo_root),
        active=active,
        scan_segments=scan_segments,
        block=block,
        sample_rate=parse_daisy_app_config(tc.repo_root)["DAISY_SAMPLE_RATE"],
    )
