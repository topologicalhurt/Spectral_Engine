"""F-stream step 1: measure CMSIS-DSP's Q31 inverse RFFT on the counts rig.

Turns the P6/D4 structural crossover bound (osc-bank vs IFFT synthesis at
~3-12 dense partials) into a MEASURED number: exact retired instructions per
inverse-RFFT call of the real CMSIS-DSP code (v1.16.2, vendored) compiled
with the Daisy flags and executed under qemu-system-arm.

Provenance: instruction counts are [measured: qemu-tcg]; converting to
cycles uses the validated IPC band from the kernel (insns/cycles ∈ ~[1, 2]
on this in-order 2-wide core with zero-wait data) — reported as a band, not
a point, until llvm-mca runs over the FFT loops (F-stream step 2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.process import run
from .counts import CountsError, _build_plugin, _parse_counts, _run_once
from .toolchain import NATIVE_DIR, Toolchain

# CMSIS-DSP sources needed by arm_rfft_q31 (inverse, with bit-reversal).
CMSIS_DSP_SOURCES = (
    "Source/TransformFunctions/arm_rfft_q31.c",
    "Source/TransformFunctions/arm_rfft_init_q31.c",
    "Source/TransformFunctions/arm_cfft_q31.c",
    "Source/TransformFunctions/arm_cfft_init_q31.c",
    "Source/TransformFunctions/arm_cfft_radix4_q31.c",
    "Source/TransformFunctions/arm_bitreversal.c",
    "Source/TransformFunctions/arm_bitreversal2.c",
    "Source/CommonTables/arm_common_tables.c",
    "Source/CommonTables/arm_const_structs.c",
    "Source/BasicMathFunctions/arm_shift_q31.c",   # inverse-rfft output scaling
)

NM_LINE_RE = re.compile(r"^([0-9a-f]+)\s+([0-9a-f]+)\s+[TtWw]\s+(\S+)", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class FftProbeResult:
    fft_n: int
    iters: int
    insns_per_fft: float
    load_bytes_per_fft: float
    store_bytes_per_fft: float
    checksum: str
    compute_symbols: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "provenance": "measured: qemu-tcg retired instructions of vendored "
                          "CMSIS-DSP v1.16.2 under the Daisy flags; cycles need "
                          "the IPC band (~[1,2]) until mca runs the FFT loops",
            "fft_n": self.fft_n,
            "iterations": self.iters,
            "insns_per_fft": round(self.insns_per_fft, 0),
            "cycles_per_fft_band": [round(self.insns_per_fft / 2.0, 0),
                                    round(self.insns_per_fft * 2.0, 0)],
            "load_bytes_per_fft": round(self.load_bytes_per_fft, 0),
            "store_bytes_per_fft": round(self.store_bytes_per_fft, 0),
            "checksum": self.checksum,
            "compute_symbols": list(self.compute_symbols),
        }


def _build_fft_elf(tc: Toolchain, *, out_dir: Path, fft_n: int, iters: int) -> Path:
    dsp = tc.repo_root / "third_party/CMSIS-DSP"
    core = tc.repo_root / "third_party/CMSIS_6/CMSIS/Core/Include"
    if not (dsp / "Include/arm_math.h").is_file():
        raise CountsError("third_party/CMSIS-DSP not materialized — "
                          "run: python -m spectral_tools.vendor submodule sync")
    if not (core / "cmsis_compiler.h").is_file():
        raise CountsError("third_party/CMSIS_6 core headers missing — "
                          "run: python -m spectral_tools.vendor submodule sync")

    elf = out_dir / "fft_probe.elf"
    sources = [
        str(NATIVE_DIR / "qemu" / "startup.c"),
        str(NATIVE_DIR / "qemu" / "fft_main.c"),
        *(str(dsp / rel) for rel in CMSIS_DSP_SOURCES),
    ]
    cmd = [
        tc.arm_gcc, *tc.target_flags, "-ffreestanding", "-ffunction-sections",
        f"-DFFT_N={fft_n}", f"-DFFT_ITERS={iters}",
        "-I", str(dsp / "Include"), "-I", str(dsp / "PrivateInclude"),
        "-I", str(core),
        "-T", str(NATIVE_DIR / "qemu" / "mps2_an500.ld"),
        "-nostartfiles", "-lc_nano", "-lm", "-Wl,--gc-sections",
        *sources, "-o", str(elf),
    ]
    result = run(cmd, cwd=tc.repo_root, check=False)
    if result.returncode != 0:
        raise CountsError(f"fft probe build failed:\n{result.stderr[-3000:]}")
    return elf


def _fft_symbol_ranges(tc: Toolchain, elf: Path) -> list[tuple[str, int, int]]:
    """Every arm_* code symbol, init symbols separated out by the caller."""
    result = run([tc.arm_nm, "-S", str(elf)], cwd=tc.repo_root, check=False)
    if result.returncode != 0:
        raise CountsError(f"nm failed:\n{result.stderr}")
    ranges = []
    for m in NM_LINE_RE.finditer(result.stdout):
        addr, size, name = int(m.group(1), 16), int(m.group(2), 16), m.group(3)
        if name.startswith("arm_") and size > 0:
            ranges.append((name, addr, addr + size))
    if not ranges:
        raise CountsError("no arm_* symbols in the fft probe ELF — gc-sections drift?")
    return sorted(ranges)


def probe(tc: Toolchain, *, out_dir: Path, fft_n: int = 512,
          iters: int = 8) -> FftProbeResult:
    if tc.qemu_system_arm is None:
        raise CountsError("qemu-system-arm unavailable")
    out_dir.mkdir(parents=True, exist_ok=True)
    elf = _build_fft_elf(tc, out_dir=out_dir, fft_n=fft_n, iters=iters)
    ranges = _fft_symbol_ranges(tc, elf)
    plugin = _build_plugin(tc, out_dir)

    table, checksum, rendered = _run_once(
        tc, elf, plugin, ranges, out_dir / "fft_counts.txt")
    if rendered != iters:
        raise CountsError(f"fft probe rendered {rendered}, expected {iters}")
    parsed, _, _ = _parse_counts(table)

    insns = loads = stores = 0
    compute = []
    for r in parsed:
        if not r.name.startswith("arm_"):
            continue
        if "_init" in r.name:
            continue   # one-time setup, not per-FFT cost
        if r.insns == 0:
            continue
        compute.append(r.name)
        insns += r.insns
        loads += r.load_bytes
        stores += r.store_bytes

    if insns == 0:
        raise CountsError("no instructions attributed to arm_* compute symbols")
    return FftProbeResult(
        fft_n=fft_n, iters=iters,
        insns_per_fft=insns / iters,
        load_bytes_per_fft=loads / iters,
        store_bytes_per_fft=stores / iters,
        checksum=checksum,
        compute_symbols=tuple(sorted(compute)),
    )
