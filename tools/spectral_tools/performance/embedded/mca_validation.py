"""Layer 2 validation: llvm-mca CortexM7Model vs hand-derived M7 timing (P3).

A microbench set of hand-analyzable Thumb-2 loops whose expected
cycles/iteration follow from a small, citable fact base: the Cortex-M7 TRM's
pipeline-structure statements plus the two community DWT-measured timing
references (ARM publishes no per-instruction cycle table for the M7 — that
absence is itself sourced). Each case runs through the SAME llvm-mca
invocation path as the kernel loop analysis (codegen.mca_region) in two
variants:

  body  — the loop body only. The hand expectation is EXACT under the
          documented issue rules; the body delta is the validation signal.
  loop  — body + ``subs``/``bne`` back-edge, the shape the kernel analysis
          measures. Real M7 folds a predicted-taken back-edge into the
          destination (~0-1 cycles); the mca model has no branch folding and
          charges ~2. The difference is reported as the known systematic
          back-edge divergence, NOT mixed into the body validation.

Provenance discipline: expected values are [derived: TRM + measured-community],
mca outputs are [modeled: llvm-mca/CortexM7Model]. A validation delta is the
distance between two models of the machine — with no board, neither side is a
measurement of hardware (M7_PERF_MODEL_PLAN, fidelity contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .codegen import CodegenError, mca_region
from .toolchain import Toolchain

# The fact base behind every expected value. Verification status is explicit:
# "direct" = the cited source was fetched and the quote confirmed first-party;
# "corroborated" = cross-checked against independent secondary write-ups, the
# primary quote itself not re-confirmed.
FACTS: dict[str, dict[str, str]] = {
    "trm-dual-issue": {
        "claim": "In-order super-scalar pipeline; many instructions can be "
                 "dual-issued, including load/load and load/store pairs "
                 "(multiple memory interfaces).",
        "source": "Arm Cortex-M7 TRM, DDI 0489F sec 1.1",
        "verified": "corroborated",
    },
    "trm-two-alus": {
        "claim": "DPU has two ALUs (one SIMD-capable) and a 6-read/4-write "
                 "register file sized for dual-issue, with extensive "
                 "forwarding to minimise interlocks.",
        "source": "Arm Cortex-M7 TRM, DDI 0489F sec 1.2.1",
        "verified": "corroborated",
    },
    "no-official-cycle-table": {
        "claim": "Unlike M0/M3/M4, ARM does not publish per-instruction cycle "
                 "counts for the M7; community measurement is the only "
                 "public per-instruction source.",
        "source": "quinapalus.com/cm7cycles.html (intro)",
        "verified": "direct",
    },
    "q-alu-pairs": {
        "claim": "Independent ALU pairs dual-issue: eor+eor and adc+adc = 1 "
                 "cycle; mutual carry dependency (adcs;adcs) = 2 cycles.",
        "source": "quinapalus.com/cm7cycles.html (measured, STM32H730@480MHz, caches on)",
        "verified": "direct",
    },
    "q-mac-family": {
        "claim": "MUL/UMULL throughput 1/cycle, latency 2 from multiplier "
                 "inputs, 1 from addend; umlal accumulator chain = 1 cycle; "
                 "umull/umlal dual-issue with an independent ALU op; two "
                 "multiplies never dual-issue (single MAC pipe).",
        "source": "quinapalus.com/cm7cycles.html (measured)",
        "verified": "direct",
    },
    "q-loads": {
        "claim": "ldr standalone = 1 cycle; ldr+ldr measured 2 (no pair in "
                 "that addressing pattern); ldr dual-issues with independent "
                 "ALU/MUL/MLA (= 1 cycle); data-to-address dependency = 3.",
        "source": "quinapalus.com/cm7cycles.html (measured)",
        "verified": "direct",
    },
    "j-banked-loads": {
        "claim": "Two loads (DTCM or D-cache) dual-issue iff they target "
                 "different (even/odd word) banks; sequential-word streams "
                 "alternate banks. 'Slippery' (alignment-sensitive).",
        "source": "github.com/jnk0le/random pipeline cycle test/doc/cortex_m7.md (measured, STM32H743)",
        "verified": "direct",
    },
    "j-load-use": {
        "claim": "Word loads are consumed by the late ALU in the NEXT cycle "
                 "(no visible load-use stall in normal ALU consumption); "
                 "MUL/MAC cannot consume an ldr result from the previous "
                 "cycle except via the accumulator operand.",
        "source": "github.com/jnk0le/random .../cortex_m7.md (measured)",
        "verified": "direct",
    },
    "j-mac-chain": {
        "claim": "All multiply-accumulates chain on the accumulator "
                 "dependency every cycle (incl. the DSP dual-half family); "
                 "+1 cycle result latency vs ALU ops; MACs cannot dual-issue "
                 "with each other or with stores.",
        "source": "github.com/jnk0le/random .../cortex_m7.md (measured)",
        "verified": "direct",
    },
    "j-stores": {
        "claim": "Two stores cannot dual-issue (single store channel); a "
                 "store can dual-issue with a load; a store can consume any "
                 "same-cycle result.",
        "source": "github.com/jnk0le/random .../cortex_m7.md (measured)",
        "verified": "direct",
    },
    "j-branch-fold": {
        "claim": "A predicted taken branch can dual-issue with one "
                 "instruction at the destination (back-edge ~0-1 cycles in a "
                 "steady loop, alignment-sensitive); not-taken branches "
                 "dual-issue with the following instruction; misprediction "
                 "8 cycles (6 with subs flag-forwarding placed >=3 earlier).",
        "source": "github.com/jnk0le/random .../cortex_m7.md (measured)",
        "verified": "direct",
    },
}

LOOP_TAIL = ("\tsubs r8, r8, #1", "\tbne .Ltop")

# Real-M7 expected cost of the subs+bne back-edge per iteration on top of the
# body, from j-branch-fold + q-alu-pairs (subs pairs into the stream, the
# predicted-taken bne folds into the destination; both alignment-"slippery").
LOOP_OVERHEAD_HAND = (0.0, 2.0)


@dataclass(frozen=True, slots=True)
class ValidationCase:
    name: str
    description: str
    body: tuple[str, ...]
    expected_body_cycles: float   # exact hand derivation, steady state
    derivation: str               # the hand analysis, citing FACTS ids
    facts: tuple[str, ...]        # FACTS ids the derivation rests on


CASES: tuple[ValidationCase, ...] = (
    ValidationCase(
        name="alu_independent_8",
        description="8 independent single-cycle ALU adds (no flag writes)",
        body=tuple(f"\tadd r{i}, r{i}, #1" for i in range(8)),
        expected_body_cycles=4.0,
        derivation="Two ALUs issue independent single-cycle ops in pairs "
                   "[trm-two-alus, q-alu-pairs]: 8 ops / 2 per cycle = 4.",
        facts=("trm-two-alus", "q-alu-pairs"),
    ),
    ValidationCase(
        name="alu_dependent_8",
        description="8 serially dependent ALU adds (register chain)",
        body=tuple("\tadd r0, r0, #1" for _ in range(8)),
        expected_body_cycles=8.0,
        derivation="A register-dependent chain serializes to 1 op/cycle "
                   "(forwarding removes stalls but not order) [q-alu-pairs "
                   "mutual-dependency rows]: 8 ops = 8 cycles.",
        facts=("q-alu-pairs",),
    ),
    ValidationCase(
        name="load_independent_8",
        description="8 independent word loads from sequential addresses",
        body=tuple(f"\tldr r{i}, [r9, #{4 * i}]" for i in range(8)),
        expected_body_cycles=4.0,
        derivation="Load/load pairs dual-issue [trm-dual-issue] when they hit "
                   "different even/odd word banks [j-banked-loads]; "
                   "sequential words alternate banks: 8 loads / 2 = 4. "
                   "Caveat: bank pairing is measured 'slippery'; the "
                   "non-banked floor is 8 [q-loads].",
        facts=("trm-dual-issue", "j-banked-loads", "q-loads"),
    ),
    ValidationCase(
        name="load_use_4",
        description="4x (word load -> dependent ALU add) interleaved",
        body=tuple(
            line
            for i in range(4)
            for line in (f"\tldr r{2 * i}, [r9, #{8 * i}]",
                         f"\tadd r{2 * i + 1}, r{2 * i}, #1")
        ),
        expected_body_cycles=4.0,
        derivation="A word load's result is consumable by the (late) ALU the "
                   "very next cycle [j-load-use], and each add pairs with the "
                   "next, independent ldr [q-loads ldr+adc=1]: steady state 1 "
                   "cycle per ldr+add pair = 4.",
        facts=("j-load-use", "q-loads"),
    ),
    ValidationCase(
        name="mul_independent_4",
        description="4 independent 64-bit multiplies (smull)",
        body=tuple(f"\tsmull r{2 * i}, r{2 * i + 1}, r10, r11" for i in range(4)),
        expected_body_cycles=4.0,
        derivation="One MAC pipe: multiplies never dual-issue with each "
                   "other, throughput 1/cycle [q-mac-family]: 4 muls = 4.",
        facts=("q-mac-family",),
    ),
    ValidationCase(
        name="smlal_chain_4",
        description="4 smlal accumulating into the same 64-bit accumulator",
        body=tuple("\tsmlal r0, r1, r10, r11" for _ in range(4)),
        expected_body_cycles=4.0,
        derivation="MACs chain on the accumulator dependency every cycle "
                   "[q-mac-family umlal=1, j-mac-chain]: 4 MACs = 4. This is "
                   "the spectral_coupled_step accumulation shape.",
        facts=("q-mac-family", "j-mac-chain"),
    ),
    ValidationCase(
        name="smlald_chain_4",
        description="4 dual-16-bit MACs (smlald) into one 64-bit accumulator",
        body=tuple("\tsmlald r0, r1, r10, r11" for _ in range(4)),
        expected_body_cycles=4.0,
        derivation="The DSP dual-half MAC family chains like all MACs, "
                   "1/cycle on the accumulator [j-mac-chain]: 4 = 4. This is "
                   "the synth_core_pair_m7 dual-voice shape.",
        facts=("j-mac-chain",),
    ),
    ValidationCase(
        name="store_stream_4",
        description="4 independent word stores",
        body=tuple(f"\tstr r{i}, [r9, #{4 * i}]" for i in range(4)),
        expected_body_cycles=4.0,
        derivation="Single store channel: two stores never dual-issue "
                   "[j-stores]: 4 stores = 4 cycles.",
        facts=("j-stores",),
    ),
    ValidationCase(
        name="load_store_pairs_2",
        description="2x (independent word load + word store)",
        body=(
            "\tldr r0, [r9]",
            "\tstr r2, [r10]",
            "\tldr r1, [r9, #8]",
            "\tstr r3, [r10, #8]",
        ),
        expected_body_cycles=2.0,
        derivation="Load/store pairs dual-issue (TRM names this pair class "
                   "explicitly) [trm-dual-issue, j-stores]: 2 pairs = 2.",
        facts=("trm-dual-issue", "j-stores"),
    ),
)


@dataclass(frozen=True, slots=True)
class CaseResult:
    case: ValidationCase
    mca_body_cycles: float
    mca_loop_cycles: float

    @property
    def body_delta_rel(self) -> float:
        """(mca - hand) / hand on the body — the validation signal."""
        return (self.mca_body_cycles - self.case.expected_body_cycles) \
            / self.case.expected_body_cycles

    @property
    def mca_loop_overhead(self) -> float:
        """Cycles mca charges for the subs+bne back-edge per iteration."""
        return self.mca_loop_cycles - self.mca_body_cycles

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.name,
            "description": self.case.description,
            "expected_body_cycles": self.case.expected_body_cycles,
            "expected_provenance": "derived: TRM DDI 0489F + measured-community",
            "mca_body_cycles": self.mca_body_cycles,
            "mca_loop_cycles": self.mca_loop_cycles,
            "mca_provenance": "modeled: llvm-mca/CortexM7Model",
            "body_delta_rel": round(self.body_delta_rel, 4),
            "mca_loop_overhead_cycles": round(self.mca_loop_overhead, 2),
            "hand_loop_overhead_range": list(LOOP_OVERHEAD_HAND),
            "derivation": self.case.derivation,
            "facts": list(self.case.facts),
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    results: tuple[CaseResult, ...]
    iterations: int

    @property
    def max_abs_body_delta(self) -> float:
        return max(abs(r.body_delta_rel) for r in self.results)

    @property
    def mean_backedge_divergence(self) -> float:
        """Mean excess of mca's back-edge cost over the hand upper bound."""
        excess = [max(0.0, r.mca_loop_overhead - LOOP_OVERHEAD_HAND[1])
                  for r in self.results]
        return sum(excess) / len(excess)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provenance": "validation: llvm-mca/CortexM7Model vs hand-derived "
                          "(TRM + measured-community); no hardware — both "
                          "sides are models",
            "iterations": self.iterations,
            "cases": [r.as_dict() for r in self.results],
            "summary": {
                "max_abs_body_delta_rel": round(self.max_abs_body_delta, 4),
                "mean_backedge_divergence_cycles": round(self.mean_backedge_divergence, 2),
                "interpretation": (
                    "body deltas validate steady-state issue/latency modeling; "
                    "the back-edge divergence is a known systematic (the model "
                    "has no branch folding; real M7 folds predicted-taken "
                    "back-edges [j-branch-fold]) and bounds the per-loop "
                    "overestimate of kernel cycles/iteration"
                ),
            },
            "facts": {fid: dict(f) for fid, f in FACTS.items()},
        }


def validate(tc: Toolchain, *, out_dir: Path, iterations: int = 100) -> ValidationReport:
    """Run the validation set; raises CodegenError if any region fails (a
    case that silently vanishes would fake a clean validation)."""
    case_ids = {fid for case in CASES for fid in case.facts}
    missing = case_ids - FACTS.keys()
    if missing:
        raise CodegenError(f"validation cases cite unknown facts: {sorted(missing)}")

    results: list[CaseResult] = []
    for case in CASES:
        variants: dict[str, float] = {}
        for variant, instructions in (
            ("body", case.body),
            ("loop", (*case.body, *LOOP_TAIL)),
        ):
            analyses = mca_region(
                tc,
                name=f"{case.name}/{variant}",
                instructions=instructions,
                out_dir=out_dir,
                file_stem=f"validate_{case.name}_{variant}",
                iterations=iterations,
            )
            if not analyses:
                raise CodegenError(f"mca produced no analysis for {case.name}/{variant}")
            variants[variant] = analyses[0].cycles_per_iter
        results.append(
            CaseResult(
                case=case,
                mca_body_cycles=variants["body"],
                mca_loop_cycles=variants["loop"],
            )
        )
    return ValidationReport(results=tuple(results), iterations=iterations)
