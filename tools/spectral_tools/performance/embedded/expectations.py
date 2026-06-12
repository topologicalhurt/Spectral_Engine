"""Centralized performance expectations: ONE generated baseline, ONE gate.

The rule: testing constants must not float around the
codebase — they are generated from the live measurement stack into a single
frozen fixture, and every tolerance lives HERE with a name and a
justification. This mirrors the resource-hash generate/verify pattern:

  python -m spectral_tools.testing.benchmark_workflow m7-baseline --generate
      runs the live stack (census + counts + wcet scenarios) and freezes the
      numbers into tests/fixtures/m7_baseline.json. Regenerating is a
      DELIBERATE act (like re-signing a golden): do it only when an intended
      change moves the numbers, and say so in the commit.

  python -m spectral_tools.testing.benchmark_workflow m7-baseline (verify)
      re-runs the live stack and compares against the frozen baseline within
      the named tolerance bands below, PLUS the absolute set-in-stone
      ceilings — the deterministic capacity promises the engine must keep.

The pytest gate (tests/tools/test_perf_gate.py) runs verify, so a kernel or
toolchain change that erodes performance fails CI with the exact quantity
that moved, not a vague slowdown.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.utils import find_repo_root
from .codegen import codegen_report
from .counts import measure
from .memory_model import load_constants
from .toolchain import Toolchain
from .wcet import SAMPLES_PER_ITER, WcetReport, derive_inputs, parse_daisy_app_config

BASELINE_RELPATH = "tests/fixtures/m7_baseline.json"

# --- the tolerance table: every band named, justified, defined ONCE ---------
TOLERANCES: dict[str, dict[str, Any]] = {
    "modeled_cycles_rel": {
        "value": 0.10,
        "justification": "llvm-mca CortexM7Model drift across llvm releases; "
                         "P3 validated body throughput to <=1% of hand-derived "
                         "timing, so a 10% move means the shipped model changed "
                         "materially and the validation must be re-run",
    },
    "measured_insns_rel": {
        "value": 0.05,
        "justification": "arm-none-eabi-gcc codegen drift across releases was "
                         "measured at 0.03% (15.2.0 -> 15.2.1); 5% absorbs a "
                         "major-version bump while still catching a real "
                         "kernel regression",
    },
    "wcet_rel": {
        "value": 0.15,
        "justification": "compound of the two bands above plus residual "
                         "scaling; a 15% WCET move with unchanged inputs means "
                         "an engine change shifted the workload shape",
    },
    "working_set_lines_abs": {
        "value": 64,
        "justification": "32B-line working sets move with layout/padding; 64 "
                         "lines (2 KB) of slack covers linker noise while "
                         "catching a data-structure growth regression",
    },
}

# --- the set-in-stone ceilings (the deterministic capacity promises) --------
# These are ABSOLUTE gates, independent of the frozen baseline: the engine
# must always be able to keep the published capacity table
# (docs/core_audit/M7_PERF_MODEL_PLAN.md). Budgets derive from the C SSOTs
# at gate time — nothing here hardcodes a clock or sample rate.
STONE_SCENARIOS: tuple[dict[str, Any], ...] = (
    # Daisy default callback: 32 voices, 128-segment scan, 48-sample block
    # must fit the real-time budget (published guarantee: 88% of budget).
    {"active": 32, "scan_segments": 128, "block": 48, "max_budget_fraction": 1.0},
    # Batch path: the BSP active cap (128) at 256-sample blocks must fit
    # (published guarantee: 89% of budget).
    {"active": 128, "scan_segments": 1024, "block": 256, "max_budget_fraction": 1.0},
)
# The validated synthesis rate must not regress past this ceiling
# (published: worst kernel 24.0 cyc/voice-sample incl. back-edge bias).
STONE_MAX_CYC_PER_VOICE_SAMPLE = 25.0


class ExpectationsError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Baseline:
    doc: dict[str, Any]

    @classmethod
    def load(cls, repo_root: Path) -> "Baseline":
        path = repo_root / BASELINE_RELPATH
        try:
            return cls(doc=json.loads(path.read_text(encoding="utf-8")))
        except OSError as exc:
            raise ExpectationsError(
                f"no frozen baseline at {BASELINE_RELPATH} — run "
                "benchmark_workflow m7-baseline --generate (deliberate act)"
            ) from exc


def _live_numbers(tc: Toolchain, *, out_dir: Path) -> dict[str, Any]:
    """One stack pass producing every gated quantity."""
    report = codegen_report(tc, out_dir=out_dir)
    if report.failed_regions:
        raise ExpectationsError(f"mca regions failed: {report.failed_regions}")
    kernels = {}
    for loop in report.loops:
        key = f"{loop.kernel}/{loop.label}"
        if key in SAMPLES_PER_ITER:
            kernels[key] = {
                "cycles_per_iter": loop.cycles_per_iter,
                "insns_per_iter": loop.instructions_per_iter,
                "samples_per_iter": SAMPLES_PER_ITER[key],
            }

    counts = measure(tc, out_dir=out_dir, verify_reproducible=False)
    process = counts.range("spectral_arm32_process")
    if process is None:
        raise ExpectationsError("spectral_arm32_process missing from counts")

    constants = load_constants(tc.repo_root)
    inputs = derive_inputs(tc, out_dir=out_dir)
    app = parse_daisy_app_config(tc.repo_root)

    scenarios = []
    for s in STONE_SCENARIOS:
        rep = WcetReport(inputs=inputs, constants=constants,
                         active=s["active"], scan_segments=s["scan_segments"],
                         block=s["block"], sample_rate=app["DAISY_SAMPLE_RATE"])
        cpu_hz = constants.defines["SPECTRAL_DAISY_CPU_HZ"]
        budget = cpu_hz / app["DAISY_SAMPLE_RATE"] * s["block"]
        scenarios.append({
            "active": s["active"], "scan_segments": s["scan_segments"],
            "block": s["block"],
            "wcet_cycles": round(rep.wcet_cycles, 0),
            "budget_cycles": round(budget, 0),
            "budget_fraction": round(rep.wcet_cycles / budget, 4),
            "max_budget_fraction": s["max_budget_fraction"],
        })

    return {
        "kernels": kernels,
        "counts": {
            "process_insns": process.insns,
            "process_load_bytes": process.load_bytes,
            "process_store_bytes": process.store_bytes,
            "code_lines": counts.region_lines32.get("code", 0),
            "data_lines": (counts.region_lines32.get("ssram23", 0)
                           + counts.region_lines32.get("bulk60", 0)),
            "fixture_digest": counts.fixture_digest,
            "checksum": counts.checksum,
        },
        "derived_memory": {
            "linefill_row_hit": constants.line_fill_cycles(True),
            "linefill_row_miss": constants.line_fill_cycles(False),
            "writeback": constants.writeback_cycles(),
        },
        "worst_cyc_per_voice_sample": round(inputs.worst_cyc_per_voice_sample, 3),
        "wcet_scenarios": scenarios,
        "daisy_app_config": app,
    }


def generate(tc: Toolchain, *, out_dir: Path) -> Path:
    """Freeze the live numbers into the baseline fixture (deliberate act)."""
    live = _live_numbers(tc, out_dir=out_dir)
    doc = {
        "_comment": "GENERATED by benchmark_workflow m7-baseline --generate. "
                    "Regenerating re-signs the performance contract — do it "
                    "only for an intended change and say so in the commit. "
                    "Tolerances + stone ceilings live in expectations.py.",
        "meta": {
            "arm_gcc": tc.arm_gcc,
            "llvm_mca": tc.llvm_mca,
            "tolerances": {k: v["value"] for k, v in TOLERANCES.items()},
        },
        **live,
    }
    path = tc.repo_root / BASELINE_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _rel(a: float, b: float) -> float:
    return abs(a - b) / b if b else float("inf")


def verify(tc: Toolchain, *, out_dir: Path) -> list[str]:
    """Compare live stack vs the frozen baseline and the stone ceilings.
    Returns failure strings (empty = gate passes)."""
    return compare(Baseline.load(tc.repo_root).doc,
                   _live_numbers(tc, out_dir=out_dir))


def compare(base: dict[str, Any], live: dict[str, Any]) -> list[str]:
    """Pure gate logic (unit-testable without the toolchain)."""
    fails: list[str] = []

    tol_cyc = TOLERANCES["modeled_cycles_rel"]["value"]
    tol_insn = TOLERANCES["measured_insns_rel"]["value"]
    tol_wcet = TOLERANCES["wcet_rel"]["value"]
    tol_lines = TOLERANCES["working_set_lines_abs"]["value"]

    for key, b in base["kernels"].items():
        l = live["kernels"].get(key)
        if l is None:
            fails.append(f"kernel {key} vanished from codegen")
            continue
        if _rel(l["cycles_per_iter"], b["cycles_per_iter"]) > tol_cyc:
            fails.append(f"{key} cycles/iter {l['cycles_per_iter']} vs "
                         f"baseline {b['cycles_per_iter']} (> {tol_cyc:.0%})")
        if _rel(l["insns_per_iter"], b["insns_per_iter"]) > tol_insn:
            fails.append(f"{key} insns/iter {l['insns_per_iter']} vs "
                         f"baseline {b['insns_per_iter']} (> {tol_insn:.0%})")

    if _rel(live["counts"]["process_insns"], base["counts"]["process_insns"]) > tol_insn:
        fails.append(f"process insns {live['counts']['process_insns']} vs "
                     f"baseline {base['counts']['process_insns']} (> {tol_insn:.0%})")
    for lines_key in ("code_lines", "data_lines"):
        if abs(live["counts"][lines_key] - base["counts"][lines_key]) > tol_lines:
            fails.append(f"{lines_key} {live['counts'][lines_key]} vs baseline "
                         f"{base['counts'][lines_key]} (> {tol_lines} lines)")

    # Scenario sets must MATCH before comparing: a drifted STONE_SCENARIOS
    # table (length or parameters) must fail the gate, not silently drop
    # checks via zip truncation.
    base_params = [(s["active"], s["scan_segments"], s["block"])
                   for s in base["wcet_scenarios"]]
    live_params = [(s["active"], s["scan_segments"], s["block"])
                   for s in live["wcet_scenarios"]]
    if base_params != live_params:
        fails.append(f"scenario set drifted: baseline {base_params} vs "
                     f"live {live_params} — regenerate the baseline deliberately")
    for b_s, l_s in zip(base["wcet_scenarios"], live["wcet_scenarios"]):
        label = f"wcet({l_s['active']}a/{l_s['scan_segments']}s/{l_s['block']}b)"
        if _rel(l_s["wcet_cycles"], b_s["wcet_cycles"]) > tol_wcet:
            fails.append(f"{label} {l_s['wcet_cycles']} vs baseline "
                         f"{b_s['wcet_cycles']} (> {tol_wcet:.0%})")
        # The stone ceiling: independent of the baseline.
        if l_s["budget_fraction"] > l_s["max_budget_fraction"]:
            fails.append(f"STONE: {label} = {l_s['budget_fraction']:.0%} of "
                         f"budget exceeds the published capacity ceiling "
                         f"{l_s['max_budget_fraction']:.0%}")

    if live["worst_cyc_per_voice_sample"] > STONE_MAX_CYC_PER_VOICE_SAMPLE:
        fails.append(f"STONE: worst cyc/voice-sample "
                     f"{live['worst_cyc_per_voice_sample']} exceeds "
                     f"{STONE_MAX_CYC_PER_VOICE_SAMPLE}")

    return fails
