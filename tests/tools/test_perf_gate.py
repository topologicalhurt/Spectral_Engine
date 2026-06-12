"""The embedded performance regression gate (maintainer mandate, pass 255).

Verifies the LIVE measurement stack against the frozen baseline
(tests/fixtures/m7_baseline.json) within the named tolerance bands, plus the
absolute set-in-stone capacity ceilings — the published deterministic
promises (M7_PERF_MODEL_PLAN capacity table). A failure names the exact
quantity that moved.

Regenerating the baseline (benchmark_workflow m7-baseline --generate) is a
deliberate re-signing of the performance contract: only for an intended
change, stated in the commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from spectral_tools.performance.embedded import expectations, toolchain  # noqa: E402


def _have(*need: str) -> bool:
    try:
        toolchain.discover(ROOT, need=frozenset(need))
    except toolchain.ToolchainError:
        return False
    return True


def test_tolerances_are_named_and_justified():
    for name, band in expectations.TOLERANCES.items():
        assert band["value"] > 0, name
        assert len(band["justification"]) > 40, f"{name} needs a real justification"
    # The stone scenarios must reference real published ceilings.
    assert len(expectations.STONE_SCENARIOS) >= 2
    for s in expectations.STONE_SCENARIOS:
        assert 0 < s["max_budget_fraction"] <= 1.0
    assert expectations.STONE_MAX_CYC_PER_VOICE_SAMPLE > 0


def test_baseline_fixture_is_frozen_and_complete():
    base = expectations.Baseline.load(ROOT).doc
    assert "GENERATED" in base["_comment"]
    assert set(base["kernels"]) == set(
        __import__("spectral_tools.performance.embedded.wcet",
                   fromlist=["SAMPLES_PER_ITER"]).SAMPLES_PER_ITER)
    assert base["counts"]["process_insns"] > 0
    assert base["counts"]["fixture_digest"]
    assert len(base["wcet_scenarios"]) == len(expectations.STONE_SCENARIOS)
    for s in base["wcet_scenarios"]:
        assert s["budget_fraction"] <= s["max_budget_fraction"], (
            "frozen baseline itself violates a stone ceiling — the published "
            "capacity table no longer holds")


def test_daisy_app_config_parses_from_c():
    from spectral_tools.performance.embedded.wcet import parse_daisy_app_config

    app = parse_daisy_app_config(ROOT)
    assert app["DAISY_SAMPLE_RATE"] in (44100, 48000, 96000)
    assert app["DAISY_AUDIO_BLOCK_SIZE"] > 0
    assert app["DAISY_MAX_ACTIVE"] > 0


@pytest.mark.skipif(not _have("mca", "qemu"),
                    reason="needs newlib arm-none-eabi-gcc + llvm-mca + qemu")
def test_perf_gate_live_stack_within_contract(tmp_path):
    tc = toolchain.discover(ROOT, need=frozenset({"mca", "qemu"}))
    fails = expectations.verify(tc, out_dir=tmp_path)
    assert not fails, "performance contract violated:\n" + "\n".join(fails)


def _fake_docs():
    scenario = {"active": 32, "scan_segments": 128, "block": 48,
                "wcet_cycles": 350000.0, "budget_cycles": 400000.0,
                "budget_fraction": 0.875, "max_budget_fraction": 1.0}
    doc = {
        "kernels": {"synth_core_m7/.L452": {"cycles_per_iter": 340.0,
                                            "insns_per_iter": 430,
                                            "samples_per_iter": 16.0}},
        "counts": {"process_insns": 3_762_455, "code_lines": 310, "data_lines": 4500},
        "wcet_scenarios": [scenario],
        "worst_cyc_per_voice_sample": 24.0,
    }
    import copy
    return doc, copy.deepcopy(doc)


def test_gate_compare_passes_on_identical_docs():
    base, live = _fake_docs()
    assert expectations.compare(base, live) == []


def test_gate_compare_fails_on_scenario_drift_not_silently():
    """Inline-audit finding (pass 256): zip() truncation silently dropped
    checks when the scenario sets drifted. The gate must NAME the drift."""
    base, live = _fake_docs()
    live["wcet_scenarios"] = []   # drifted set (e.g. edited STONE_SCENARIOS)
    fails = expectations.compare(base, live)
    assert any("scenario set drifted" in f for f in fails)


def test_gate_compare_catches_regressions_and_stone_breach():
    base, live = _fake_docs()
    live["kernels"]["synth_core_m7/.L452"]["cycles_per_iter"] = 340.0 * 1.2
    live["counts"]["process_insns"] = int(3_762_455 * 1.10)
    live["wcet_scenarios"][0]["budget_fraction"] = 1.05
    live["worst_cyc_per_voice_sample"] = 26.0
    fails = expectations.compare(base, live)
    assert any("cycles/iter" in f for f in fails)
    assert any("process insns" in f for f in fails)
    assert sum("STONE" in f for f in fails) == 2
