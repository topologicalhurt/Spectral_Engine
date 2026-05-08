#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_window_metric_helpers_validate_finite_input_samples() -> None:
    src = read("spectral_engine/core/spectral_windows.c")

    assert "#include <float.h>" in src
    assert "spectral_window_samples_valid" in src
    assert "!isfinite(window[i])" in src
    assert "!spectral_window_samples_valid(window, length)" in src

def test_window_sum_and_energy_reject_unrepresentable_accumulation() -> None:
    src = read("spectral_engine/core/spectral_windows.c")

    assert "sum > (double)FLT_MAX" in src
    assert "sum < -(double)FLT_MAX" in src
    assert "energy > (double)FLT_MAX" in src

def test_window_metrics_refuses_invalid_span_before_deriving_flags() -> None:
    src = read("spectral_engine/core/spectral_windows.c")

    assert "if (!spectral_window_samples_valid(window, length)) {" in src
    metrics_idx = src.index("if (!spectral_window_samples_valid(window, length)) {")
    sum_idx = src.index("metrics.sum = spectral_window_sum(window, length);")
    assert metrics_idx < sum_idx

def test_pass65_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS65.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "window metric finite-input contract" in notes
    assert "calibration" in notes
    assert "pass 65 window metrics" in audit
    assert "for pass_num in range(1, 66):" in audit
