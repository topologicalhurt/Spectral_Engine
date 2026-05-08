#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_magnitude_parabolic_estimator_uses_double_quadratic_fit() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")

    assert "double denom = 0.0;" in src
    assert "double numer = 0.0;" in src
    assert "double offset_d = 0.0;" in src
    assert "denom = (double)left - 2.0 * (double)center + (double)right;" in src
    assert "numer = 0.5 * ((double)left - (double)right);" in src
    assert "offset_d = numer / denom;" in src
    assert "offset = 0.5f * (left - right) / denom;" not in src

def test_magnitude_parabolic_clamps_before_float_narrowing() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")

    assert "if (offset_d > 0.5)" in src
    assert "*out_offset = (float)offset_d;" in src

def test_pass78_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS78.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "magnitude-parabolic estimator finite-product contract" in notes
    assert "quadratic" in notes
    assert "pass 78 magnitude parabolic" in audit
    assert "for pass_num in range(1, 79):" in audit
