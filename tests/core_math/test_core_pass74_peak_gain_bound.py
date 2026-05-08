#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_peak_amplitude_gain_bound_uses_checked_double_product() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")

    assert "double gain_limit = 0.0;" in src
    assert "gain_limit = (double)center_magsq * (double)max_gain;" in src
    assert "!isfinite(gain_limit)" in src
    assert "return (double)peak_magsq <= gain_limit;" in src
    assert "peak_magsq <= center_magsq * max_gain" not in src

def test_pass74_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS74.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "peak amplitude gain-bound finite-product contract" in notes
    assert "max_gain" in notes
    assert "pass 74 peak amplitude gain" in audit
    assert "for pass_num in range(1, 75):" in audit
