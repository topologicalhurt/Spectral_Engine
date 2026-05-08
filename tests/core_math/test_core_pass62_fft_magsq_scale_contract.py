#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_fft_magsq_scaling_uses_checked_finite_product_helper() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_fft.c")

    assert "#include <float.h>" in src
    assert "spectral_fft_scaled_magsq" in src
    assert "scaled > (double)FLT_MAX" in src
    assert "magsq[0] = spectral_fft_scaled_magsq(magsq[0], endpoint_scale);" in src
    assert "magsq[i] = spectral_fft_scaled_magsq(magsq[i], positive_scale);" in src

def test_fft_trackable_max_ignores_nonfinite_bins() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_fft.c")

    assert "if (isfinite(magsq[i]) && magsq[i] > max_magsq)" in src

def test_pass62_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS62.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "FFT magnitude scaling finite-output contract" in notes
    assert "FLT_MAX" in notes
    assert "pass 62 FFT magnitude scaling" in audit
    assert "for pass_num in range(1, 63):" in audit
