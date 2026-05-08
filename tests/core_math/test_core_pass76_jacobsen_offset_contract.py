#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_jacobsen_complex_offset_uses_checked_double_quotient_products() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")

    assert "double num_re = 0.0, num_im = 0.0;" in src
    assert "double den_mag2 = 0.0;" in src
    assert "double offset_d = 0.0;" in src
    assert "den_mag2 = den_re * den_re + den_im * den_im;" in src
    assert "offset_d = (num_re * den_re + num_im * den_im) / den_mag2;" in src
    assert "if (!isfinite(offset_d)) return 0;" in src
    assert "offset = (num_re * den_re + num_im * den_im) / den_mag2;" not in src

def test_jacobsen_offset_clamps_before_float_narrowing() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")

    assert "if (offset_d > 0.5)" in src
    assert "*out_offset = (float)offset_d;" in src

def test_pass76_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS76.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "Jacobsen/Candan complex-offset finite-product contract" in notes
    assert "Candan" in notes
    assert "pass 76 Jacobsen complex" in audit
    assert "for pass_num in range(1, 77):" in audit
