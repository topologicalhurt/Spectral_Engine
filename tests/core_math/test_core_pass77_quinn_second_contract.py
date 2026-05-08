#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_quinn_second_uses_double_for_dot_products_and_corrections() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")

    assert "double mag0 = 0.0;" in src
    assert "double ap = 0.0, am = 0.0;" in src
    assert "double dp = 0.0, dm = 0.0;" in src
    assert "ap = ((double)xp.re * (double)x0.re" in src
    assert "dm2 > (double)FLT_MAX" in src
    assert "tau_p = spectral_peak_quinn_tau((float)dp2);" in src

def test_quinn_second_clamps_offset_before_float_narrowing() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")

    assert "double offset_d = 0.0;" in src
    assert "offset_d = 0.5 * (dp + dm)" in src
    assert "if (offset_d > 0.5)" in src
    assert "*out_offset = (float)offset_d;" in src

def test_pass77_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS77.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "Quinn second estimator finite-product contract" in notes
    assert "dp²" in notes
    assert "pass 77 Quinn second" in audit
    assert "for pass_num in range(1, 78):" in audit
