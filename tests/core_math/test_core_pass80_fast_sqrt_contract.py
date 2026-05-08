#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_fast_sqrt_helpers_reject_nonfinite_inputs() -> None:
    src = read("spectral_engine/core/spectral_fast_math.c")

    assert "float fast_inv_sqrt(float x)" in src
    assert "float fast_sqrt(float x)" in src
    assert src.count("if (!(x > 0.0f) || !isfinite(x)) return 0.0f;") >= 2
    assert "if (x <= 0.0f) return 0.0f;" not in src

def test_fast_sqrt_approximation_branches_validate_result() -> None:
    src = read("spectral_engine/core/spectral_fast_math.c")

    assert "return isfinite(y) && y > 0.0f ? y : 0.0f;" in src

def test_pass80_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS80.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "fast sqrt finite-input contract" in notes
    assert "NaN" in notes
    assert "pass 80 fast sqrt" in audit
    assert "for pass_num in range(1, 81):" in audit
