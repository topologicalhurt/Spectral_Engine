#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_float_normalization_rejects_nonfinite_buffer_and_headroom() -> None:
    src = read("spectral_engine/core/spectral_out.c")

    assert "spectral_float_buffer_all_finite" in src
    assert "!spectral_is_finite_f32(buffer[i])" in src
    assert "!spectral_is_finite_f32(headroom) || headroom < 0.0f" in src
    assert "!spectral_float_buffer_all_finite(buffer, len)" in src

def test_float_normalization_validates_max_and_scale_before_applying() -> None:
    src = read("spectral_engine/core/spectral_out.c")

    assert "!spectral_is_finite_f32(max_amp) || max_amp < 0.0f" in src
    assert "float scale = headroom / max_amp;" in src
    assert "if (!spectral_is_finite_f32(scale)) return 0.0f;" in src

def test_pass52_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS52.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "float normalization finite-state contract" in notes
    assert "headroom" in notes
    assert "pass 52 float normalization" in audit
    assert "for pass_num in range(1, 53):" in audit
