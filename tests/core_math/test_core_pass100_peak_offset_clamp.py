#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_peak_double_offset_clamping_has_single_helper() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")

    assert "spectral_peak_store_clamped_offset_d" in src
    assert src.count("return spectral_peak_store_clamped_offset_d(out_offset, offset_d);") >= 3
    assert "if (offset_d > 0.5) {\n        *out_offset = 0.5f;" not in src

def test_float_offset_clamp_remains_for_float_callback_policy() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")

    assert "static float spectral_peak_clamp_offset(float p)" in src
    assert "spectral_peak_clamp_offset(raw_offset)" in src

def test_pass100_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS100.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "peak offset clamp consolidation" in notes
    assert "one narrowing/clamp policy" in notes
    assert "pass 100 peak offset clamp" in audit
    assert "for pass_num in range(1, 101):" in audit
