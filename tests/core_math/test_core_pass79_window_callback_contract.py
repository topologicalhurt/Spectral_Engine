#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_current_window_peak_callback_validates_left_right_before_dispatch() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")

    assert "left = input->magsq_row[center_bin - 1u];" in src
    assert "right = input->magsq_row[center_bin + 1u];" in src
    assert "!spectral_peak_finite_nonnegative(left)" in src
    assert "peak_magsq(left, center_magsq, right, offset, &candidate)" in src

def test_next_window_peak_callback_validates_neighborhood_and_offset() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")

    assert "float left = input->next_magsq_row[next_bin - 1u];" in src
    assert "float right = input->next_magsq_row[next_bin + 1u];" in src
    assert "!isfinite(next_offset)" in src
    assert "!spectral_peak_finite_nonnegative(right)" in src
    assert "peak_magsq(left, center, right, next_offset, &candidate)" in src

def test_pass79_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS79.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "peak window callback neighborhood contract" in notes
    assert "custom" in notes
    assert "pass 79 peak window callback" in audit
    assert "for pass_num in range(1, 80):" in audit
