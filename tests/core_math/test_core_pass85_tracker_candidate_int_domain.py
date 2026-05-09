#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_candidate_validation_checks_cf_before_best_next_int_narrowing() -> None:
    src = read("spectral_engine/analysis/spectral_peak_interp.c")

    assert "#include <limits.h>" in src
    assert "cf > (size_t)INT_MAX" in src
    assert "*out_best_next = (int)cf + best_idx - 1;" in src

    check_idx = src.index("cf > (size_t)INT_MAX")
    assign_idx = src.index("*out_best_next = (int)cf + best_idx - 1;")
    assert check_idx < assign_idx

def test_pass85_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS85.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "tracker candidate best-next integer-domain contract" in notes
    assert "best_next_bin" in notes
    assert "pass 85 tracker candidate int" in audit
    assert "for pass_num in range(1, 86):" in audit
