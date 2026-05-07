#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_gpu_tile_fill_checks_atomic_cursor_bounds_before_write() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "int fill_overflow = 0;" in src
    assert "pos >= tile_ranges[tt].count" in src
    assert "pos > total_refs - tile_ranges[tt].start" in src
    assert "write_index >= total_refs" in src
    assert "#pragma omp atomic write" in src

def test_gpu_tile_fill_verifies_cursors_match_counting_pass() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "tile_cursors[t] != tile_counts[t]" in src
    assert "return_err = SPECTRAL_ERR_FILE_CORRUPT;" in src
    assert "pass 45" not in src.lower()  # keep source comments contract-focused, not pass-number focused

def test_pass45_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS45.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "GPU tile fill cursor contract" in notes
    assert "counting pass" in notes
    assert "write pass" in notes
    assert "pass 45 GPU tile fill" in audit
    assert "for pass_num in range(1, 46):" in audit
