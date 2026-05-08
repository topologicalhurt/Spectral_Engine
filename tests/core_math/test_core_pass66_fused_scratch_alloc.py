#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_fused_analysis_precomputes_checked_nfreqs_row_bytes() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_fused.c")

    assert "size_t n_freqs_f32_bytes = 0;" in src
    assert "spectral_array_bytes(n_freqs, sizeof(float), &n_freqs_f32_bytes)" in src
    assert "spectral_aligned_alloc(n_freqs_f32_bytes)" in src
    assert src.count("spectral_aligned_alloc(n_freqs_f32_bytes)") >= 5

def test_fused_analysis_no_raw_nfreqs_float_allocation_products_remain() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_fused.c")

    assert "n_freqs * sizeof(float)" not in src

def test_pass66_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS66.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "fused-analysis scratch allocation byte-count contract" in notes
    assert "n_freqs_f32_bytes" in notes
    assert "pass 66 fused analysis scratch" in audit
    assert "for pass_num in range(1, 67):" in audit
