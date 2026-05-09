#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_fused_scratch_rows_have_single_owner_api() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_fused.c")

    assert "typedef struct SpectralFusedScratchRows" in src
    assert "spectral_fused_scratch_rows_alloc" in src
    assert "spectral_fused_scratch_rows_free" in src
    assert "spectral_fused_scratch_rows_rotate" in src

def test_fused_worker_uses_scratch_rows_object() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_fused.c")

    assert "SpectralFusedScratchRows rows = {0};" in src
    assert "spectral_fused_scratch_rows_alloc(&rows, n_freqs_f32_bytes)" in src
    assert "fctx.row = rows.row_curr;" in src
    assert "fctx.next_row = rows.row_next;" in src
    assert "spectral_fused_scratch_rows_rotate(&rows);" in src
    assert "spectral_fused_scratch_rows_free(&rows);" in src

def test_pass94_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS94.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "fused-analysis scratch row ownership" in notes
    assert "rotate" in notes
    assert "pass 94 fused scratch rows" in audit
    assert "for pass_num in range(1, 95):" in audit
