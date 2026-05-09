#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_frame_context_constructor_exists() -> None:
    header = read("spectral_engine/analysis/spectral_peak_track_internal.h")

    assert "spectral_tracker_frame_context_init" in header
    assert "spectral_tracker_frame_time_from_index(frame_index, hop_float, &t_hop)" in header
    assert "ctx->row = row;" in header
    assert "ctx->can_start_new = can_start_new;" in header

def test_fused_path_uses_frame_context_constructor() -> None:
    fused = read("spectral_engine/analysis/spectral_analysis_fused.c")

    assert "spectral_tracker_frame_context_init(&fctx" in fused
    assert "fctx.row = " not in fused
    assert "fctx.threshsq = " not in fused

def test_pass113_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS113.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "tracker frame-context constructor" in notes
    assert "SpectralFrameContext" in notes
    assert "pass 113 tracker frame context" in audit
    assert "for pass_num in range(1, 114):" in audit
