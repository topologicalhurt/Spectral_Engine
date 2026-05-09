#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_tracker_frame_time_helper_lives_in_internal_header() -> None:
    header = read("spectral_engine/analysis/spectral_peak_track_internal.h")

    assert "spectral_tracker_frame_time_from_index" in header
    assert "t_hop_d = (double)frame_index * (double)hop_float;" in header
    assert "t_hop_d > (double)FLT_MAX" in header

def test_fused_and_tracker_process_use_frame_time_helper() -> None:
    fused = read("spectral_engine/analysis/spectral_analysis_fused.c")
    track = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "spectral_tracker_frame_time_from_index(pair, (float)hop, &t_hop)" in fused
    assert "spectral_tracker_frame_time_from_index(global_frame_offset + t, hop_float, &t_hop)" in track
    assert "double t_hop_d = (double)pair * (double)hop;" not in fused
    assert "double t_hop_d = (double)(global_frame_offset + t) * (double)hop_float;" not in track

def test_pass99_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS99.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "tracker frame-time helper consolidation" in notes
    assert "same tracker time coordinate" in notes
    assert "pass 99 tracker frame time" in audit
    assert "for pass_num in range(1, 100):" in audit
