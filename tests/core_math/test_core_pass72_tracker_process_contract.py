#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_tracker_process_checks_matrix_shape_and_global_frame_index_domain() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "size_t chunk_bins = 0;" in src
    assert "spectral_size_mul(chunk_n_frames, n_freqs, &chunk_bins)" in src
    assert "global_frame_offset > SIZE_MAX - (n_pairs - 1u)" in src
    assert "spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);" in src

def test_tracker_process_checks_t_hop_float_representability() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "double t_hop_d = (double)(global_frame_offset + t) * (double)hop_float;" in src
    assert "t_hop_d > (double)FLT_MAX" in src
    assert "const float t_hop = (float)t_hop_d;" in src
    assert "const float t_hop = (global_frame_offset + t) * hop_float;" not in src

def test_pass72_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS72.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "tracker process matrix and frame-time contract" in notes
    assert "chunk_n_frames" in notes
    assert "pass 72 tracker process" in audit
    assert "for pass_num in range(1, 73):" in audit
