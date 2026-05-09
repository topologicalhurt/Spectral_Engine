#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_tracker_emit_validates_final_segment_fields_before_storage_write() -> None:
    src = read("spectral_engine/analysis/spectral_peak_interp.c")

    assert "spectral_tracker_emitted_segment_valid" in src
    assert "!isfinite(t_hop) || t_hop < 0.0f" in src
    assert "!isfinite(hop_float) || hop_float <= 0.0f" in src
    assert "!isfinite(phase)" in src
    assert "!isfinite(estimate->omega) || estimate->omega < 0.0f" in src
    assert "spectral_tracker_set_error(tracker, SPECTRAL_ERR_PARAM);" in src

    validate_idx = src.index("if (!spectral_tracker_emitted_segment_valid(")
    storage_idx = src.index("count = tracker->seg_counts[tid * SPECTRAL_CACHE_LINE_STRIDE];")
    assert validate_idx < storage_idx

def test_pass81_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS81.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "tracker emitted-segment validity contract" in notes
    assert "producer boundary" in notes
    assert "pass 81 tracker emitted segment" in audit
    assert "for pass_num in range(1, 82):" in audit
