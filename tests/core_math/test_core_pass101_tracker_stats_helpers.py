#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_tracker_stats_use_counter_and_time_accumulation_helpers() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "spectral_tracker_accumulate_counter_checked" in src
    assert "spectral_tracker_accumulate_time_checked" in src
    assert "spectral_tracker_accumulate_counter_checked(tracker, &tracker->total_pairs" in src
    assert "spectral_tracker_accumulate_time_checked(tracker, &tracker->process_time_total" in src

def test_tracker_stats_repetition_is_removed_from_accumulate_function() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")
    fn = src[src.index("void spectral_tracker_accumulate_stats("):src.index("void spectral_segment_array_free")]

    assert "next_time = tracker->process_time_total + local_track_time;" not in fn
    assert "tracker->total_pairs = next_u64;" not in fn

def test_pass101_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS101.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "tracker stats accumulation helper consolidation" in notes
    assert "one reusable implementation" in notes
    assert "pass 101 tracker stats helpers" in audit
    assert "for pass_num in range(1, 102):" in audit
