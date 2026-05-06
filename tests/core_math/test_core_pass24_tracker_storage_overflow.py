#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_tracker_create_segment_storage_uses_checked_byte_counts() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "size_t thread_slots = 0;" in src
    assert "size_t thread_slots_bytes = 0;" in src
    assert "size_t init_seg_bytes = 0;" in src
    assert "spectral_size_mul((size_t)n_threads, (size_t)SPECTRAL_CACHE_LINE_STRIDE, &thread_slots)" in src
    assert "spectral_size_mul(thread_slots, sizeof(size_t), &thread_slots_bytes)" in src
    assert "spectral_size_mul(init_cap, sizeof(TrackSegment), &init_seg_bytes)" in src
    assert "spectral_aligned_alloc(thread_slots_bytes)" in src
    assert "spectral_aligned_alloc(init_seg_bytes)" in src
    assert "memset(tracker->seg_counts, 0, thread_slots_bytes)" in src
    assert "memset(tracker->seg_capacities, 0, thread_slots_bytes)" in src

    assert "(size_t)n_threads * SPECTRAL_CACHE_LINE_STRIDE * sizeof(size_t)" not in src
    assert "init_cap * sizeof(TrackSegment)" not in src

def test_tracker_growth_realloc_uses_checked_byte_counts() -> None:
    src = read("spectral_engine/analysis/spectral_peak_interp.c")

    assert '#include "spectral_utils.h"' in src
    assert "size_t new_bytes = 0;" in src
    assert "size_t copy_bytes = 0;" in src
    assert "spectral_size_mul(new_cap, sizeof(TrackSegment), &new_bytes)" in src
    assert "spectral_size_mul(count, sizeof(TrackSegment), &copy_bytes)" in src
    assert "spectral_aligned_alloc(new_bytes)" in src
    assert "memcpy(new_arr, tracker->seg_arrays[tid], copy_bytes)" in src

    assert "spectral_aligned_alloc(new_cap * sizeof(TrackSegment))" not in src
    assert "memcpy(new_arr, tracker->seg_arrays[tid], count * sizeof(TrackSegment))" not in src

def test_pass24_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS24.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "tracker segment-storage allocation overflow" in notes
    assert "growth path" in notes
    assert "pass 24 tracker create segment storage" in audit
    assert "pass 24 tracker segment growth" in audit
    assert "for pass_num in range(1, 25):" in audit
