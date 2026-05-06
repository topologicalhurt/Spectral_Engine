#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_segment_cache_append_blob_lengths_are_checked_once() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert "int has_tile_blob = (tile_count > 0u && tile_total_refs > 0u &&" in src
    assert "size_t seg_bytes = 0;" in src
    assert "size_t gpu_seg_bytes = 0;" in src
    assert "size_t tile_ranges_bytes = 0;" in src
    assert "size_t tile_refs_bytes = 0;" in src
    assert "spectral_array_bytes((size_t)sa->count, sizeof(Segment), &seg_bytes)" in src
    assert "spectral_array_bytes((size_t)sa->count, sizeof(SegmentGpu), &gpu_seg_bytes)" in src
    assert "spectral_array_bytes((size_t)tile_count, sizeof(uint32_t) * 2u, &tile_ranges_bytes)" in src
    assert "spectral_array_bytes((size_t)tile_total_refs, sizeof(uint32_t), &tile_refs_bytes)" in src

def test_segment_cache_append_writes_use_checked_lengths() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert "spectral_seg_cache_fs_data_append_write(&w, scratch, seg_bytes)" in src
    assert "spectral_seg_cache_fs_data_append_write(&w, scratch, gpu_seg_bytes)" in src
    assert "spectral_seg_cache_fs_data_append_write(&w, ranges_scratch, tile_ranges_bytes)" in src
    assert "spectral_seg_cache_fs_data_append_write(&w, refs_scratch, tile_refs_bytes)" in src
    assert "spectral_seg_cache_fs_data_append_write(&w, tile_ranges, tile_ranges_bytes)" in src
    assert "spectral_seg_cache_fs_data_append_write(&w, tile_segment_ids, tile_refs_bytes)" in src

    assert "(size_t)sa->count * sizeof(Segment)" not in src
    assert "(size_t)sa->count * sizeof(SegmentGpu)" not in src
    assert "(size_t)tile_count * sizeof(uint32_t) * 2u" not in src
    assert "(size_t)tile_total_refs * sizeof(uint32_t)" not in src

def test_segment_cache_tile_metadata_matches_written_blob_contract() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert "uint32_t tc = has_tile_blob ? tile_count : 0;" in src
    assert "uint32_t tr = has_tile_blob ? tile_total_refs : 0;" in src
    assert "if (has_tile_blob)" in src
    assert "(tile_ranges && tile_segment_ids) ? tile_count : 0" not in src

def test_pass27_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS27.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "segment cache append blob byte-count contract" in notes
    assert "metadata must match bytes actually written" in notes
    assert "pass 27 segment cache append" in audit
    assert "for pass_num in range(1, 28):" in audit
