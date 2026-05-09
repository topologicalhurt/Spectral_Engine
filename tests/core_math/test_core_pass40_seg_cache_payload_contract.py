#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_segment_cache_store_validates_segment_payload_before_append() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert "spectral_segment_payload_valid" in src
    assert "spectral_segment_array_payload_valid" in src
    assert "!spectral_is_finite_f32(s->omega) || s->omega < 0.0f" in src
    assert "if (!spectral_segment_array_payload_valid(sa->segs, sa->count, NULL))" in src

    validation_idx = src.index("if (!spectral_segment_array_payload_valid(sa->segs, sa->count, NULL))")
    append_idx = src.index("spectral_seg_cache_fs_data_append_begin(cache_dir, &w)")
    assert validation_idx < append_idx

def test_segment_cache_lookup_validates_mmap_segment_and_gpu_payload() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert "spectral_segment_gpu_array_matches_segments" in src
    assert "SegmentGpu expected = spectral_segment_pack_gpu(&segs[i]);" in src
    assert "memcmp(&expected, &gpu_segs[i], sizeof(expected)) != 0" in src
    assert "SPECTRAL_ERR_FILE_CORRUPT" in src
    assert "spectral_seg_cache_fs_data_unmap(&mv)" in src
    assert "Segment cache payload corrupt" in src

    validate_idx = src.index("spectral_segment_gpu_array_matches_segments(mapped_segments, mapped_gpu_segs")
    result_idx = src.index("result->segments.segs = mapped_segments;")
    assert validate_idx < result_idx

def test_segment_cache_lookup_validates_heap_payload_after_endian_swap() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert "if (!spectral_segment_array_payload_valid(segs, e->seg_count, NULL))" in src
    assert "return SPECTRAL_ERR_FILE_CORRUPT;" in src

def test_segment_cache_tile_layout_references_are_validated_before_publication_or_store() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert "spectral_gpu_tile_layout_words_valid" in src
    assert "start != running_refs" in src
    assert "segment_id >= seg_count" in src
    assert "Segment cache tile references invalid" in src
    assert "has_tile_blob &&" in src
    assert "!spectral_gpu_tile_layout_words_valid(tile_ranges," in src

def test_pass40_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS40.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "segment cache payload validity contract" in notes
    assert "mmap-backed" in notes
    assert "GPU-segment" in notes
    assert "tile references" in notes
    assert "pass 40 segment cache payload" in audit
    assert "for pass_num in range(1, 41):" in audit
