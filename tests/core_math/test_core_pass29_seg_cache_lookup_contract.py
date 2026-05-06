#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_segment_cache_lookup_validates_entry_scalar_metadata_before_casting() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert "seg_cache_entry_metadata_valid" in src
    assert "e->sample_rate < (uint32_t)SPECTRAL_MIN_SAMPLE_RATE" in src
    assert "e->sample_rate > (uint32_t)SPECTRAL_MAX_SAMPLE_RATE" in src
    assert "spectral_is_finite_positive_f32(e->stretch)" in src
    assert "e->stretch > SPECTRAL_MAX_STRETCH" in src
    assert "spectral_is_finite_f32(e->pitch)" in src
    assert "e->pitch < SPECTRAL_MIN_PITCH" in src
    assert "e->pitch > SPECTRAL_MAX_PITCH" in src
    assert "(uint64_t)(size_t)e->output_length != e->output_length" in src
    assert "if (!seg_cache_entry_metadata_valid(e))" in src

    validate_idx = src.index("if (!seg_cache_entry_metadata_valid(e))")
    sample_assign_idx = src.index("result->sample_rate = (int)e->sample_rate;")
    assert validate_idx < sample_assign_idx

def test_segment_cache_lookup_validates_data_extent_before_mapping_or_heap_hit() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert "seg_cache_validate_data_extent" in src
    assert "spectral_seg_cache_fs_data_file_size(cache_dir, &data_file_size)" in src
    assert "total_data_bytes_u64 > UINT64_MAX - data_offset" in src
    assert "data_file_size < data_offset + total_data_bytes_u64" in src
    assert "SPECTRAL_ERR_FILE_CORRUPT" in src
    assert "err = seg_cache_validate_data_extent(cache_dir, e->data_offset, total_data_bytes);" in src

    extent_idx = src.index("err = seg_cache_validate_data_extent(cache_dir, e->data_offset, total_data_bytes);")
    mmap_idx = src.index("spectral_seg_cache_fs_data_map_ro(")
    heap_idx = src.index("Segment* segs = (Segment*)spectral_malloc_array(")
    assert extent_idx < mmap_idx
    assert extent_idx < heap_idx

def test_segment_cache_tile_metadata_consistency_is_validated_on_lookup() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert "e->seg_count == 0 && (e->tile_count != 0 || e->tile_total_refs != 0)" in src
    assert "(e->tile_count == 0) != (e->tile_total_refs == 0)" in src

def test_pass29_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS29.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "segment cache lookup metadata and data extent contract" in notes
    assert "before narrowing" in notes
    assert "before mapping" in notes
    assert "pass 29 segment cache lookup" in audit
    assert "for pass_num in range(1, 30):" in audit
