#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_segment_cache_key_guards_float_to_int_conversions() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert "#include <limits.h>" in src
    assert "seg_cache_scale_to_i32" in src
    assert "scaled < (double)INT_MIN" in src
    assert "scaled > (double)INT_MAX" in src
    assert "!seg_cache_scale_to_i32(db_thresh, 10.0, &db_thresh_i)" in src
    assert "!seg_cache_scale_to_i32(start_sec, 1000.0, &start_ms_i)" in src
    assert "!seg_cache_scale_to_i32(end_sec, 1000.0, &end_ms_i)" in src
    assert "!seg_cache_scale_to_i32(stretch, 1000000.0, &stretch_ppm_i)" in src

    assert "(int)(db_thresh * 10.0f)" not in src
    assert "(int)(start_sec * 1000.0f)" not in src
    assert "(int)(end_sec * 1000.0f)" not in src
    assert "(int)(stretch * 1000000.0f)" not in src

def test_segment_cache_key_rejects_truncated_format_strings() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert "len <= 0 || (size_t)len >= sizeof(buf)" in src
    assert "len = (int)(sizeof(buf) - 1)" not in src

def test_segment_cache_store_validates_metadata_before_append() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert "seg_cache_store_metadata_valid" in src
    assert "key == 0u" in src
    assert "sample_rate < SPECTRAL_MIN_SAMPLE_RATE" in src
    assert "sample_rate > SPECTRAL_MAX_SAMPLE_RATE" in src
    assert "!spectral_is_finite_positive_f32(stretch)" in src
    assert "stretch > SPECTRAL_MAX_STRETCH" in src
    assert "!spectral_is_finite_f32(pitch)" in src
    assert "pitch < SPECTRAL_MIN_PITCH" in src
    assert "pitch > SPECTRAL_MAX_PITCH" in src
    assert "(size_t)output_length_u64 != output_length" in src
    assert "if (!seg_cache_store_metadata_valid(key, sa, sample_rate, stretch, pitch, output_length))" in src

    helper_idx = src.index("seg_cache_store_metadata_valid")
    append_idx = src.index("spectral_seg_cache_fs_data_append_begin")
    assert helper_idx < append_idx

def test_pass30_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS30.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "segment cache key and store scalar metadata contract" in notes
    assert "undefined behavior" in notes
    assert "truncated parameter string" in notes
    assert "pass 30 segment cache key" in audit
    assert "pass 30 segment cache store" in audit
    assert "for pass_num in range(1, 31):" in audit
