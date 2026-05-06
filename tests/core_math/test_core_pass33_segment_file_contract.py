#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_segment_file_metadata_is_validated_on_save_and_load() -> None:
    src = read("spectral_engine/core/spectral_segment_parser.c")

    assert "segment_file_metadata_valid_u32" in src
    assert "sr < (uint32_t)SPECTRAL_MIN_SAMPLE_RATE" in src
    assert "sr > (uint32_t)SPECTRAL_MAX_SAMPLE_RATE" in src
    assert "!spectral_is_finite_positive_f32(stretch)" in src
    assert "stretch > SPECTRAL_MAX_STRETCH" in src
    assert "!spectral_is_finite_f32(pitch)" in src
    assert "pitch < SPECTRAL_MIN_PITCH" in src
    assert "pitch > SPECTRAL_MAX_PITCH" in src
    assert "if (!segment_file_metadata_valid_u32((uint32_t)sr, stretch, pitch))" in src
    assert "if (!segment_file_metadata_valid_u32(hdr.sr, hdr.stretch, hdr.pitch))" in src

def test_segment_file_load_validates_file_shape_before_allocation() -> None:
    src = read("spectral_engine/core/spectral_segment_parser.c")

    assert "segment_file_expected_bytes" in src
    assert "spectral_size_add(sizeof(SegmentFileHeader), seg_bytes, out_bytes)" in src
    assert "spectral_fs_file_size(f, &file_size)" in src
    assert "file_size != (uint64_t)expected_file_bytes" in src
    assert "err = SPECTRAL_ERR_FILE_CORRUPT;" in src
    assert "spectral_malloc_array((size_t)hdr.count, sizeof(Segment))" in src
    assert "malloc(alloc_bytes)" not in src

    file_shape_idx = src.index("file_size != (uint64_t)expected_file_bytes")
    alloc_idx = src.index("loaded_segs = (Segment*)spectral_malloc_array")
    assert file_shape_idx < alloc_idx

def test_segment_file_save_refuses_corrupt_segments() -> None:
    src = read("spectral_engine/core/spectral_segment_parser.c")

    assert "segments_validate_all(sa->segs, sa->count, &bad_idx)" in src
    assert "Refusing to save corrupt segment data" in src

def test_pass33_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS33.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "segment file metadata and shape contract" in notes
    assert "before allocation" in notes
    assert "zero-count files" in notes
    assert "pass 33 segment file parser" in audit
    assert "for pass_num in range(1, 34):" in audit
