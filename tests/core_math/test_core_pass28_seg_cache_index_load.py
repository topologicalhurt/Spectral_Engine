#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_index_load_validates_count_derived_file_size_before_allocating() -> None:
    src = read("spectral_engine/core/spectral_seg_cache_fs.c")

    assert "spectral_array_bytes((size_t)hdr.count, sizeof(SpectralSegCacheEntry), &entries_bytes)" in src
    assert "spectral_size_add(sizeof(SpectralSegCacheHeader), entries_bytes, &expected_index_bytes)" in src
    assert "spectral_fs_file_size(f, &file_size)" in src
    assert "file_size != (uint64_t)expected_index_bytes" in src
    assert "err = SPECTRAL_ERR_FILE_CORRUPT;" in src

    size_check = src.index("file_size != (uint64_t)expected_index_bytes")
    alloc = src.index("entries = (SpectralSegCacheEntry*)spectral_malloc_array(")
    assert size_check < alloc

def test_zero_count_index_file_must_be_exact_header_size() -> None:
    src = read("spectral_engine/core/spectral_seg_cache_fs.c")

    assert "if (hdr.count == 0) {" in src
    assert "file_size != (uint64_t)sizeof(SpectralSegCacheHeader)" in src

def test_index_load_no_longer_allocates_before_overflow_check() -> None:
    src = read("spectral_engine/core/spectral_seg_cache_fs.c")
    count = src.count("spectral_array_bytes((size_t)hdr.count, sizeof(SpectralSegCacheEntry), &entries_bytes)")
    assert count >= 1

    array_bytes_idx = src.index("spectral_array_bytes((size_t)hdr.count, sizeof(SpectralSegCacheEntry), &entries_bytes)")
    alloc_idx = src.index("entries = (SpectralSegCacheEntry*)spectral_malloc_array(")
    assert array_bytes_idx < alloc_idx

def test_pass28_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS28.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "segment cache index load count/file-size contract" in notes
    assert "before allocating" in notes
    assert "SPECTRAL_ERR_FILE_CORRUPT" in notes
    assert "pass 28 segment cache index load" in audit
    assert "for pass_num in range(1, 29):" in audit
