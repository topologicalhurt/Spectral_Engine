#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_segment_cache_index_insert_count_is_checked_before_allocation() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert "size_t new_count = 0;" in src
    assert "size_t new_entries_bytes = 0;" in src
    assert "spectral_size_add((size_t)count, 1u, &new_count)" in src
    assert "new_count > (size_t)UINT32_MAX" in src
    assert "spectral_array_bytes(new_count, sizeof(SpectralSegCacheEntry), &new_entries_bytes)" in src
    assert "spectral_malloc_array(\n                new_count, sizeof(SpectralSegCacheEntry))" in src
    assert "(size_t)(count + 1)" not in src
    assert "count++;" not in src
    assert "count = (uint32_t)new_count;" in src

def test_segment_cache_index_insert_copy_sizes_are_checked() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert "size_t prefix_bytes = 0;" in src
    assert "size_t suffix_count = 0;" in src
    assert "size_t suffix_bytes = 0;" in src
    assert "spectral_array_bytes((size_t)ins, sizeof(SpectralSegCacheEntry), &prefix_bytes)" in src
    assert "spectral_array_bytes(suffix_count, sizeof(SpectralSegCacheEntry), &suffix_bytes)" in src
    assert "memcpy(new_entries, entries, prefix_bytes)" in src
    assert "memcpy(&new_entries[(size_t)ins + 1u], &entries[ins], suffix_bytes)" in src
    assert "ins * sizeof(SpectralSegCacheEntry)" not in src
    assert "(count - ins) * sizeof(SpectralSegCacheEntry)" not in src

def test_pass26_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS26.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "segment cache index insertion overflow" in notes
    assert "UINT32_MAX" in notes
    assert "persistent on-disk entry count" in notes
    assert "pass 26 segment cache index insertion" in audit
    assert "for pass_num in range(1, 27):" in audit
