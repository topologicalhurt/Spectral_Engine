#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_full_direct_hash_does_not_narrow_uint64_file_length_to_size_t() -> None:
    src = read("spectral_engine/core/spectral_hash_xx32_xx3.c")

    assert "#include <stdint.h>" in src
    assert "uint64_t total_len_u64 = 0;" in src
    assert "total_len_u64 = end_pos - start_pos;" in src
    assert "start_pos > (uint64_t)INT64_MAX" in src
    assert "total_len_u64 > (uint64_t)SIZE_MAX" in src
    assert "return spectral_hash_method_consume_file_stream_impl(method, file);" in src
    assert "total_len = (size_t)total_len_u64;" in src
    assert "total_len = (size_t)(end_pos - start_pos)" not in src

def test_unimplemented_mmap_hash_method_is_not_advertised_available() -> None:
    src = read("spectral_engine/core/spectral_hash_xx32_xx3.c")
    header = read("spectral_engine/core/spectral_hash_xx32_xx3.h")

    assert "FULL_MMAP is registered for future support but is not advertised as available" in src
    assert "[SPECTRAL_HASH_FILE_FULL_MMAP]" in src
    assert ".available = 0" in src
    assert "available=1" in header
    assert "unavailable" in header

def test_pass32_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS32.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "hash file full-direct size and capability contract" in notes
    assert "uint64_t" in notes
    assert "size_t" in notes
    assert "pass 32 hash full-direct" in audit
    assert "for pass_num in range(1, 33):" in audit
