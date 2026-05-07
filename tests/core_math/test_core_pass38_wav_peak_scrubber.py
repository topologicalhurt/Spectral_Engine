#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_wav_peak_scrubber_checks_chunk_extents_before_seek() -> None:
    src = read("spectral_engine/core/spectral_out.c")

    assert "spectral_wav_seek_u64_checked" in src
    assert "offset > (uint64_t)INT64_MAX" in src
    assert "spectral_fs_file_size(f, &file_size)" in src
    assert "file_size - chunk_header_pos < (uint64_t)sizeof(chunk)" in src
    assert "(uint64_t)size > file_size - data_pos" in src
    assert "next_chunk_pos > file_size" in src
    assert "spectral_wav_seek_u64_checked(f, next_chunk_pos" in src

def test_wav_peak_scrubber_uses_exact_io_for_chunk_headers_and_timestamp() -> None:
    src = read("spectral_engine/core/spectral_out.c")

    assert "spectral_fs_read_exact(f, hdr, sizeof(hdr), SPECTRAL_ERR_FILE_READ)" in src
    assert "spectral_fs_read_exact(f, chunk, sizeof(chunk), SPECTRAL_ERR_FILE_READ)" in src
    assert "spectral_fs_write_exact(f, zero4, sizeof(zero4), SPECTRAL_ERR_FILE_WRITE)" in src
    assert "spectral_fs_read(hdr, 1, sizeof(hdr), f)" not in src
    assert "spectral_fs_read(chunk, 1, sizeof(chunk), f)" not in src
    assert "spectral_fs_write(zero4, 1, sizeof(zero4), f)" not in src

def test_wav_peak_timestamp_offset_is_checked_before_write() -> None:
    src = read("spectral_engine/core/spectral_out.c")

    assert "uint64_t timestamp_pos = data_pos + 4u;" in src
    assert "sizeof(zero4) > (size_t)(file_size - timestamp_pos)" in src
    assert "spectral_wav_seek_u64_checked(f, timestamp_pos, SPECTRAL_ERR_FILE_WRITE)" in src

def test_pass38_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS38.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "WAV PEAK scrubber chunk extent contract" in notes
    assert "RIFF chunk" in notes
    assert "before seeking" in notes
    assert "pass 38 WAV PEAK scrubber" in audit
    assert "for pass_num in range(1, 39):" in audit
