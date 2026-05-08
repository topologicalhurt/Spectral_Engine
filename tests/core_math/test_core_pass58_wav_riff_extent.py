#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_wav_peak_scrubber_parses_and_validates_riff_declared_extent() -> None:
    src = read("spectral_engine/core/spectral_out.c")

    assert "uint64_t riff_payload_size = 0;" in src
    assert "uint64_t riff_end = 0;" in src
    assert "riff_payload_size = (uint64_t)hdr[4]" in src
    assert "riff_payload_size < 4u || riff_payload_size > file_size - 8u" in src
    assert "riff_end = 8u + riff_payload_size;" in src

def test_wav_peak_scrubber_bounds_chunks_by_riff_end_not_physical_eof() -> None:
    src = read("spectral_engine/core/spectral_out.c")

    assert "riff_end - chunk_header_pos < (uint64_t)sizeof(chunk)" in src
    assert "(uint64_t)size > riff_end - data_pos" in src
    assert "next_chunk_pos > riff_end" in src
    assert "timestamp_pos > riff_end" in src

def test_pass58_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS58.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "WAV RIFF declared-extent contract" in notes
    assert "riff_end" in notes
    assert "pass 58 WAV RIFF" in audit
    assert "for pass_num in range(1, 59):" in audit
