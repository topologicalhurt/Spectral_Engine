#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_spwt_loader_validates_exact_file_shape_before_payload_read() -> None:
    src = read("spectral_engine/core/spectral_wavetable.c")

    assert "wavetable_file_expected_bytes" in src
    assert "wavetable_file_payload_bytes" in src
    assert "spectral_fs_file_size(f, &file_size)" in src
    assert "file_size != (uint64_t)expected_file_bytes" in src
    assert "spectral_fs_read_exact(f, temp, payload_bytes, SPECTRAL_ERR_FILE_READ)" in src
    assert "spectral_fs_read(temp, sizeof(spectral_sample_t), hdr.size, f)" not in src
    assert "spectral_fs_read(temp, sizeof(float), hdr.size, f)" not in src
    assert "spectral_fs_read(temp, sizeof(int16_t), hdr.size, f)" not in src

    file_shape_idx = src.index("file_size != (uint64_t)expected_file_bytes")
    payload_read_idx = src.index("spectral_fs_read_exact(f, temp, payload_bytes")
    assert file_shape_idx < payload_read_idx

def test_spwt_loader_and_saver_validate_sample_finiteness() -> None:
    src = read("spectral_engine/core/spectral_wavetable.c")

    assert "wavetable_float_samples_finite" in src
    assert "wavetable_runtime_samples_valid" in src
    assert "!spectral_is_finite_f32(samples[i])" in src
    assert "wavetable_runtime_samples_valid(temp, hdr.size)" in src
    assert "wavetable_float_samples_finite(temp, hdr.size)" in src
    assert "wavetable_runtime_samples_valid(table->samples, SPECTRAL_WAVETABLE_SIZE)" in src

def test_spwt_header_metadata_is_validated() -> None:
    src = read("spectral_engine/core/spectral_wavetable.c")

    assert "hdr.timbre_id >= SPECTRAL_MAX_WAVETABLES" in src
    assert "hdr.size != SPECTRAL_WAVETABLE_SIZE" in src
    assert "hdr.format != WAVETABLE_FORMAT_FLOAT && hdr.format != WAVETABLE_FORMAT_Q15" in src

def test_spwt_saver_uses_exact_writes_and_checked_sample_bytes() -> None:
    src = read("spectral_engine/core/spectral_wavetable.c")

    assert "spectral_array_bytes(SPECTRAL_WAVETABLE_SIZE, sizeof(spectral_sample_t), &sample_bytes)" in src
    assert "spectral_fs_write_exact(f, &hdr, sizeof(hdr), SPECTRAL_ERR_FILE_WRITE)" in src
    assert "spectral_fs_write_exact(f, table->samples, sample_bytes, SPECTRAL_ERR_FILE_WRITE)" in src
    assert "spectral_fs_write(table->samples, sizeof(spectral_sample_t), SPECTRAL_WAVETABLE_SIZE, f)" not in src

def test_pass34_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS34.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "wavetable .spwt file shape and sample contract" in notes
    assert "before allocation" in notes
    assert "non-finite" in notes
    assert "pass 34 wavetable parser" in audit
    assert "for pass_num in range(1, 35):" in audit
