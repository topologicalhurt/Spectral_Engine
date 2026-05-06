#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_raw_wavetable_loader_validates_exact_file_shape_and_samples() -> None:
    src = read("spectral_engine/core/spectral_wavetable.c")

    assert "spectral_fs_file_size(f, &file_size)" in src
    assert "file_size != (uint64_t)sample_bytes" in src
    assert "spectral_fs_read_exact(f, temp, sample_bytes, SPECTRAL_ERR_FILE_READ)" in src
    assert "wavetable_runtime_samples_valid(temp, SPECTRAL_WAVETABLE_SIZE)" in src
    assert "spectral_fs_read(temp, sizeof(spectral_sample_t), SPECTRAL_WAVETABLE_SIZE, f)" not in src

def test_hex_wavetable_loader_requires_eof_full_coverage_and_valid_samples() -> None:
    src = read("spectral_engine/core/spectral_wavetable.c")

    assert "int saw_eof = 0;" in src
    assert "saw_eof = 1;" in src
    assert "if (!saw_eof)" in src
    assert "covered_bytes != expected_bytes" in src
    assert "wavetable_runtime_samples_valid(temp_table, SPECTRAL_WAVETABLE_SIZE)" in src
    assert "data_len > expected_bytes - offset" in src
    assert "(size_t)address + data_len > expected_bytes" not in src
    assert "spectral_size_add(SPECTRAL_WAVETABLE_SIZE, 1u, &temp_table_count)" in src

def test_buffer_wavetable_loader_rejects_nonfinite_runtime_samples() -> None:
    src = read("spectral_engine/core/spectral_wavetable.c")

    assert "WavetableError spectral_wavetable_load_buffer" in src
    assert "wavetable_runtime_samples_valid(data, SPECTRAL_WAVETABLE_SIZE)" in src

def test_pass35_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS35.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "wavetable raw/hex/buffer ingestion contract" in notes
    assert "Intel HEX EOF" in notes
    assert "non-finite" in notes
    assert "pass 35 wavetable raw/hex" in audit
    assert "for pass_num in range(1, 36):" in audit
