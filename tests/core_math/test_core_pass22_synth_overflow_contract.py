#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_shared_synth_preflight_reports_output_byte_overflow() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")
    assert "out_len * elem_size overflow is a contract failure" in src
    assert "size_t preflight_out_bytes = 0;" in src
    assert "!spectral_size_mul(out_len, elem_size, &preflight_out_bytes)" in src
    assert "pf.error = SPECTRAL_ERR_OVERFLOW;" in src

    overflow_idx = src.index("!spectral_size_mul(out_len, elem_size, &preflight_out_bytes)")
    validate_idx = src.index("synth_validate_inputs(out_buffer, out_len, elem_size, sa, t_synth)")
    assert overflow_idx < validate_idx

def test_cpu_synth_zeroing_uses_validated_byte_count() -> None:
    src = read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")
    assert "size_t out_bytes = 0;" in src
    assert "!spectral_size_mul(out_len, elem_size, &out_bytes)" in src
    assert "return SPECTRAL_ERR_OVERFLOW;" in src
    assert "memset(out_buffer, 0, out_bytes);" in src
    assert "memset(out_buffer, 0, out_len * elem_size)" not in src

def test_pass22_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS22.md")
    audit = read("tools/core_audit/core_static_audit.py")
    assert "SPECTRAL_ERR_OVERFLOW" in notes
    assert "benign early exit" in notes
    assert "pass 22 synth preflight" in audit
