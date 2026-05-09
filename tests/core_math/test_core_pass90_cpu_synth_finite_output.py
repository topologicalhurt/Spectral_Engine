#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_cpu_float_reduction_validates_finite_output_postcondition() -> None:
    src = read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")

    assert "spectral_f32_span_finite" in src
    assert "!spectral_is_finite_f32(out_buffer[i])" in src
    assert "if (!spectral_f32_span_finite(out_buffer, out_len))" in src
    assert "return SPECTRAL_ERR_PARAM;" in src

    check_idx = src.index("if (!spectral_f32_span_finite(out_buffer, out_len))")
    return_idx = src.index("return SPECTRAL_OK;", check_idx)
    assert check_idx < return_idx

def test_cpu_driver_existing_reducer_error_path_zeroes_output() -> None:
    src = read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")

    assert "reduce_err = reduce_fn(&tb, out_buffer, out_len, out_bytes);" in src
    assert "if (reduce_err != SPECTRAL_OK)" in src
    assert "memset(out_buffer, 0, out_bytes);" in src

def test_pass90_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS90.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "CPU synth finite-output postcondition" in notes
    assert "SPECTRAL_OK" in notes
    assert "pass 90 CPU synth finite output" in audit
    assert "for pass_num in range(1, 91):" in audit
