#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_cpu_reduce_uses_checked_driver_byte_count_not_raw_product() -> None:
    src = read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")

    assert "thread_buffers_reduce_float(const ThreadBuffers* tb, float* out_buffer,\n                                                 size_t out_len, size_t out_bytes)" in src
    assert "memcpy(out_buffer, tb->bufs[0], out_bytes)" in src
    assert "out_len * sizeof(float)" not in src
    assert "tb->buf_size < out_bytes" in src

def test_cpu_driver_propagates_reduction_error_and_zeroes_output() -> None:
    src = read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")

    assert "SpectralError reduce_err = SPECTRAL_OK;" in src
    assert "reduce_err = reduce_fn(&tb, out_buffer, out_len, out_bytes);" in src
    assert "if (reduce_err != SPECTRAL_OK)" in src
    assert "memset(out_buffer, 0, out_bytes);" in src
    assert "return reduce_err;" in src

def test_reduce_function_signature_carries_checked_byte_count() -> None:
    src = read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")

    assert "typedef SpectralError (*ReduceFn)(const ThreadBuffers* tb, void* out_buffer, size_t out_len, size_t out_bytes);" in src
    assert "return thread_buffers_reduce_native(tb, (spectral_sample_t*)out, len, out_bytes);" in src

def test_pass51_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS51.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "CPU synth reduction byte-count contract" in notes
    assert "out_len * sizeof(float)" in notes
    assert "pass 51 CPU synth reduction" in audit
    assert "for pass_num in range(1, 52):" in audit
