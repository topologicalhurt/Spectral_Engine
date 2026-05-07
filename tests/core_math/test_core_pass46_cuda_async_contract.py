#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_cuda_async_copies_are_checked() -> None:
    src = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")

    assert "cudaError_t copy_err = cudaSuccess;" in src
    assert "copy_err = cudaMemcpyAsync" in src
    assert "copy_err != cudaSuccess" in src
    assert "SPECTRAL_LOG_ERROR_STDERR(\"CUDA async copy error: %s\"" in src

def test_cuda_event_records_and_stream_sync_are_checked() -> None:
    src = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")

    assert "cudaEventRecord(ev_start, g_cuda.stream) != cudaSuccess" in src
    assert "cudaEventRecord(ev_stop, g_cuda.stream) != cudaSuccess" in src
    assert "sync_err = cudaStreamSynchronize(g_cuda.stream)" in src
    assert "sync_err != cudaSuccess" in src
    assert "cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop) != cudaSuccess" in src

def test_cuda_kernel_launch_is_checked_before_copyback() -> None:
    src = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")

    kernel_idx = src.index("kernel_err = cudaGetLastError();")
    copyback_idx = src.index("copy_err = cudaMemcpyAsync(out_buffer, g_cuda.d_output")
    assert kernel_idx < copyback_idx

def test_pass46_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS46.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "CUDA async transfer and timing contract" in notes
    assert "stale output" in notes
    assert "pass 46 CUDA async" in audit
    assert "for pass_num in range(1, 47):" in audit
