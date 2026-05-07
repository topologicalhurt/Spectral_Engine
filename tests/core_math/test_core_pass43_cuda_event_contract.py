#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_cuda_events_are_declared_for_cleanup_scope_and_checked() -> None:
    src = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")

    assert "cudaEvent_t ev_start = NULL;" in src
    assert "cudaEvent_t ev_stop = NULL;" in src
    assert "cudaEventCreate(&ev_start) != cudaSuccess" in src
    assert "cudaEventCreate(&ev_stop) != cudaSuccess" in src
    assert "return_err = SPECTRAL_ERR_GPU_INIT;" in src

def test_cuda_events_are_destroyed_on_all_cleanup_paths() -> None:
    src = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")

    assert "if (ev_start) cudaEventDestroy(ev_start);" in src
    assert "if (ev_stop) cudaEventDestroy(ev_stop);" in src
    assert "cudaEventDestroy(ev_start);\n    cudaEventDestroy(ev_stop);\ncleanup:" not in src

def test_pass43_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS43.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "CUDA event lifecycle contract" in notes
    assert "leak" in notes
    assert "pass 43 CUDA events" in audit
    assert "for pass_num in range(1, 44):" in audit
