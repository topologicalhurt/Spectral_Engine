#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_metal_zero_reference_tile_layout_returns_silence_before_dispatch() -> None:
    src = read("spectral_engine/synth/backends/gpu/metal/spectral_synth_metal.m")

    assert "if (td.total_refs == 0u)" in src
    assert "memset(out_buffer, 0, output_size);" in src
    assert "*t_synth = omp_get_wtime() - pf.start_time;" in src

    zero_idx = src.index("if (td.total_refs == 0u)")
    params_idx = src.index("return_err = gpu_synth_params_pack_checked")
    assert zero_idx < params_idx

def test_cuda_zero_reference_tile_layout_returns_silence_before_dispatch() -> None:
    src = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")

    assert "if (td.total_refs == 0u)" in src
    assert "sync_err = cudaStreamSynchronize(g_cuda.stream);" in src
    assert "memset(out_buffer, 0, out_size);" in src

    zero_idx = src.index("if (td.total_refs == 0u)")
    tile_ids_idx = src.index("spectral_array_bytes((size_t)td.total_refs")
    assert zero_idx < tile_ids_idx

def test_pass49_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS49.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "zero-reference GPU tile dispatch contract" in notes
    assert "silence" in notes
    assert "pass 49 zero-reference GPU" in audit
    assert "for pass_num in range(1, 50):" in audit
