#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_gpu_params_pack_has_checked_boundary_function() -> None:
    header = read("spectral_engine/core/spectral_synth_internal.h")

    assert '#include "spectral_utils.h"' in header
    assert "gpu_synth_params_pack_checked" in header
    assert "sp->out_len > (size_t)UINT32_MAX" in header
    assert "sp->num_segments > UINT32_MAX" in header
    assert "spectral_is_finite_positive_f32(sp->inv_stretch_sq)" in header

def test_metal_and_cuda_use_checked_gpu_pack() -> None:
    metal = read("spectral_engine/synth/backends/gpu/metal/spectral_synth_metal.m")
    cuda = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")

    assert "gpu_synth_params_pack_checked(&pf.params, SPECTRAL_GPU_TILE_SIZE, timbre, &params)" in metal
    assert "gpu_synth_params_pack_checked(&pf.params, SPECTRAL_GPU_TILE_SIZE, timbre, &gp)" in cuda
    assert "params = gpu_synth_params_pack(&pf.params" not in metal
    assert "gp = gpu_synth_params_pack(&pf.params" not in cuda

def test_pass42_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS42.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "GPU synth parameter packing contract" in notes
    assert "uint32_t" in notes
    assert "size_t" in notes
    assert "pass 42 GPU synth params" in audit
    assert "for pass_num in range(1, 43):" in audit
