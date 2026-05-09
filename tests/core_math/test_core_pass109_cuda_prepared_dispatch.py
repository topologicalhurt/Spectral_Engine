#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_cuda_consumes_prepared_gpu_dispatch_plan() -> None:
    src = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")

    assert "SpectralGpuDispatchPlan plan = {0};" in src
    assert "spectral_gpu_dispatch_plan_init(&plan" in src
    assert "spectral_gpu_dispatch_plan_free(&plan)" in src
    assert "plan.segment_source" in src
    assert "plan.tiles.num_tiles" in src
    assert "plan.params.out_len" in src

def test_cuda_no_longer_reimplements_host_side_gpu_preparation() -> None:
    src = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")

    assert "gpu_seg_cache_try_get" not in src
    assert "gpu_tile_preprocess_cached" not in src
    assert "gpu_synth_params_pack_checked" not in src
    assert "plan.tile_ids_bytes" in src
    assert "plan.tile_ranges_bytes" in src

def test_pass109_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS109.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "CUDA prepared GPU dispatch wiring" in notes
    assert "CUDA-specific work" in notes
    assert "pass 109 CUDA prepared dispatch" in audit
    assert "for pass_num in range(1, 110):" in audit
