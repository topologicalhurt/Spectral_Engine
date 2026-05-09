#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_metal_consumes_prepared_gpu_dispatch_plan() -> None:
    src = read("spectral_engine/synth/backends/gpu/metal/spectral_synth_metal.m")

    assert "SpectralGpuDispatchPlan plan = {0};" in src
    assert "spectral_gpu_dispatch_plan_init(&plan" in src
    assert "spectral_gpu_dispatch_plan_free(&plan)" in src
    assert "plan.segment_source" in src
    assert "plan.tiles.num_tiles" in src
    assert "plan.params" in src

def test_metal_no_longer_reimplements_host_side_gpu_preparation() -> None:
    src = read("spectral_engine/synth/backends/gpu/metal/spectral_synth_metal.m")

    assert "gpu_seg_cache_try_get" not in src
    assert "gpu_tile_preprocess_cached" not in src
    assert "gpu_synth_params_pack_checked" not in src
    assert "plan.tile_ids_bytes" in src
    assert "plan.tile_ranges_bytes" in src

def test_pass108_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS108.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "Metal prepared GPU dispatch wiring" in notes
    assert "Metal-specific work" in notes
    assert "pass 108 Metal prepared dispatch" in audit
    assert "for pass_num in range(1, 109):" in audit
