#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_gpu_cache_try_get_is_not_exported_from_synth_internal_header() -> None:
    header = read("spectral_engine/core/spectral_synth_internal.h")

    assert "gpu_tile_cache_set" in header
    assert "gpu_seg_cache_set" in header
    assert "gpu_tile_cache_try_get" not in header
    assert "gpu_seg_cache_try_get" not in header

def test_try_get_is_used_only_inside_synth_internal_dispatch_plan() -> None:
    metal = read("spectral_engine/synth/backends/gpu/metal/spectral_synth_metal.m")
    cuda = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")
    synth = read("spectral_engine/core/spectral_synth_internal.c")

    assert "gpu_tile_cache_try_get" not in metal
    assert "gpu_seg_cache_try_get" not in metal
    assert "gpu_tile_cache_try_get" not in cuda
    assert "gpu_seg_cache_try_get" not in cuda
    assert "gpu_seg_cache_try_get(sa.count" in synth

def test_pass110_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS110.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "GPU cache lookup encapsulation" in notes
    assert "prepared dispatch plan" in notes
    assert "pass 110 GPU cache encapsulation" in audit
    assert "for pass_num in range(1, 111):" in audit
