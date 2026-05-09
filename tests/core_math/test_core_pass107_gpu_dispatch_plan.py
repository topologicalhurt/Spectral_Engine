#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_gpu_dispatch_plan_api_exists() -> None:
    header = read("spectral_engine/core/spectral_synth_internal.h")
    impl = read("spectral_engine/core/spectral_synth_internal.c")

    assert "typedef struct SpectralGpuDispatchPlan" in header
    assert "spectral_gpu_dispatch_plan_init" in header
    assert "spectral_gpu_dispatch_plan_free" in header
    assert "gpu_seg_cache_try_get(sa.count, &cached_gpu_segs)" in impl
    assert "gpu_tile_preprocess_cached(sa, stretch" in impl
    assert "gpu_synth_params_pack_checked(params" in impl

def test_gpu_dispatch_plan_owns_zero_output_and_tile_bytes() -> None:
    impl = read("spectral_engine/core/spectral_synth_internal.c")

    assert "plan->zero_output = 1;" in impl
    assert "plan->tile_ids_bytes" in impl
    assert "plan->tile_ranges_bytes" in impl
    assert "spectral_gpu_dispatch_plan_free(plan)" in impl

def test_pass107_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS107.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "prepared GPU dispatch plan" in notes
    assert "Phase D" in notes
    assert "pass 107 prepared GPU dispatch" in audit
    assert "for pass_num in range(1, 108):" in audit
