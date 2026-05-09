#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_gpu_segment_cache_is_one_shot_not_count_only_identity() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "int gpu_seg_cache_try_get(uint32_t count, const SegmentGpu** out)" in src
    assert "gpu_seg_cache_clear();" in src
    assert "Count is not identity" in src
    assert "segs = g_gpu_seg_cache.segs;" in src
    assert "*out = segs;" in src

def test_gpu_segment_cache_clears_on_mismatch() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "g_gpu_seg_cache.count != count || !g_gpu_seg_cache.segs" in src

def test_pass87_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS87.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "GPU segment cache one-shot identity contract" in notes
    assert "Count is not identity" in notes
    assert "pass 87 GPU segment cache" in audit
    assert "for pass_num in range(1, 88):" in audit
