#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_gpu_tile_cache_is_one_shot_handoff_not_shape_identity() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "int gpu_tile_cache_try_get(float stretch, size_t out_len, GpuTileData* out)" in src
    assert "Tile layout depends on segment start/length" in src
    assert "gpu_tile_cache_clear();" in src
    assert "*out = td;" in src

def test_gpu_tile_cache_clears_on_shape_mismatch() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "g_gpu_tile_cache.out_len != out_len" in src

def test_pass88_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS88.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "GPU tile cache one-shot layout identity contract" in notes
    assert "segment start/length" in notes
    assert "pass 88 GPU tile cache" in audit
    assert "for pass_num in range(1, 89):" in audit
