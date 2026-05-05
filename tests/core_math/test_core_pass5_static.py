#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


def test_gpu_tile_preprocess_has_single_span_helper() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")
    assert "static int spectral_gpu_segment_tile_span" in src
    assert src.count("spectral_gpu_segment_tile_span(&sa.segs[i], stretch, tile_size") == 2


def test_gpu_tile_span_uses_sample_domain_bounds() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")
    assert "floor(start / tile_size_d)" in src
    assert "ceil(end / tile_size_d)" in src
    assert "int start_tile = (int)(start / tile_size)" not in src
    assert "int end_tile = (int)(end / tile_size)" not in src


def test_zero_active_refs_are_valid() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")
    assert "if (total_refs > 0u)" in src
    assert "out->total_refs = total_refs;" in src


def test_tile_cache_reuse_is_tile_size_gated() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")
    assert "tile_size == (uint32_t)SPECTRAL_GPU_TILE_SIZE" in src


if __name__ == "__main__":
    test_gpu_tile_preprocess_has_single_span_helper()
    test_gpu_tile_span_uses_sample_domain_bounds()
    test_zero_active_refs_are_valid()
    test_tile_cache_reuse_is_tile_size_gated()
    print("pass5 static tests passed")
