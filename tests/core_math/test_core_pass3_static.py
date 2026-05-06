#!/usr/bin/env python3
"""Interim pass-3 static contracts for merged minimal branch."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


def test_fused_analysis_uses_adjacent_frame_pair_contract() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_fused.c")
    assert "const size_t pair_count = n_frames - 1u" in src
    assert "pair_start" in src and "pair_end" in src
    assert "spectral_fft_single_frame(&res, tid, audio, hop, window_func, pair_start" in src
    assert "spectral_fft_single_frame(&res, tid, audio, hop, window_func, pair + 1u" in src
    assert "fctx.row = row_curr" in src
    assert "fctx.next_row = row_next" in src
    assert "fctx.phase_row = phase_curr" in src
    assert "float t_hop = (float)pair * (float)hop" in src
    assert "t - 1" not in src
    assert "row_prev" not in src
    assert "phase_prev" not in src


def test_fused_analysis_max_discovery_is_unconditional() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_fused.c")
    start = src.index("Global maximum discovery")
    end = src.index("tracker = spectral_tracker_create")
    assert "SPECTRAL_CUSTOM_FAST_MATH_MODE" not in src[start:end]
    assert "pass1_failed" in src


def test_metal_normalize_string_has_no_dead_norm_variable() -> None:
    src = read("spectral_engine/core/oscillator.c")
    assert "float norm = p * INV_TWO_PI" not in src
    assert "return p - TWO_PI * floor(p * INV_TWO_PI + 0.5f);" in src


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))
