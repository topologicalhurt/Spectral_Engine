#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FFT = (ROOT / "spectral_engine/analysis/spectral_analysis_fft.c").read_text(encoding="utf-8")

def test_fft_resource_pointer_arrays_are_zero_initialized() -> None:
    assert "res->fft_setups = spectral_calloc_array" in FFT
    assert "res->thread_real = spectral_calloc_array" in FFT
    assert "res->thread_imag = spectral_calloc_array" in FFT
    assert "res->thread_windowed = spectral_calloc_array" in FFT
    assert "res->thread_imag_sq = spectral_calloc_array" in FFT
    assert "res->fft_plans = spectral_calloc_array" in FFT
    assert "res->thread_in = spectral_calloc_array" in FFT
    assert "res->thread_out = spectral_calloc_array" in FFT

def test_fft_resource_alloc_uses_single_cleanup_path() -> None:
    assert "fail:\n    spectral_fft_resources_free(res);\n    return 0;" in FFT
    assert "goto fail;" in FFT

def test_fft_resource_free_is_null_safe_and_zeroes_state() -> None:
    free_start = FFT.index("void spectral_fft_resources_free")
    free_body = FFT[free_start:]
    assert "if (!res) return;" in free_body
    assert "memset(res, 0, sizeof(*res));" in free_body
