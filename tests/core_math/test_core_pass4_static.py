#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_synth_preflight_propagates_errors() -> None:
    h = read("spectral_engine/core/spectral_synth_internal.h")
    c = read("spectral_engine/core/spectral_synth_internal.c")
    assert "SpectralError error" in h
    assert "synth_preflight_native" in h
    assert "SpectralError synth_validate_params" in c
    assert "SPECTRAL_MAX_STRETCH" in c
    assert "SPECTRAL_MIN_PITCH" in c and "SPECTRAL_MAX_PITCH" in c


def test_public_backends_return_preflight_error() -> None:
    cpu = read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")
    cuda = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")
    metal = read("spectral_engine/synth/backends/gpu/metal/spectral_synth_metal.m")
    assert cpu.count("return pf.error;") >= 4
    assert "if (!pf.ok) return pf.error;" in cuda
    assert "if (!pf.ok) return pf.error;" in metal


def test_segment_and_tile_finite_guards() -> None:
    c = read("spectral_engine/core/spectral_synth_internal.c")
    assert "!spectral_is_finite_f32(s->omega)" in c
    assert "!spectral_is_finite_f32(s->df)" in c
    assert "!spectral_is_finite_f32(s->amp)" in c
    gpu_region = c[c.find("gpu_tile_preprocess") :]
    assert "spectral_is_finite_positive_f32(stretch)" in gpu_region


def test_analysis_input_guards() -> None:
    analysis = read("spectral_engine/analysis/spectral_analysis.c")
    fft = read("spectral_engine/analysis/spectral_analysis_fft.c")
    assert "SPECTRAL_MIN_SAMPLE_RATE" in analysis
    assert "SPECTRAL_MAX_SAMPLE_RATE" in analysis
    assert "SPECTRAL_MIN_FFT_SIZE" in analysis
    assert "n_freqs != (n_fft / 2u + 1u)" in fft
