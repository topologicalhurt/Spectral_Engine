#!/usr/bin/env python3
"""Pass-2 static/math contract tests for Spectral Engine core."""
from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_fast_sine_default_is_exact() -> None:
    src = text("spectral_engine/core/spectral_osc_formulas.h")
    assert "#define SPECTRAL_OSC_FORMULAS_VERSION 3" in src
    assert "#if SPECTRAL_ENABLE_APPROX_TRIG" in src
    assert "return sinf(x);" in src


def test_old_pade_endpoint_error_would_have_failed_contract() -> None:
    # This documents why the old coefficient set must not be the default.
    c1 = 0.16605
    c2 = 0.00761
    c3 = 0.00766
    x = math.pi
    x2 = x * x
    old = x * (1.0 - x2 * (c1 - x2 * c2)) / (1.0 + x2 * c3)
    assert abs(old - math.sin(x)) > 0.25


def test_fused_analysis_has_distinct_phase_rows() -> None:
    src = text("spectral_engine/analysis/spectral_analysis_fused.c")
    assert "float* phase_prev" in src
    assert "float* phase_curr" in src
    assert "float* phase_next" in src
    assert "target_row, target_phase" in src
    assert "phase_curr = phase_next" in src


def test_fused_analysis_max_pass_not_gated_by_fast_math() -> None:
    src = text("spectral_engine/analysis/spectral_analysis_fused.c")
    max_pass_region = src[src.find("Global Maximum Discovery") - 120:src.find("tracker = spectral_tracker_create")]
    assert "SPECTRAL_CUSTOM_FAST_MATH_MODE" not in max_pass_region


def test_cpu_native_and_wavetable_paths_apply_fade() -> None:
    src = text("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")
    for fn in ["segment_fn_wavetable_float", "segment_fn_native_timbre", "segment_fn_native_wavetable"]:
        start = src.find(f"static void {fn}")
        assert start >= 0, fn
        body = src[start:src.find("}\n", start) + 2]
        assert "fade_params_init" in body, fn
        assert "fade_envelope" in body, fn


def test_wavetable_builtins_use_canonical_oscillators() -> None:
    src = text("spectral_engine/core/spectral_wavetable.c")
    assert '#include "spectral_osc_formulas.h"' in src
    assert "spectral_osc_saw" in src
    assert "spectral_osc_square" in src
    assert "spectral_osc_triangle" in src
    assert "2.0 * t - 1.0" not in src


if __name__ == "__main__":
    tests = [name for name in globals() if name.startswith("test_")]
    for name in tests:
        globals()[name]()
    print(f"passed {len(tests)} pass-2 core math contract tests")
