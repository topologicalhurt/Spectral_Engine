#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_quantized_oscillator_checks_float_to_int_domain() -> None:
    header = read("spectral_engine/core/spectral_osc_formulas.h")

    assert "#define SPECTRAL_OSC_FORMULAS_VERSION 4" in header
    assert "#include <limits.h>" in header
    assert "scaled < (float)INT_MIN" in header
    assert "scaled > (float)INT_MAX" in header
    assert "return (float)(int)scaled * inv_w;" in header
    assert "(int)(rads * width)" not in header

def test_pwm_oscillator_rejects_nonfinite_width_and_phase() -> None:
    header = read("spectral_engine/core/spectral_osc_formulas.h")

    assert "OSC_FORMULA_FUNC float spectral_osc_pwm" in header
    assert "if (!isfinite(rads) || !isfinite(width)) return 0.0f;" in header

def test_formula_version_guards_are_bumped() -> None:
    osc = read("spectral_engine/core/oscillator.c")
    metal = read("spectral_engine/synth/backends/gpu/metal/spectral_synth_metal.m")

    assert "SPECTRAL_OSC_FORMULAS_VERSION == 4" in osc
    assert "SPECTRAL_OSC_FORMULAS_VERSION == 4" in metal

def test_pass89_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS89.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "oscillator width and quantized-cast contract" in notes
    assert "undefined behavior" in notes
    assert "pass 89 oscillator quantized" in audit
    assert "for pass_num in range(1, 90):" in audit
