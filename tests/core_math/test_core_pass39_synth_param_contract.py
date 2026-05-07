#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_synth_param_validation_checks_derived_scalars_not_only_inputs() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "synth_derive_param_scalars" in src
    assert "stretch_sq = stretch * stretch;" in src
    assert "stretch_sq <= 0.0f" in src
    assert "inv_stretch = 1.0f / stretch;" in src
    assert "inv_stretch_sq = 1.0f / stretch_sq;" in src
    assert "pitch_factor = SPECTRAL_PITCH_FACTOR(pitch);" in src
    assert "spectral_is_finite_positive_f32(inv_stretch)" in src
    assert "spectral_is_finite_positive_f32(inv_stretch_sq)" in src
    assert "spectral_is_finite_positive_f32(pitch_factor)" in src

def test_make_synth_params_reuses_checked_derived_scalars() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "synth_derive_param_scalars(stretch, pitch," in src
    assert ".inv_stretch = inv_stretch" in src
    assert ".inv_stretch_sq = inv_stretch_sq" in src
    assert ".pitch_factor = pitch_factor" in src
    assert ".inv_stretch = 1.0f / stretch" not in src
    assert ".inv_stretch_sq = 1.0f / (stretch * stretch)" not in src
    assert ".pitch_factor = SPECTRAL_PITCH_FACTOR(pitch)" not in src

def test_preflight_still_zeroes_output_on_invalid_derived_params() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "pf.error = synth_validate_params(stretch, pitch);" in src
    assert "synth_zero_output_if_valid(out_buffer, out_len, elem_size);" in src

def test_pass39_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS39.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "synthesis derived-parameter finiteness contract" in notes
    assert "tiny positive stretch" in notes
    assert "inv_stretch_sq" in notes
    assert "pass 39 synth params" in audit
    assert "for pass_num in range(1, 40):" in audit
