#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_segment_loop_rejects_nonfinite_derived_hot_loop_scalars() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "float alpha = 0.0f;" in src
    assert "float beta = 0.0f;" in src
    assert "float d_amp = 0.0f;" in src
    assert "alpha = spectral_segment_alpha_f32" in src
    assert "beta = spectral_segment_beta_f32" in src
    assert "d_amp = spectral_segment_d_amp_f32" in src
    assert "!spectral_is_finite_f32(alpha)" in src
    assert "!spectral_is_finite_f32(beta)" in src
    assert "!spectral_is_finite_f32(d_amp)" in src

def test_segment_loop_rejects_nonfinite_endpoint_phase_and_amplitude() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "const float last = (float)(lp.length - 1u);" in src
    assert "spectral_segment_phase_at_f32(s->phase, alpha, beta, last)" in src
    assert "spectral_segment_amp_at_f32(s->amp, d_amp, last)" in src
    assert "!spectral_is_finite_f32(phase1)" in src
    assert "!spectral_is_finite_f32(amp1)" in src

def test_pass41_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS41.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "segment loop derived-scalar contract" in notes
    assert "endpoint phase/amplitude" in notes
    assert "pass 41 segment loop" in audit
    assert "for pass_num in range(1, 42):" in audit
