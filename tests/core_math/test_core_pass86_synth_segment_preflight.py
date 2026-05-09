#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_synth_preflight_validates_segment_payload_before_backend_params() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "synth_segment_payload_valid" in src
    assert "!spectral_is_finite_f32(s->length) || s->length <= 0.0f" in src
    assert "!spectral_is_finite_f32(s->omega) || s->omega < 0.0f" in src
    assert "!spectral_is_finite_f32(s->amp) || s->amp < 0.0f" in src
    assert "if (!synth_segment_payload_valid(sa.segs, sa.count))" in src
    assert "pf.error = SPECTRAL_ERR_PARAM;" in src

    payload_idx = src.index("if (!synth_segment_payload_valid(sa.segs, sa.count))")
    params_idx = src.index("pf.params = make_synth_params")
    assert payload_idx < params_idx

def test_pass86_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS86.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "synthesis segment payload preflight contract" in notes
    assert "caller-provided" in notes
    assert "pass 86 synth segment preflight" in audit
    assert "for pass_num in range(1, 87):" in audit
