#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_audio_window_rejects_invalid_time_domain_inputs() -> None:
    src = read("spectral_engine/core/spectral_in.c")

    assert "*out_start = NULL;" in src
    assert "*out_frames = 0;" in src
    assert "sample_rate < SPECTRAL_MIN_SAMPLE_RATE" in src
    assert "sample_rate > SPECTRAL_MAX_SAMPLE_RATE" in src
    assert "!spectral_is_finite_f32(start_sec)" in src
    assert "end_sec >= 0.0f && !spectral_is_finite_f32(end_sec)" in src
    assert "(start_sec > 0.0f) ? (double)start_sec * (double)sample_rate : 0.0" in src

def test_pass53_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS53.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "audio window time-domain contract" in notes
    assert "NaN" in notes
    assert "pass 53 audio window" in audit
    assert "for pass_num in range(1, 54):" in audit
