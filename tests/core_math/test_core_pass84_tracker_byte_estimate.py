#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_tracker_estimate_bytes_checks_stft_accounting_addition() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "if (!spectral_size_mul(pairs, tracker->n_freqs, &stft_writes) ||" in src
    assert "!spectral_size_add(total_read_floats, fft_mags_phases, &total_read_floats))" in src
    assert "spectral_size_add(total_read_floats, fft_mags_phases, &total_read_floats);" not in src

def test_pass84_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS84.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "tracker byte-estimate STFT accounting overflow contract" in notes
    assert "ignored" in notes
    assert "pass 84 tracker byte estimate" in audit
    assert "for pass_num in range(1, 85):" in audit
