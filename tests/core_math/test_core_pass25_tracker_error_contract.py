#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_tracker_error_setter_preserves_first_error() -> None:
    track_c = read("spectral_engine/analysis/spectral_peak_track.c")
    internal_h = read("spectral_engine/analysis/spectral_peak_track_internal.h")

    assert "void spectral_tracker_set_error(SpectralTracker* tracker, SpectralError error);" in internal_h
    assert "void spectral_tracker_set_error(SpectralTracker* tracker, SpectralError error)" in track_c
    assert "atomic_compare_exchange_strong_explicit" in track_c
    assert "expected = SPECTRAL_OK" in track_c
    assert "spectral_tracker_set_error(tracker, SPECTRAL_ERR_MEMORY);" in track_c

def test_emit_and_batch_paths_do_not_overwrite_error_with_raw_store() -> None:
    interp_c = read("spectral_engine/analysis/spectral_peak_interp.c")
    track_c = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);" in interp_c
    assert "spectral_tracker_set_error(tracker, SPECTRAL_ERR_MEMORY);" in interp_c
    assert "spectral_tracker_set_error(tracker, SPECTRAL_ERR_MEMORY);" in track_c

    assert "atomic_store_explicit(&tracker->last_error, SPECTRAL_ERR_OVERFLOW" not in interp_c
    assert "atomic_store_explicit(&tracker->last_error, SPECTRAL_ERR_MEMORY" not in interp_c
    assert "atomic_store_explicit(&tracker->last_error, SPECTRAL_ERR_MEMORY" not in track_c

def test_pass25_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS25.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "first non-OK tracker error" in notes
    assert "SPECTRAL_ERR_OVERFLOW" in notes
    assert "SPECTRAL_ERR_MEMORY" in notes
    assert "pass 25 tracker error propagation" in audit
    assert "for pass_num in range(1, 26):" in audit
