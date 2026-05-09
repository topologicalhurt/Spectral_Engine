#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_tracker_create_rejects_frequency_counts_outside_estimator_int_domain() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "n_freqs > (size_t)INT_MAX" in src
    assert "n_freqs != ((size_t)n_fft / 2u + 1u)" in src

def test_tracker_create_rejects_thread_counts_outside_engine_domain() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "if (n_threads > SPECTRAL_MAX_THREADS) return NULL;" in src
    assert "if (n_threads < 1) n_threads = 1;" in src

def test_pass83_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS83.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "tracker creation thread/frequency index-domain contract" in notes
    assert "SPECTRAL_MAX_THREADS" in notes
    assert "pass 83 tracker creation" in audit
    assert "for pass_num in range(1, 84):" in audit
