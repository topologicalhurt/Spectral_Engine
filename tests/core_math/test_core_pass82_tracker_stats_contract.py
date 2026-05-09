#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_tracker_stats_accumulation_checks_uint64_overflow() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "spectral_tracker_u64_add_checked" in src
    assert "b > UINT64_MAX - a" in src
    assert "#pragma omp critical(spectral_tracker_stats_accum)" in src
    assert "spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW)" in src

def test_tracker_stats_accumulation_rejects_nonfinite_timing() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "!isfinite(local_track_time) || local_track_time < 0.0" in src
    assert "next_time = tracker->process_time_total + local_track_time;" in src
    assert "!isfinite(next_time)" in src

def test_pass82_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS82.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "tracker stats accumulation overflow contract" in notes
    assert "uint64_t" in notes
    assert "pass 82 tracker stats" in audit
    assert "for pass_num in range(1, 83):" in audit
