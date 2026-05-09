#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_worker_stats_context_exists() -> None:
    header = read("spectral_engine/analysis/spectral_peak_track_internal.h")

    assert "typedef struct SpectralTrackerWorkerStats" in header
    assert "spectral_tracker_worker_stats_commit" in header
    assert "stats->pairs" in header
    assert "stats->track_time" in header

def test_fused_worker_commits_stats_through_context() -> None:
    fused = read("spectral_engine/analysis/spectral_analysis_fused.c")

    assert "SpectralTrackerWorkerStats worker_stats = {0};" in fused
    assert "spectral_tracker_worker_stats_commit(tracker, &worker_stats);" in fused

def test_pass114_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS114.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "tracker worker-stats context" in notes
    assert "repeated argument group" in notes
    assert "pass 114 tracker worker stats" in audit
    assert "for pass_num in range(1, 115):" in audit
