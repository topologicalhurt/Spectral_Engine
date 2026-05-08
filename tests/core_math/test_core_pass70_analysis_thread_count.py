#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_analysis_exposes_bounded_effective_thread_count_helper() -> None:
    analysis = read("spectral_engine/analysis/spectral_analysis.c")
    header = read("spectral_engine/analysis/spectral_analysis_internal.h")

    assert "int spectral_analysis_effective_thread_count(void)" in analysis
    assert "SPECTRAL_MAX_THREADS" in analysis
    assert "int spectral_analysis_effective_thread_count(void);" in header

def test_full_and_fused_analysis_use_bounded_thread_count() -> None:
    full = read("spectral_engine/analysis/spectral_analysis_full.c")
    fused = read("spectral_engine/analysis/spectral_analysis_fused.c")

    assert "int n_threads = spectral_analysis_effective_thread_count();" in full
    assert "int n_threads = spectral_analysis_effective_thread_count();" in fused
    assert "int n_threads = omp_get_max_threads();" not in full
    assert "int n_threads = omp_get_max_threads();" not in fused

def test_pass70_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS70.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "analysis OpenMP thread-count domain contract" in notes
    assert "SPECTRAL_MAX_THREADS" in notes
    assert "pass 70 analysis thread count" in audit
    assert "for pass_num in range(1, 71):" in audit
