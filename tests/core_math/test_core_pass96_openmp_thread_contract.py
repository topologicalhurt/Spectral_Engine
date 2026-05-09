#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_openmp_effective_thread_count_lives_in_omp_shim() -> None:
    header = read("spectral_engine/core/spectral_omp.h")

    assert "spectral_omp_effective_thread_count" in header
    assert "SPECTRAL_MAX_THREADS" in header
    assert "omp_get_max_threads()" in header

def test_analysis_and_gpu_tile_preprocess_use_shared_thread_helper() -> None:
    analysis_c = read("spectral_engine/analysis/spectral_analysis.c")
    analysis_h = read("spectral_engine/analysis/spectral_analysis_internal.h")
    full = read("spectral_engine/analysis/spectral_analysis_full.c")
    fused = read("spectral_engine/analysis/spectral_analysis_fused.c")
    synth = read("spectral_engine/core/spectral_synth_internal.c")

    assert "spectral_analysis_effective_thread_count" not in analysis_c
    assert "spectral_analysis_effective_thread_count" not in analysis_h
    assert "int n_threads = spectral_omp_effective_thread_count();" in full
    assert "int n_threads = spectral_omp_effective_thread_count();" in fused
    assert "int n_threads = spectral_omp_effective_thread_count();" in synth

def test_pass96_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS96.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "shared OpenMP effective-thread contract" in notes
    assert "allocation shape" in notes
    assert "pass 96 OpenMP effective thread" in audit
    assert "for pass_num in range(1, 97):" in audit
