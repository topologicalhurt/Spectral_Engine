#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_fft_frames_validates_range_and_output_shape_before_dispatch() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_fft.c")

    assert "if (frame_end < frame_start) return 0.0f;" in src
    assert "if (!magsq_only && !out_phases) return 0.0f;" in src
    assert "spectral_size_mul(local_n_frames, n_freqs, &output_bins)" in src
    assert "local_n_frames = frame_end - frame_start;" in src

def test_fft_frames_binds_openmp_team_to_allocated_resource_threads() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_fft.c")

    assert "#pragma omp parallel reduction(max:max_magsq) num_threads(res->n_threads)" in src
    assert "int tid = omp_get_thread_num();" in src

def test_pass61_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS61.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "FFT frame dispatch range/thread contract" in notes
    assert "res->n_threads" in notes
    assert "pass 61 FFT frame dispatch" in audit
    assert "for pass_num in range(1, 62):" in audit
