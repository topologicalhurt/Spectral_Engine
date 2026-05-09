#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_stft_matrix_owner_api_exists() -> None:
    header = read("spectral_engine/analysis/spectral_analysis_internal.h")
    impl = read("spectral_engine/analysis/spectral_analysis.c")

    assert "SpectralAnalysisStftMatrix" in header
    assert "spectral_analysis_stft_matrix_alloc" in header
    assert "spectral_analysis_stft_matrix_free" in header
    assert "matrix->magsq" in impl
    assert "matrix->phases" in impl

def test_full_analysis_uses_stft_matrix_owner() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_full.c")

    assert "SpectralAnalysisStftMatrix stft = {0};" in src
    assert "spectral_analysis_stft_matrix_alloc(&stft, n_frames, n_freqs)" in src
    assert "spectral_analysis_stft_matrix_free(&stft)" in src
    assert "stft.magsq" in src
    assert "stft.phases" in src
    assert "float* magsq" not in src
    assert "float* phases" not in src

def test_pass98_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS98.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "full-analysis STFT matrix ownership" in notes
    assert "reusable owner" in notes
    assert "pass 98 STFT matrix" in audit
    assert "for pass_num in range(1, 99):" in audit
