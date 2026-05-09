#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_fft_frame_dispatch_has_single_backend_agnostic_definition() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_fft.c")

    assert src.count("float spectral_fft_frames(") == 1
    assert "#pragma omp parallel reduction(max:max_magsq) num_threads(res->n_threads)" in src
    assert "spectral_fft_single_frame(res, tid" in src

def test_backend_specific_single_frame_definitions_remain() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_fft.c")

    assert src.count("void spectral_fft_single_frame(") == 2
    assert "#if SPECTRAL_USE_VDSP" in src
    assert "#else" in src

def test_pass97_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS97.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "FFT frame dispatch consolidation" in notes
    assert "one owner" in notes
    assert "pass 97 FFT frame dispatch" in audit
    assert "for pass_num in range(1, 98):" in audit
