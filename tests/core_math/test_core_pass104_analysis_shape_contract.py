#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_analysis_shape_contract_helper_exists() -> None:
    header = read("spectral_engine/analysis/spectral_analysis_internal.h")
    impl = read("spectral_engine/analysis/spectral_analysis.c")

    assert "SpectralAnalysisShape" in header
    assert "spectral_analysis_shape_init" in header
    assert "spectral_analysis_path_name" in header
    assert "shape->n_frames" in impl
    assert "shape->use_fused_path" in impl

def test_analyze_audio_uses_shape_contract() -> None:
    src = read("spectral_engine/analysis/spectral_analysis.c")

    assert "SpectralAnalysisShape shape = {0};" in src
    assert "spectral_analysis_shape_init(&shape" in src
    assert "shape.n_frames" in src
    assert "shape.n_freqs" in src
    assert "spectral_analysis_path_name(shape.use_fused_path)" in src

def test_pass104_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS104.md")
    audit = read("tools/core_audit/core_static_audit.py")
    assert "analysis shape contract consolidation" in notes
    assert "pass 104 analysis shape" in audit
    assert "for pass_num in range(1, 105):" in audit
