#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_analysis_window_context_api_exists() -> None:
    header = read("spectral_engine/analysis/spectral_analysis_internal.h")
    impl = read("spectral_engine/analysis/spectral_analysis.c")

    assert "SpectralAnalysisWindowContext" in header
    assert "spectral_analysis_window_context_init" in header
    assert "spectral_analysis_window_context_apply_magsq_scales" in header
    assert "ctx->metrics = spectral_window_metrics" in impl
    assert "spectral_fft_resources_set_magsq_scales" in impl

def test_full_and_fused_use_shared_window_context() -> None:
    full = read("spectral_engine/analysis/spectral_analysis_full.c")
    fused = read("spectral_engine/analysis/spectral_analysis_fused.c")

    assert "SpectralAnalysisWindowContext window_ctx = {0};" in full
    assert "SpectralAnalysisWindowContext window_ctx = {0};" in fused
    assert "spectral_analysis_window_context_init(&window_ctx" in full
    assert "spectral_analysis_window_context_init(&window_ctx" in fused
    assert "spectral_analysis_window_context_apply_magsq_scales(&window_ctx, &res);" in full
    assert "spectral_analysis_window_context_apply_magsq_scales(&window_ctx, &res);" in fused
    assert "spectral_window_generate(window_func" not in full
    assert "spectral_window_generate(window_func" not in fused

def test_pass93_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS93.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "analysis window context consolidation" in notes
    assert "full/fused parity" in notes
    assert "pass 93 analysis window context" in audit
    assert "for pass_num in range(1, 94):" in audit
