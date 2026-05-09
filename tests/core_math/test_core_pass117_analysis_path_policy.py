#!/usr/bin/env python3
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def read(rel): return (ROOT / rel).read_text(encoding="utf-8", errors="replace")
def test_analysis_path_decision_type_exists():
    h = read("spectral_engine/analysis/spectral_analysis_internal.h")
    c = read("spectral_engine/analysis/spectral_analysis.c")
    assert "SpectralAnalysisPathDecision" in h
    assert "SPECTRAL_ANALYSIS_PATH_FULL" in h
    assert "SPECTRAL_ANALYSIS_PATH_FUSED" in h
    assert "spectral_analysis_path_decide" in c
    assert "shape->path = spectral_analysis_path_decide" in c
def test_pass117_docs_and_audit_track_contract():
    notes = read("docs/core_audit/PATCH_NOTES_PASS117.md")
    audit = read("tools/core_audit/core_static_audit.py")
    assert "analysis path policy object" in notes
    assert "pass 117 analysis path policy" in audit
    assert "for pass_num in range(1, 118):" in audit
