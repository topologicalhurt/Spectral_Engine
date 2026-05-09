#!/usr/bin/env python3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(rel): return (ROOT/rel).read_text(encoding="utf-8", errors="replace")
def test_analysis_exposes_forced_path_entrypoint():
    h=read("spectral_engine/analysis/spectral_analysis_internal.h")
    c=read("spectral_engine/analysis/spectral_analysis.c")
    assert "analyze_audio_with_path_mode" in h
    assert "SegmentArray analyze_audio_with_path_mode" in c
    assert "SPECTRAL_ANALYSIS_PATH_FULL" in h
    assert "SPECTRAL_ANALYSIS_PATH_FUSED" in h
def test_default_analyze_audio_delegates_to_auto_path_mode():
    c=read("spectral_engine/analysis/spectral_analysis.c")
    assert "return analyze_audio_with_path_mode" in c
    assert "SPECTRAL_ANALYSIS_PATH_AUTO" in c
