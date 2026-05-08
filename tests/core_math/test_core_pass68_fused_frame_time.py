#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_fused_analysis_checks_t_hop_float_representability() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_fused.c")

    assert '#include "spectral_peak_track_internal.h"' in src
    assert "#include <float.h>" in src
    assert "double t_hop_d = (double)pair * (double)hop;" in src
    assert "t_hop_d > (double)FLT_MAX" in src
    assert "spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);" in src
    assert "t_hop = (float)t_hop_d;" in src
    assert "float t_hop = (float)pair * (float)hop;" not in src

def test_pass68_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS68.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "fused-analysis frame-time float contract" in notes
    assert "FLT_MAX" in notes
    assert "pass 68 fused frame time" in audit
    assert "for pass_num in range(1, 69):" in audit
