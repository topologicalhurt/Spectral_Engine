#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_segment_loop_checks_float_representability_of_final_sample_offset() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "#include <float.h>" in src
    assert "double last_offset_d = 0.0;" in src
    assert "last_offset_d = (double)(lp.length - 1u);" in src
    assert "last_offset_d > (double)FLT_MAX" in src
    assert "const float last = (float)last_offset_d;" in src
    assert "const float last = (float)(lp.length - 1u);" not in src

def test_pass47_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS47.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "sample-offset float representability" in notes
    assert "FLT_MAX" in notes
    assert "pass 47 segment loop" in audit
    assert "for pass_num in range(1, 48):" in audit
