#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_candidate_flush_is_static_implementation_detail() -> None:
    header = read("spectral_engine/analysis/spectral_peak_track_internal.h")
    src = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "spectral_tracker_flush_candidate_batch(" not in header
    assert "static int spectral_tracker_flush_candidate_batch(" in src

def test_pass115_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS115.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "tracker flush API encapsulation" in notes
    assert "implementation detail" in notes
    assert "pass 115 tracker flush encapsulation" in audit
    assert "for pass_num in range(1, 116):" in audit
