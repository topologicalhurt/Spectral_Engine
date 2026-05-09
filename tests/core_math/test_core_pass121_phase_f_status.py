#!/usr/bin/env python3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(rel): return (ROOT/rel).read_text(encoding="utf-8", errors="replace")
def test_architecture_status_marks_phase_f_seam_complete():
    status=read("docs/core_audit/ARCHITECTURE_CLEANUP_STATUS.md")
    assert "Phase F: full/fused parity testing seam and harness specification" in status
    assert "Phase G: implement compiled full/fused behavioral parity harness" in status
    assert "do not continue the campaign without a real bug" in status
    assert "ARM/embedded redesign" in status
def test_pass121_docs_and_audit_track_contract():
    notes=read("docs/core_audit/PATCH_NOTES_PASS121.md")
    audit=read("tools/core_audit/core_static_audit.py")
    assert "Phase F seam completion status" in notes
    assert "pass 121 phase F status" in audit
    assert "for pass_num in range(1, 122):" in audit
