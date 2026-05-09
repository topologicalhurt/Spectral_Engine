#!/usr/bin/env python3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(rel): return (ROOT/rel).read_text(encoding="utf-8", errors="replace")
def test_master_plan_closure_criteria_exists():
    doc=read("docs/core_audit/MASTER_PLAN_CLOSURE_CRITERIA.md")
    assert "Close the campaign" in doc
    assert "Do not continue with passes" in doc
    assert "Require justification for every future pass" in doc
    assert "compiled full/fused parity harness" in doc
def test_pass120_docs_and_audit_track_contract():
    notes=read("docs/core_audit/PATCH_NOTES_PASS120.md")
    audit=read("tools/core_audit/core_static_audit.py")
    assert "master-plan closure criteria" in notes
    assert "pass 120 master plan closure" in audit
    assert "for pass_num in range(1, 121):" in audit
