#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_architecture_status_marks_phase_d_complete() -> None:
    status = read("docs/core_audit/ARCHITECTURE_CLEANUP_STATUS.md")

    assert "Phase D: explicit prepared GPU dispatch object" in status
    assert "explicit prepared GPU dispatch plan" in status
    assert "one-shot GPU cache lookup encapsulation" in status
    assert "backend-owned host-side GPU dispatch preparation" in status
    assert "Phase E: tracker candidate context object" in status

def test_pass111_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS111.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "Phase D completion status" in notes
    assert "SpectralGpuDispatchPlan" in notes
    assert "pass 111 phase D completion" in audit
    assert "for pass_num in range(1, 112):" in audit
