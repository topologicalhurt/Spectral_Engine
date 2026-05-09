#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_architecture_status_marks_phase_e_foundation_complete() -> None:
    status = read("docs/core_audit/ARCHITECTURE_CLEANUP_STATUS.md")

    assert "Phase E: tracker candidate context object foundation" in status
    assert "candidate batch owner" in status
    assert "worker-stats context" in status
    assert "frame-context constructor" in status
    assert "Phase F: full/fused parity harness" in status

def test_pass116_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS116.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "Phase E candidate-context status" in notes
    assert "full/fused behavioral parity tests" in notes
    assert "pass 116 phase E status" in audit
    assert "for pass_num in range(1, 117):" in audit
