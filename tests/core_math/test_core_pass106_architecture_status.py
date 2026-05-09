#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_architecture_cleanup_status_exists() -> None:
    status = read("docs/core_audit/ARCHITECTURE_CLEANUP_STATUS.md")

    assert "Completed reusable owners" in status
    assert "Remaining high-value dedup targets" in status
    assert "Forbidden anti-patterns" in status
    assert "explicit prepared GPU dispatch object" in status

def test_pass106_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS106.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "architecture cleanup status" in notes
    assert "ARCHITECTURE_CLEANUP_STATUS.md" in notes
    assert "pass 106 architecture cleanup status" in audit
    assert "for pass_num in range(1, 107):" in audit
