#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_vector_primitives_guard_null_or_empty_spans_before_simd_access() -> None:
    src = read("spectral_engine/core/spectral_vector_ops.c")

    assert "if (!a || !b || !dst || len == 0) return;" in src
    assert "if (!src || !dst || len == 0) return;" in src
    assert "if (!y || !x || !dst || len == 0) return;" in src
    assert "if (!out) return;" in src
    assert "if (!src || len == 0) { *out = 0.0f; return; }" in src

def test_pass60_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS60.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "vector primitive null/span contract" in notes
    assert "SIMD" in notes
    assert "pass 60 vector primitives" in audit
    assert "for pass_num in range(1, 61):" in audit
