#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_hash_consume_file_validates_lifecycle_and_descriptor_before_dispatch() -> None:
    src = read("spectral_engine/core/spectral_hash_xx32_xx3.c")

    assert "SpectralError spectral_hash_file_method_consume_file(" in src
    assert "if (!method || !file)" in src
    assert "if (!method->initialized)" in src
    assert "return SPECTRAL_ERR_NOTINIT;" in src
    assert "spectral_hash_file_method_get_descriptor(method->type, &desc)" in src
    assert "switch (desc->type)" in src
    assert "switch (method->type)" not in src

def test_pass63_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS63.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "hash consume-file lifecycle contract" in notes
    assert "initialized" in notes
    assert "pass 63 hash consume" in audit
    assert "for pass_num in range(1, 64):" in audit
