#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_hash_oneshot_defines_null_span_contract() -> None:
    src = read("spectral_engine/core/spectral_hash_xx32_xx3.c")

    assert "static const unsigned char empty = 0u;" in src
    assert "if (!src)" in src
    assert "if (len > 0u)" in src
    assert "return (SpectralHashDigest)0;" in src
    assert "src = &empty;" in src
    assert "XXH3_64bits(src, len)" in src
    assert "XXH3_64bits(data, len)" not in src

def test_pass64_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS64.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "hash oneshot null-span contract" in notes
    assert "NULL" in notes
    assert "pass 64 hash oneshot" in audit
    assert "for pass_num in range(1, 65):" in audit
