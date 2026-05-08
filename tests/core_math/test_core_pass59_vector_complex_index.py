#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_vector_complex_interleaved_helpers_check_count_domain() -> None:
    src = read("spectral_engine/core/spectral_vector_ops.c")

    assert "spectral_complex_interleaved_count_valid" in src
    assert "count <= SIZE_MAX / 2u" in src
    assert "if (!interleaved || !re || !im || count == 0 || !spectral_complex_interleaved_count_valid(count)) return;" in src
    assert "if (!interleaved || !magsq || !phase || !max_magsq)" in src
    assert "if (!spectral_complex_interleaved_count_valid(count)) return;" in src

def test_magsq_helpers_clear_max_on_invalid_or_empty_input() -> None:
    src = read("spectral_engine/core/spectral_vector_ops.c")

    assert "*max_magsq = 0.0f;" in src
    assert src.count("*max_magsq = 0.0f;") >= 2

def test_pass59_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS59.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "vector complex-interleaved index contract" in notes
    assert "SIZE_MAX / 2" in notes
    assert "pass 59 vector complex" in audit
    assert "for pass_num in range(1, 60):" in audit
