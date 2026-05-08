#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_hex_line_parser_validates_exact_declared_record_length() -> None:
    src = read("spectral_engine/core/spectral_wavetable.c")

    assert "line_len = strcspn(line, \"\\r\\n\");" in src
    assert "required_len = 11u + ((size_t)byte_count * 2u);" in src
    assert "if (line_len != required_len) return WAVETABLE_ERR_FORMAT;" in src
    assert "if (!line || !data || !data_len || !address || !record_type) return WAVETABLE_ERR_PARAM;" in src

def test_hex_loader_requires_canonical_eof_record_shape() -> None:
    src = read("spectral_engine/core/spectral_wavetable.c")

    assert "if (data_len != 0u || address != 0u)" in src
    assert "return WAVETABLE_ERR_FORMAT;" in src
    assert "saw_eof = 1;" in src

def test_pass54_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS54.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "wavetable HEX record shape contract" in notes
    assert "EOF" in notes
    assert "pass 54 wavetable HEX" in audit
    assert "for pass_num in range(1, 55):" in audit
