#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_mono_to_stereo_helpers_reject_unrepresentable_stereo_index_domain() -> None:
    src = read("spectral_engine/core/spectral_out.c")

    assert "num_frames > SIZE_MAX / 2u" in src
    assert src.count("num_frames > SIZE_MAX / 2u") >= 2
    assert "size_t stereo_i = i * 2u;" in src
    assert "stereo[i * 2]" not in src
    assert "stereo[i * 2 + 1]" not in src

def test_pass57_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS57.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "mono-to-stereo index-domain contract" in notes
    assert "SIZE_MAX / 2" in notes
    assert "pass 57 mono-to-stereo" in audit
    assert "for pass_num in range(1, 58):" in audit
