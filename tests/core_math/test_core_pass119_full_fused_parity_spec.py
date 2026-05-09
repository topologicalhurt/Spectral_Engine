#!/usr/bin/env python3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(rel): return (ROOT/rel).read_text(encoding="utf-8", errors="replace")
def test_full_fused_parity_spec_exists():
    spec=read("docs/core_audit/FULL_FUSED_PARITY_HARNESS.md")
    assert "analyze_audio_with_path_mode" in spec
    assert "SPECTRAL_ANALYSIS_PATH_FULL" in spec
    assert "SPECTRAL_ANALYSIS_PATH_FUSED" in spec
    assert "Deterministic fixtures" in spec
    assert "Comparison rules" in spec
def test_pass119_docs_and_audit_track_contract():
    notes=read("docs/core_audit/PATCH_NOTES_PASS119.md")
    audit=read("tools/core_audit/core_static_audit.py")
    assert "full/fused parity harness specification" in notes
    assert "pass 119 full/fused parity spec" in audit
    assert "for pass_num in range(1, 120):" in audit
