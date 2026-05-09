#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def body(src: str, fn: str) -> str:
    start = src.index(fn)
    brace = src.index("{", start)
    depth = 0
    for i in range(brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i+1]
    raise AssertionError("function body not found")

def test_peak_offset_clamp_helper_is_not_recursive() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")
    fn = body(src, "static int spectral_peak_store_clamped_offset_d")

    assert fn.count("spectral_peak_store_clamped_offset_d(") == 1
    assert "if (offset_d > 0.5)" in fn
    assert "*out_offset = 0.5f;" in fn
    assert "*out_offset = -0.5f;" in fn
    assert "*out_offset = (float)offset_d;" in fn

def test_estimators_still_use_shared_clamp_helper() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")
    assert src.count("return spectral_peak_store_clamped_offset_d(out_offset, offset_d);") >= 3

def test_pass102_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS102.md")
    audit = read("tools/core_audit/core_static_audit.py")
    assert "regression repair" in notes
    assert "pass 102 peak clamp regression" in audit
    assert "for pass_num in range(1, 103):" in audit
