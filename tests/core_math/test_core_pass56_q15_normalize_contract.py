#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_q15_normalization_rejects_lengths_outside_cmsis_domain() -> None:
    src = read("spectral_engine/core/spectral_out.c")

    assert "if (len > (size_t)UINT32_MAX)" in src
    assert "uint32_t len_u32 = (uint32_t)len;" in src
    assert "arm_absmax_q15(buffer, len_u32" in src
    assert "arm_shift_q15(buffer, -shift_amt, buffer, len_u32)" in src
    assert "arm_absmax_q15(buffer, (uint32_t)len" not in src
    assert "arm_shift_q15(buffer, -shift_amt, buffer, (uint32_t)len)" not in src

def test_q15_normalization_clears_shift_on_all_early_returns() -> None:
    src = read("spectral_engine/core/spectral_out.c")

    fn = src[src.index("q15_t spectral_normalize_q15"):src.index("/*\n * Stereo Interleaving")]
    assert "if (shift) *shift = 0;" in fn
    assert fn.index("if (shift) *shift = 0;") < fn.index("if (!buffer || len == 0)")

def test_pass56_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS56.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "Q15 normalization CMSIS length contract" in notes
    assert "UINT32_MAX" in notes
    assert "pass 56 Q15 normalization" in audit
    assert "for pass_num in range(1, 57):" in audit
