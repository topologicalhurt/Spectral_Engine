#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_checked_accumulation_helpers_live_in_contract_header() -> None:
    header = read("spectral_engine/core/spectral_contracts.h")

    assert "spectral_u64_add_checked" in header
    assert "spectral_double_accumulate_nonnegative_checked" in header
    assert "b > UINT64_MAX - a" in header

def test_tracker_stats_use_canonical_checked_accumulation() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "spectral_u64_add_checked(*field, delta, &next)" in src
    assert "spectral_double_accumulate_nonnegative_checked(*field, delta, &next)" in src
    assert "spectral_tracker_u64_add_checked" not in src

def test_pass105_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS105.md")
    audit = read("tools/core_audit/core_static_audit.py")
    assert "checked accumulation contract promotion" in notes
    assert "pass 105 checked accumulation" in audit
    assert "for pass_num in range(1, 106):" in audit
