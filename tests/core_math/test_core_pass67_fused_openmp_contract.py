#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_fused_analysis_parallel_regions_bind_to_actual_threads() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_fused.c")

    assert "#pragma omp parallel reduction(max:pass1_max) num_threads(actual_threads)" in src
    assert "#pragma omp parallel num_threads(actual_threads)" in src
    assert "omp_set_num_threads(actual_threads)" not in src

def test_pass67_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS67.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "fused-analysis OpenMP resource-thread contract" in notes
    assert "num_threads(actual_threads)" in notes
    assert "pass 67 fused analysis OpenMP" in audit
    assert "for pass_num in range(1, 68):" in audit
