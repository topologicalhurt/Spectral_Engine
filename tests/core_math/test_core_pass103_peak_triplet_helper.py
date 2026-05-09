#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_peak_magsq_triplet_helper_exists() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")

    assert "typedef struct SpectralPeakMagsqTriplet" in src
    assert "spectral_peak_load_magsq_triplet" in src
    assert "triplet.left = row[center_bin - 1u];" in src
    assert "triplet.right = row[center_bin + 1u];" in src

def test_estimator_paths_reuse_triplet_helper() -> None:
    src = read("spectral_engine/analysis/spectral_peak_estimator.c")

    assert src.count("spectral_peak_load_magsq_triplet(") >= 6
    assert "spectral_peak_load_magsq_triplet(input->magsq_row" in src
    assert "spectral_peak_load_magsq_triplet(input->next_magsq_row" in src
    assert "peak_magsq(triplet.left, triplet.center, triplet.right" in src

def test_pass103_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS103.md")
    audit = read("tools/core_audit/core_static_audit.py")
    assert "peak magnitude triplet helper consolidation" in notes
    assert "pass 103 peak triplet helper" in audit
    assert "for pass_num in range(1, 104):" in audit
