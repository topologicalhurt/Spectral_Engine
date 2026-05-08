#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_candidate_flush_validates_count_and_pointer_contract() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "!candidate_batch_count" in src
    assert "*candidate_batch_count > SPECTRAL_TRACK_CANDIDATE_BATCH" in src
    assert "*candidate_batch_count = 0;" in src
    assert "!candidate_batch || !row || !next_row || !phase_row" in src

def test_candidate_queue_checks_capacity_before_write() -> None:
    src = read("spectral_engine/analysis/spectral_peak_track.c")

    assert "*candidate_batch_count >= SPECTRAL_TRACK_CANDIDATE_BATCH" in src
    assert "candidate_batch[(*candidate_batch_count)++] = candidate;" in src

    check_idx = src.index("*candidate_batch_count >= SPECTRAL_TRACK_CANDIDATE_BATCH")
    write_idx = src.index("candidate_batch[(*candidate_batch_count)++] = candidate;")
    assert check_idx < write_idx

def test_pass73_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS73.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "tracker candidate-batch bounds contract" in notes
    assert "fixed stack batch" in notes
    assert "pass 73 tracker candidate batch" in audit
    assert "for pass_num in range(1, 74):" in audit
