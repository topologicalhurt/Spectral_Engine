#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_candidate_batch_owner_exists() -> None:
    header = read("spectral_engine/analysis/spectral_peak_track_internal.h")

    assert "typedef struct SpectralTrackerCandidateBatch" in header
    assert "uint32_t ids[SPECTRAL_TRACK_CANDIDATE_BATCH];" in header
    assert "size_t count;" in header
    assert "spectral_tracker_candidate_batch_reset" in header

def test_fused_worker_uses_candidate_batch_owner() -> None:
    fused = read("spectral_engine/analysis/spectral_analysis_fused.c")

    assert "SpectralTrackerCandidateBatch candidate_batch = {0};" in fused
    assert "candidate_batch.ids" in fused
    assert "&candidate_batch.count" in fused
    assert "uint32_t candidate_batch[SPECTRAL_TRACK_CANDIDATE_BATCH];" not in fused
    assert "size_t candidate_batch_count = 0;" not in fused

def test_pass112_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS112.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "tracker candidate batch owner" in notes
    assert "Phase E" in notes
    assert "pass 112 tracker candidate batch" in audit
    assert "for pass_num in range(1, 113):" in audit
