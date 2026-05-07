#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_gpu_tile_preprocess_rejects_segment_counts_outside_uint32_id_domain() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "if (sa.count > (size_t)UINT32_MAX) return SPECTRAL_ERR_OVERFLOW;" in src
    assert "tile_segment_ids[write_index] = (uint32_t)i;" in src

    check_idx = src.index("if (sa.count > (size_t)UINT32_MAX) return SPECTRAL_ERR_OVERFLOW;")
    parallel_idx = src.index("#pragma omp parallel")
    assert check_idx < parallel_idx

def test_pass48_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS48.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "segment-ID narrowing contract" in notes
    assert "uint32_t" in notes
    assert "pass 48 GPU tile preprocess" in audit
    assert "for pass_num in range(1, 49):" in audit
