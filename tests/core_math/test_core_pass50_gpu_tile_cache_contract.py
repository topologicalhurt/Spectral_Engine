#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_gpu_tile_cache_set_fails_closed_on_invalid_pointer_count_shapes() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "gpu_tile_data_refs_valid" in src
    assert "if (num_tiles == 0u || (total_refs > 0u && (!ranges || !segment_ids)))" in src
    assert "gpu_tile_cache_clear();" in src

def test_gpu_tile_cache_hit_validates_refs_against_current_segment_count() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "gpu_tile_cache_try_get(stretch, out_len, out_td)" in src
    assert "gpu_tile_data_refs_valid(out_td->ranges" in src
    assert "(uint32_t)sa.count" in src
    assert "return SPECTRAL_OK;" in src

def test_pass50_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS50.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "GPU tile cache validation contract" in notes
    assert "process-local" in notes
    assert "pass 50 GPU tile cache" in audit
    assert "for pass_num in range(1, 51):" in audit
