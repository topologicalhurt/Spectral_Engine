#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_canonical_contract_header_exists_and_owns_reusable_invariants() -> None:
    header = read("spectral_engine/core/spectral_contracts.h")

    assert "spectral_f32_span_finite" in header
    assert "spectral_segment_payload_valid" in header
    assert "spectral_segment_valid_for_synth" in header
    assert "spectral_segment_gpu_array_matches_segments" in header
    assert "spectral_gpu_tile_layout_words_valid" in header

def test_segment_cache_delegates_payload_validation_to_canonical_contracts() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert '#include "spectral_contracts.h"' in src
    assert "spectral_segment_array_payload_valid(mapped_segments" in src
    assert "spectral_segment_gpu_array_matches_segments(mapped_segments" in src
    assert "spectral_gpu_tile_layout_words_valid(ranges_base" in src
    assert "seg_cache_segments_valid(" not in src

def test_synth_preflight_and_loop_use_canonical_segment_contracts() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert '#include "spectral_contracts.h"' in src
    assert "spectral_segment_array_valid_for_synth(&sa)" in src
    assert "if (!spectral_segment_valid_for_synth(s))" in src
    assert "spectral_gpu_tile_layout_words_valid((const void*)out_td->ranges" in src

def test_finite_span_wrappers_delegate_to_canonical_contract() -> None:
    combined = (
        read("spectral_engine/core/spectral_in.c") +
        read("spectral_engine/core/spectral_out.c") +
        read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c") +
        read("spectral_engine/core/spectral_wavetable.c") +
        read("spectral_engine/core/spectral_windows.c")
    )

    assert combined.count("return spectral_f32_span_finite(") >= 4

def test_phase_c_docs_and_audit_are_present() -> None:
    contracts = read("docs/core_audit/CORE_CONTRACTS.md")
    ownership = read("docs/core_audit/VALIDATION_OWNERSHIP.md")
    notes = read("docs/core_audit/PATCH_NOTES_PHASE_C.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "Validation belongs at boundaries" in contracts
    assert "Boundary-owned checks" in ownership
    assert "contract consolidation and deduplication" in notes
    assert "Phase C contract consolidation" in audit
