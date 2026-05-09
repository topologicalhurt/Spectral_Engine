#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_seg_cache_uses_canonical_contracts_directly_without_alias_wrappers() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")

    assert "seg_cache_segments_valid(" not in src
    assert "seg_cache_gpu_segments_match(" not in src
    assert "seg_cache_tile_layout_words_valid(" not in src
    assert "spectral_segment_array_payload_valid(mapped_segments" in src
    assert "spectral_segment_gpu_array_matches_segments(mapped_segments" in src
    assert "spectral_gpu_tile_layout_words_valid(ranges_base" in src
    assert "spectral_segment_array_payload_valid(sa->segs" in src

def test_synth_tile_cache_uses_canonical_tile_contract_directly() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "gpu_tile_data_refs_valid(" not in src
    assert "spectral_gpu_tile_layout_words_valid((const void*)out_td->ranges" in src

def test_finite_span_call_sites_are_direct_not_alias_wrappers() -> None:
    combined = (
        read("spectral_engine/core/spectral_in.c") +
        read("spectral_engine/core/spectral_out.c") +
        read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c") +
        read("spectral_engine/core/spectral_wavetable.c") +
        read("spectral_engine/core/spectral_windows.c")
    )

    forbidden = [
        "static int spectral_audio_samples_finite(",
        "static int spectral_float_buffer_all_finite(",
        "static int synth_float_output_finite(",
        "static int wavetable_float_samples_finite(",
        "static int spectral_window_samples_valid(",
    ]
    for needle in forbidden:
        assert needle not in combined

    assert combined.count("spectral_f32_span_finite(") >= 6

def test_guidelines_and_audit_track_no_alias_policy() -> None:
    guidelines = read("docs/core_audit/KERNEL_PATCHING_GUIDELINES.md")
    notes = read("docs/core_audit/PATCH_NOTES_PASS91.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "Do not add alias wrappers" in guidelines
    assert "Kernel patches must be strategically minimal" in guidelines
    assert "contract routing and alias-wrapper removal" in notes
    assert "pass 91 contract routing" in audit
    assert "for pass_num in range(1, 92):" in audit
