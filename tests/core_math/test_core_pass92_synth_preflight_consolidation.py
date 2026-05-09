#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_synth_preflight_no_longer_recomputes_through_legacy_validate_inputs() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")
    header = read("spectral_engine/core/spectral_synth_internal.h")

    assert "synth_validate_inputs" not in src
    assert "synth_validate_inputs" not in header
    assert "SynthValidateResult" not in header
    assert "SYNTH_VALIDATE_FLOAT" not in header
    assert "SYNTH_VALIDATE_NATIVE" not in header

    assert "size_t preflight_out_bytes = 0;" in src
    assert "spectral_size_mul(out_len, elem_size, &preflight_out_bytes)" in src
    assert "memset(out_buffer, 0, preflight_out_bytes)" in src

def test_synth_preflight_uses_canonical_segment_contract_directly() -> None:
    src = read("spectral_engine/core/spectral_synth_internal.c")

    assert "synth_segment_payload_valid" not in src
    assert "spectral_segment_array_valid_for_synth(&sa)" in src

def test_unchecked_gpu_synth_params_pack_alias_is_removed() -> None:
    header = read("spectral_engine/core/spectral_synth_internal.h")
    metal = read("spectral_engine/synth/backends/gpu/metal/spectral_synth_metal.m")
    cuda = read("spectral_engine/synth/backends/gpu/cuda/spectral_synth_cuda.cu")

    assert "gpu_synth_params_pack_checked" in header
    assert "static inline GpuSynthParams gpu_synth_params_pack(" not in header
    assert "gpu_synth_params_pack(&" not in metal
    assert "gpu_synth_params_pack(&" not in cuda

def test_pass92_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS92.md")
    guidelines = read("docs/core_audit/KERNEL_PATCHING_GUIDELINES.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "synthesis preflight consolidation" in notes
    assert "derive shape once" in notes
    assert "Do not add alias wrappers" in guidelines
    assert "pass 92 synth preflight consolidation" in audit
    assert "for pass_num in range(1, 93):" in audit
