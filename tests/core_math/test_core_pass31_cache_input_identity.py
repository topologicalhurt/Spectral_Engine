#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_pipeline_cache_key_uses_input_content_hash_not_basename_only() -> None:
    src = read("spectral_engine/cmd/cli/spectral_cli_pipeline.c")

    assert '#include "spectral_hash_xx32_xx3.h"' in src
    assert "pipeline_hash_input_file" in src
    assert "SpectralHashFileMethod method = SPECTRAL_HASH_FILE_METHOD_ZERO_INIT;" in src
    assert "spectral_hash_file_method_init(&method, SPECTRAL_HASH_FILE_STREAM)" in src
    assert "spectral_hash_file_method_consume_file(&method, f)" in src
    assert "spectral_hash_file_method_digest(&method, out_digest)" in src
    assert 'snprintf(input_id, sizeof(input_id), "%s|xxh=%016llx"' in src
    assert "spectral_seg_cache_key(input_id, opts->n_fft, opts->hop" in src
    assert "spectral_seg_cache_key(stem, opts->n_fft" not in src

def test_core_cache_key_requires_caller_provided_input_identity() -> None:
    src = read("spectral_engine/core/spectral_seg_cache.c")
    header = read("spectral_engine/core/spectral_seg_cache.h")

    assert "uint64_t spectral_seg_cache_key(const char* input_id" in src
    assert "spectral_is_empty_string(input_id)" in src
    assert "input_id," in src
    assert "not just its basename" in src
    assert "audio input identity string" in header
    compute_section = header[header.index("/* Compute cache key"):header.index("SpectralError spectral_seg_cache_lookup")]
    assert "basename" not in compute_section

def test_pass31_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS31.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "segment cache input identity contract" in notes
    assert "same basename" in notes
    assert "content digest" in notes
    assert "pass 31 segment cache key" in audit
    assert "for pass_num in range(1, 32):" in audit
