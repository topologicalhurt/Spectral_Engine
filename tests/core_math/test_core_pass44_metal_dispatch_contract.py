#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_metal_dispatch_objects_are_checked_before_use() -> None:
    src = read("spectral_engine/synth/backends/gpu/metal/spectral_synth_metal.m")

    assert "if (!paramsBuffer)" in src
    assert "if (!cmdBuffer)" in src
    assert "if (!encoder)" in src
    assert "return_err = SPECTRAL_ERR_GPU_INIT;" in src

def test_metal_command_buffer_completion_status_is_checked_before_copyback() -> None:
    src = read("spectral_engine/synth/backends/gpu/metal/spectral_synth_metal.m")

    assert "[cmdBuffer status] != MTLCommandBufferStatusCompleted" in src
    assert "Metal: command buffer failed" in src

    status_idx = src.index("[cmdBuffer status] != MTLCommandBufferStatusCompleted")
    copy_idx = src.index("memcpy(out_buffer, [g_mtl.outputBuf contents], output_size)")
    assert status_idx < copy_idx

def test_pass44_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS44.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "Metal dispatch completion contract" in notes
    assert "stale output" in notes
    assert "pass 44 Metal dispatch" in audit
    assert "for pass_num in range(1, 45):" in audit
