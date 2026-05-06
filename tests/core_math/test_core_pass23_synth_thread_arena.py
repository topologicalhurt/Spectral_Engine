#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


def test_thread_buffer_arena_alignment_uses_checked_arithmetic() -> None:
    src = read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")

    assert "SpectralError* out_error" in src
    assert "size_t stride_with_padding = 0;" in src
    assert "size_t align_mask = 0;" in src
    assert "spectral_size_add(tb.buf_size, align_mask, &stride_with_padding)" in src
    assert "tb.buf_stride = stride_with_padding & ~align_mask;" in src
    assert "tb.buf_stride = (tb.buf_size + SPECTRAL_CACHE_ALIGN - 1)" not in src
    assert "spectral_size_add(arena_payload_size, align_mask, &arena_size)" in src
    assert "base > UINTPTR_MAX - (uintptr_t)align_mask" in src
    assert "Checked align-up" in src
    assert "Manual aligned-base adjustment" in src
    assert "Pointer-add guard" in src


def test_thread_buffer_errors_propagate_to_cpu_driver() -> None:
    src = read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")

    assert "SpectralError tb_err = SPECTRAL_OK;" in src
    assert "thread_buffers_alloc(n_parts, out_len, elem_size, &tb_err)" in src
    assert "return tb_err != SPECTRAL_OK ? tb_err : SPECTRAL_ERR_MEMORY;" in src
    assert "return SPECTRAL_ERR_OVERFLOW;" in src
    assert "memset(out_buffer, 0, out_bytes);" in src


def test_pass23_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS23.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "cache-line stride overflow" in notes
    assert "Reviewer Walkthrough" in notes
    assert "checked align-up" in notes
    assert "pointer-add guard" in notes
    assert "SPECTRAL_ERR_MEMORY" in notes
    assert "SPECTRAL_ERR_OVERFLOW" in notes
    assert "pass 23 CPU synth arena stride alignment" in audit
    assert "pass 23 source comments must explain checked align-up" in audit
    assert "pass 23 walkthrough must document overflow-vs-memory failure" in audit
    assert "for pass_num in range(1, 24):" in audit
