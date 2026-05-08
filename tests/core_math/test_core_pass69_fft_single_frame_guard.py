#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_fft_single_frame_has_shared_argument_validator() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_fft.c")

    assert "spectral_fft_single_frame_args_valid" in src
    assert "tid < 0 || tid >= res->n_threads" in src
    assert "hop <= 0" in src
    assert "!res->fft_plans[tid]" in src or "!res->fft_setups[tid]" in src
    assert "spectral_fft_single_frame_clear" in src

def test_fft_single_frame_guards_both_backend_definitions_before_res_deref() -> None:
    src = read("spectral_engine/analysis/spectral_analysis_fft.c")

    assert src.count("if (!spectral_fft_single_frame_args_valid(res, tid, audio, hop, window_func, out_magsq))") == 2
    assert src.count("spectral_fft_single_frame_clear(out_magsq, out_phases, res ? res->n_freqs : 0u, out_frame_max);") == 2

def test_pass69_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS69.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "FFT single-frame helper boundary contract" in notes
    assert "thread ID" in notes
    assert "pass 69 FFT single frame" in audit
    assert "for pass_num in range(1, 70):" in audit
