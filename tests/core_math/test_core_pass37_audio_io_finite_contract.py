#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_audio_read_rejects_nonfinite_input_samples_before_publishing_mono() -> None:
    src = read("spectral_engine/core/spectral_in.c")

    assert "spectral_f32_span_finite" in src
    assert "!spectral_is_finite_f32(samples[i])" in src
    assert "if (!spectral_f32_span_finite(audio, total_samples))" in src
    assert "return SPECTRAL_ERR_FILE_FORMAT;" in src

    finite_idx = src.index("if (!spectral_f32_span_finite(audio, total_samples))")
    mono_alloc_idx = src.index("mono = (float*)spectral_malloc_array(frames, sizeof(float));")
    assert finite_idx < mono_alloc_idx

def test_audio_read_downmix_uses_double_accumulation_and_checked_float_narrowing() -> None:
    src = read("spectral_engine/core/spectral_in.c")

    assert "#include <float.h>" in src
    assert "const double inv_ch = 1.0 / (double)channels;" in src
    assert "double sum = 0.0;" in src
    assert "mono_d < -(double)FLT_MAX" in src
    assert "mono_d > (double)FLT_MAX" in src
    assert "mono[i] = (float)mono_d;" in src
    assert "float sum = 0.0f;" not in src
    assert "float inv_ch = 1.0f / (float)channels;" not in src

def test_audio_write_rejects_nonfinite_output_samples_before_opening_file() -> None:
    src = read("spectral_engine/core/spectral_out.c")

    assert "spectral_f32_span_finite" in src
    assert "if (!spectral_f32_span_finite(buffer, sample_count))" in src
    assert "if (!spectral_f32_span_finite(mono, num_frames))" in src

    finite_idx = src.index("if (!spectral_f32_span_finite(buffer, sample_count))")
    open_idx = src.index("file = sf_open(path, SFM_WRITE, &info);")
    assert finite_idx < open_idx

def test_pass37_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS37.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "audio file finite-sample contract" in notes
    assert "before publishing" in notes
    assert "before opening" in notes
    assert "pass 37 audio IO" in audit
    assert "for pass_num in range(1, 38):" in audit
