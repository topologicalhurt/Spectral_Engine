#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")

def test_audio_read_proves_sndfile_frame_count_representable_before_allocation() -> None:
    src = read("spectral_engine/core/spectral_in.c")

    assert "spectral_sf_count_to_size" in src
    assert "if (!spectral_sf_count_to_size(sfinfo.frames, &frames))" in src
    assert "return SPECTRAL_ERR_OVERFLOW;" in src
    assert "spectral_size_mul(frames, channels, &total_samples)" in src
    assert "spectral_array_bytes(frames, sizeof(float), &mono_bytes)" in src
    assert "memcpy(mono, audio, mono_bytes)" in src
    assert "(size_t)sfinfo.frames * sizeof(float)" not in src
    assert "sfinfo.samplerate < SPECTRAL_MIN_SAMPLE_RATE" in src
    assert "sfinfo.samplerate > SPECTRAL_MAX_SAMPLE_RATE" in src

    read_idx = src.index("sf_readf_float(file, audio, sfinfo.frames)")
    repr_idx = src.index("if (!spectral_sf_count_to_size(sfinfo.frames, &frames))")
    assert repr_idx < read_idx

def test_audio_read_uses_size_t_loop_indices_after_shape_validation() -> None:
    src = read("spectral_engine/core/spectral_in.c")

    assert "for (size_t i = 0; i < frames; i++)" in src
    assert "for (size_t ch = 0; ch < channels; ch++)" in src
    assert "size_t base = i * channels;" in src
    assert "for (sf_count_t i = 0; i < sfinfo.frames; i++)" not in src

def test_audio_write_proves_size_t_frame_count_representable_for_libsndfile() -> None:
    src = read("spectral_engine/core/spectral_out.c")

    assert "spectral_size_to_sf_count" in src
    assert "if (!spectral_size_to_sf_count(num_frames, &frames_sf)" in src
    assert "spectral_size_mul(num_frames, (size_t)channels, &sample_count)" in src
    assert "info.frames = frames_sf;" in src
    assert "sf_writef_float(file, buffer, frames_sf)" in src
    assert "(sf_count_t)num_frames" not in src
    assert "sample_rate < SPECTRAL_MIN_SAMPLE_RATE" in src
    assert "sample_rate > SPECTRAL_MAX_SAMPLE_RATE" in src
    assert "close_status = sf_close(file)" in src

def test_pass36_docs_and_audit_track_contract() -> None:
    notes = read("docs/core_audit/PATCH_NOTES_PASS36.md")
    audit = read("tools/core_audit/core_static_audit.py")

    assert "audio file frame-count representability contract" in notes
    assert "sf_count_t" in notes
    assert "size_t" in notes
    assert "pass 36 audio IO" in audit
    assert "for pass_num in range(1, 37):" in audit
