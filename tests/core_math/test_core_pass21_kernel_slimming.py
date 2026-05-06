#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


def test_pass21_removes_low_value_alias_wrappers() -> None:
    oscillator = read("spectral_engine/core/oscillator.c")
    synth_internal = read("spectral_engine/core/spectral_synth_internal.h")
    synth_internal_c = read("spectral_engine/core/spectral_synth_internal.c")
    segment_math_h = read("spectral_engine/core/spectral_segment_math.h")
    segment_mt = read("spectral_engine/core/spectral_segment_mt.c")
    spectral_in = read("spectral_engine/core/spectral_in.c")
    spectral_out = read("spectral_engine/core/spectral_out.c")
    processing_chain = read("spectral_engine/analysis/spectral_processing_chain.c")
    synth_cpu = read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")
    synth_sim = read("spectral_engine/synth/backends/sim/spectral_synth_simulation.c")
    manifest = read("spectral_engine/cmake/source-manifest.cmake")
    peak_model_h = read("spectral_engine/analysis/spectral_peak_model.h")
    peak_track = read("spectral_engine/analysis/spectral_peak_track.c")
    perf = read("spectral_engine/runtime/spectral_perf.c")

    assert "static inline float osc_sine" not in oscillator
    assert "static inline float osc_saw" not in oscillator
    assert "spectral_osc_sine," in oscillator
    assert "spectral_osc_pwm" in oscillator

    assert "compute_phase(" not in synth_internal
    assert "compute_amplitude(" not in synth_internal
    assert "spectral_segment_phase_at_index_f32" not in segment_math_h
    assert "spectral_segment_amp_at_index_f32" not in segment_math_h
    assert "spectral_segment_math.c" not in manifest

    assert "spectral_peak_model_has_capability" in peak_model_h
    assert "spectral_peak_model_requires_phase_row" not in peak_model_h
    assert "spectral_peak_model_requires_next_phase_row" not in peak_model_h
    assert perf.count("void perf_get_cpu_time") == 1
    assert not (ROOT / "spectral_engine/analysis/spectral_peak_chain.c").exists()
    assert not (ROOT / "spectral_engine/analysis/spectral_peak_chain.h").exists()
    assert "static void spectral_tracker_free_segment_storage" in peak_track
    assert "spectral_calloc_array((size_t)n_threads, sizeof(TrackSegment*))" in peak_track
    assert peak_track.count("spectral_tracker_free_segment_storage(tracker);") >= 3
    assert synth_internal_c.count("lp.amp = s->amp;") == 1
    assert "static void gpu_tile_preprocess_scratch_free" in synth_internal_c
    assert synth_internal_c.count("gpu_tile_preprocess_scratch_free(") == 3
    assert "static void segment_array_mt_apply_pending_locked" in segment_mt
    assert segment_mt.count("segment_array_mt_apply_pending_locked(sa);") == 2
    assert "float* audio = malloc(" not in spectral_in
    assert "float* mono = malloc(" not in spectral_in
    assert "float* stereo = malloc(" not in spectral_out
    assert "spectral_size_add(n, 1u, &buf_len)" in processing_chain
    assert "tb.arena = spectral_calloc_array(arena_size, 1u)" in synth_cpu
    assert "memset(tb.arena" not in synth_cpu
    assert "sim_segs = (SimSegment*)spectral_malloc_array" in synth_sim
    assert "accum = (int64_t*)spectral_calloc_array" in synth_sim


def test_pass21_docs_record_alias_wrapper_rule() -> None:
    ai_canon = read("docs/core_audit/AI_CANON.md")
    findings = read("docs/core_audit/DISCIPLINE_FINDINGS.md")
    notes = read("docs/core_audit/PATCH_NOTES_PASS21.md")

    assert "Alias wrappers are not architecture" in ai_canon
    assert "Thin alias wrappers" in findings
    assert "Do not add a wrapper whose only job is to rename another helper" in notes
    assert "dead duplicate kernel files" in notes
    assert "central cleanup helpers" in notes
    assert "safe allocation helpers" in notes
