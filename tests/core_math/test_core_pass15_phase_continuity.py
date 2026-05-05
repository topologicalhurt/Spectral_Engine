#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_phase_advance_diagnostics_without_model_override() -> None:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc, "no C compiler available"

    harness = r'''
#include "spectral_peak_estimator.h"
#include <math.h>
#include <stdio.h>

float spectral_window_interp_magsq_parabolic(float left_sq, float center_sq, float right_sq) {
    (void)left_sq;
    (void)center_sq;
    (void)right_sq;
    return 0.25f;
}

int main(void) {
    float row[5] = {0.0f, 1.0f, 4.0f, 1.0f, 0.0f};
    float next_row[5] = {0.0f, 1.0f, 4.0f, 1.0f, 0.0f};
    float phase[5] = {0.0f, 0.0f, 0.30f, 0.0f, 0.0f};
    float next_phase[5] = {0.0f, 0.0f, 2.55f, 0.0f, 0.0f};
    SpectralPeakEstimateInput input = {0};
    SpectralPeakEstimate out = {0};

    input.magsq_row = row;
    input.next_magsq_row = next_row;
    input.phase_row = phase;
    input.next_phase_row = next_phase;
    input.n_freqs = 5u;
    input.bin = 2u;
    input.curr_magsq = row[2];
    input.next_max_magsq = next_row[2];
    input.best_next_bin = 2;
    input.freq_step_omega = 0.1f;
    input.freq_step_df = 0.01f;
    input.inv_hop = 0.1f;
    input.hop_float = 10.0f;
    input.type = SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;

    if (!spectral_peak_estimate(&input, &out)) return 1;
    if (!(out.flags & SPECTRAL_PEAK_ESTIMATE_PHASE_ADVANCE_VALID)) return 2;
    if (!(out.flags & SPECTRAL_PEAK_ESTIMATE_PHASE_MODEL_CONSISTENT)) return 3;
    if (fabsf(out.bin_offset - 0.25f) > 1.0e-6f) return 4;
    if (fabsf(out.phase_bin_offset - 0.25f) > 1.0e-5f) return 5;
    if (fabsf(out.phase_omega - out.omega) > 1.0e-5f) return 6;
    if (fabsf(out.phase_error) > 1.0e-5f) return 7;

    next_phase[2] = 1.55f;
    if (!spectral_peak_estimate(&input, &out)) return 8;
    if (!(out.flags & SPECTRAL_PEAK_ESTIMATE_PHASE_ADVANCE_VALID)) return 9;
    if (out.flags & SPECTRAL_PEAK_ESTIMATE_PHASE_MODEL_CONSISTENT) return 10;

    input.next_phase_row = NULL;
    if (!spectral_peak_estimate(&input, &out)) return 11;
    if (out.flags & SPECTRAL_PEAK_ESTIMATE_PHASE_ADVANCE_VALID) return 12;

    return 0;
}
'''
    with tempfile.TemporaryDirectory(prefix="spectral-pass15-phase-") as tmp:
        tmp_path = Path(tmp)
        harness_c = tmp_path / "pass15_phase_contract.c"
        exe = tmp_path / "pass15_phase_contract"
        harness_c.write_text(harness, encoding="utf-8")
        subprocess.run(
            [
                cc,
                "-std=c11",
                "-I",
                str(ROOT / "spectral_engine/core"),
                "-I",
                str(ROOT / "spectral_engine/analysis"),
                str(ROOT / "spectral_engine/analysis/spectral_peak_estimator.c"),
                str(ROOT / "spectral_engine/core/spectral_fast_math.c"),
                str(harness_c),
                "-lm",
                "-o",
                str(exe),
            ],
            check=True,
            cwd=ROOT,
        )
        subprocess.run([str(exe)], check=True, cwd=ROOT)


def test_pass15_static_phase_wiring_present() -> None:
    est_h = (ROOT / "spectral_engine/analysis/spectral_peak_estimator.h").read_text()
    est_c = (ROOT / "spectral_engine/analysis/spectral_peak_estimator.c").read_text()
    track_h = (ROOT / "spectral_engine/analysis/spectral_peak_track.h").read_text()
    interp_h = (ROOT / "spectral_engine/analysis/spectral_peak_interp.h").read_text()
    interp_c = (ROOT / "spectral_engine/analysis/spectral_peak_interp.c").read_text()
    track_c = (ROOT / "spectral_engine/analysis/spectral_peak_track.c").read_text()
    fused_c = (ROOT / "spectral_engine/analysis/spectral_analysis_fused.c").read_text()
    config = (ROOT / "spectral_engine/core/spectral_config.h").read_text()

    assert "const float* next_phase_row;" in est_h
    assert "float phase_bin_offset;" in est_h
    assert "float phase_omega;" in est_h
    assert "float phase_error;" in est_h
    assert "SPECTRAL_PEAK_ESTIMATE_PHASE_ADVANCE_VALID" in est_h
    assert "SPECTRAL_PEAK_ESTIMATE_PHASE_MODEL_CONSISTENT" in est_h
    assert "SPECTRAL_PEAK_PHASE_CONSISTENCY_TOL_RADS" in config

    assert "spectral_peak_estimate_phase_advance" in est_c
    assert "spectral_peak_wrap_phase_pi" in est_c
    assert "phase_delta - model_omega * input->hop_float" in est_c

    assert "next_phase_row" in track_h
    assert "const float* __restrict__ next_phase_row" in interp_h
    assert "estimate_input.next_phase_row = next_phase_row;" in interp_c
    assert "spectral_tracker_emit_segment(tracker, tid, row, next_row, phase_row, next_phase_row" in track_c
    assert "fctx.next_phase_row = phase_next;" in fused_c
