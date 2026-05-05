#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_phase_policy_observe_vs_reject() -> None:
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

static void fill_input(SpectralPeakEstimateInput* input,
                       float* row, float* next_row,
                       float* phase, float* next_phase,
                       SpectralPeakPhasePolicy policy) {
    input->magsq_row = row;
    input->next_magsq_row = next_row;
    input->phase_row = phase;
    input->next_phase_row = next_phase;
    input->n_freqs = 5u;
    input->bin = 2u;
    input->curr_magsq = row[2];
    input->next_max_magsq = next_row[2];
    input->best_next_bin = 2;
    input->freq_step_omega = 0.1f;
    input->freq_step_df = 0.01f;
    input->inv_hop = 0.1f;
    input->hop_float = 10.0f;
    input->type = SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;
    input->phase_policy = policy;
}

int main(void) {
    float row[5] = {0.0f, 1.0f, 4.0f, 1.0f, 0.0f};
    float next_row[5] = {0.0f, 1.0f, 4.0f, 1.0f, 0.0f};
    float phase[5] = {0.0f, 0.0f, 0.30f, 0.0f, 0.0f};
    float next_phase[5] = {0.0f, 0.0f, 1.55f, 0.0f, 0.0f};
    SpectralPeakEstimateInput input = {0};
    SpectralPeakEstimate out = {0};

    fill_input(&input, row, next_row, phase, next_phase, SPECTRAL_PEAK_PHASE_POLICY_IGNORE);
    if (!spectral_peak_estimate(&input, &out)) return 1;
    if (out.flags & SPECTRAL_PEAK_ESTIMATE_PHASE_ADVANCE_VALID) return 2;

    fill_input(&input, row, next_row, phase, next_phase, SPECTRAL_PEAK_PHASE_POLICY_OBSERVE);
    if (!spectral_peak_estimate(&input, &out)) return 3;
    if (!(out.flags & SPECTRAL_PEAK_ESTIMATE_PHASE_ADVANCE_VALID)) return 4;
    if (out.flags & SPECTRAL_PEAK_ESTIMATE_PHASE_MODEL_CONSISTENT) return 5;

    fill_input(&input, row, next_row, phase, next_phase, SPECTRAL_PEAK_PHASE_POLICY_REJECT_INCONSISTENT);
    if (spectral_peak_estimate(&input, &out)) return 6;

    next_phase[2] = 2.55f; /* consistent with omega=(2.25*0.1) and hop=10 */
    fill_input(&input, row, next_row, phase, next_phase, SPECTRAL_PEAK_PHASE_POLICY_REJECT_INCONSISTENT);
    if (!spectral_peak_estimate(&input, &out)) return 7;
    if (!(out.flags & SPECTRAL_PEAK_ESTIMATE_PHASE_MODEL_CONSISTENT)) return 8;

    return 0;
}
'''
    with tempfile.TemporaryDirectory(prefix="spectral-pass16-phase-policy-") as tmp:
        tmp_path = Path(tmp)
        harness_c = tmp_path / "pass16_phase_policy.c"
        exe = tmp_path / "pass16_phase_policy"
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


def test_pass16_static_phase_policy_wiring_present() -> None:
    est_h = (ROOT / "spectral_engine/analysis/spectral_peak_estimator.h").read_text()
    est_c = (ROOT / "spectral_engine/analysis/spectral_peak_estimator.c").read_text()
    track_h = (ROOT / "spectral_engine/analysis/spectral_peak_track.h").read_text()
    track_i = (ROOT / "spectral_engine/analysis/spectral_peak_track_internal.h").read_text()
    interp_c = (ROOT / "spectral_engine/analysis/spectral_peak_interp.c").read_text()
    track_c = (ROOT / "spectral_engine/analysis/spectral_peak_track.c").read_text()
    config = (ROOT / "spectral_engine/core/spectral_config.h").read_text()

    assert "SpectralPeakPhasePolicy" in est_h
    assert "SPECTRAL_PEAK_PHASE_POLICY_REJECT_INCONSISTENT" in est_h
    assert "SpectralPeakPhasePolicy phase_policy;" in est_h
    assert "SPECTRAL_PEAK_PHASE_POLICY_DEFAULT" in config
    assert "phase_policy == SPECTRAL_PEAK_PHASE_POLICY_REJECT_INCONSISTENT" in est_c

    assert "SpectralPeakPhasePolicy phase_policy;" in track_i
    assert "spectral_tracker_set_phase_policy" in track_h
    assert "tracker->phase_policy = SPECTRAL_PEAK_PHASE_POLICY_DEFAULT;" in track_c
    assert "estimate_input.phase_policy = tracker->phase_policy;" in interp_c
