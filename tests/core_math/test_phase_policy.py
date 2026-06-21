#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
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

static float quarter_offset(float left_sq, float center_sq, float right_sq) {
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
    input->interp_magsq = quarter_offset;
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
        link_flags = ["-framework", "Accelerate"] if sys.platform == "darwin" else []
        subprocess.run(
            [
                cc,
                "-std=c11",
                "-I",
                str(ROOT / "spectral_engine/core"),
                "-I",
                str(ROOT / "spectral_engine/analysis"),
                "-I",
                str(ROOT / "spectral_engine/runtime"),
                str(ROOT / "spectral_engine/analysis/spectral_peak_estimator.c"),
                str(ROOT / "spectral_engine/core/spectral_fast_math.c"),
                str(ROOT / "spectral_engine/core/spectral_windows.c"),
                str(harness_c),
                "-lm",
                *link_flags,
                "-o",
                str(exe),
            ],
            check=True,
            cwd=ROOT,
        )
        subprocess.run([str(exe)], check=True, cwd=ROOT)


