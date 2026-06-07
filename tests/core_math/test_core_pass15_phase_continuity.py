#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
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

static float quarter_offset(float left_sq, float center_sq, float right_sq) {
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
    input.interp_magsq = quarter_offset;
    input.type = SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;
    input.phase_policy = SPECTRAL_PEAK_PHASE_POLICY_DEFAULT;

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


