#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_validated_estimator_rejects_invalid_scalar_context_and_nan_triplets() -> None:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc, "no C compiler available"

    harness = r'''
#include "spectral_peak_estimator.h"
#include <math.h>
#include <stdio.h>

float spectral_window_interp_magsq_parabolic(float left_sq, float center_sq, float right_sq) {
    float log_l = logf(fmaxf(left_sq, SPECTRAL_TRACK_LOG_FLOOR));
    float log_c = logf(fmaxf(center_sq, SPECTRAL_TRACK_LOG_FLOOR));
    float log_r = logf(fmaxf(right_sq, SPECTRAL_TRACK_LOG_FLOOR));
    float denom = log_l - 2.0f * log_c + log_r;
    if (fabsf(denom) < SPECTRAL_TRACK_PARABOLIC_DENOM_EPS) return 0.0f;
    return 0.5f * (log_l - log_r) / denom;
}

static int run_case(float left, float center, float right,
                    float freq_step_omega, float freq_step_df, float inv_hop) {
    float magsq[3] = {left, center, right};
    float phase[3] = {0.0f, 0.1f, 0.2f};
    SpectralPeakEstimateInput input = {0};
    SpectralPeakEstimate out = {0};

    input.magsq_row = magsq;
    input.phase_row = phase;
    input.n_freqs = 3u;
    input.bin = 1u;
    input.curr_magsq = center;
    input.next_max_magsq = center;
    input.best_next_bin = 1;
    input.freq_step_omega = freq_step_omega;
    input.freq_step_df = freq_step_df;
    input.inv_hop = inv_hop;
    input.type = SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;

    return spectral_peak_estimate_validated(&input, &out);
}

int main(void) {
    if (!run_case(1.0f, 4.0f, 1.0f, 2.0f, 0.5f, 0.25f)) return 1;

    if (run_case(NAN, 4.0f, 1.0f, 2.0f, 0.5f, 0.25f)) return 2;
    if (run_case(-1.0f, 4.0f, 1.0f, 2.0f, 0.5f, 0.25f)) return 3;
    if (run_case(1.0f, NAN, 1.0f, 2.0f, 0.5f, 0.25f)) return 4;
    if (run_case(1.0f, 4.0f, 1.0f, NAN, 0.5f, 0.25f)) return 5;
    if (run_case(1.0f, 4.0f, 1.0f, 2.0f, NAN, 0.25f)) return 6;
    if (run_case(1.0f, 4.0f, 1.0f, 2.0f, 0.5f, NAN)) return 7;

    return 0;
}
'''

    with tempfile.TemporaryDirectory(prefix="spectral-pass12-estimator-") as tmp:
        tmp_path = Path(tmp)
        harness_c = tmp_path / "pass12_estimator_contract.c"
        exe = tmp_path / "pass12_estimator_contract"
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


def test_pass12_static_estimator_guards_present() -> None:
    src = (ROOT / "spectral_engine/analysis/spectral_peak_estimator.c").read_text()
    assert "The validated entry point means the tracker has already accepted" in src
    assert "Keep these scalar-contract checks unconditional" in src
    assert "(void)neighborhood_validated;" in src
    assert "if (!spectral_peak_finite_nonnegative(magsq) || !isfinite(phase))" in src
    assert "custom descriptor callback" in src
