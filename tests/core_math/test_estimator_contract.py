#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
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


