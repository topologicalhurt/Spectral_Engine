#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_peak_df_uses_next_subbin_offset_when_available_and_coarse_fallback_when_not() -> None:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc, "no C compiler available"

    harness = r'''
#include "spectral_peak_estimator.h"
#include <math.h>
#include <stdio.h>

static float pass14_offset(float left_sq, float center_sq, float right_sq) {
    (void)left_sq;
    (void)right_sq;
    return (center_sq > 5.0f) ? -0.25f : 0.25f;
}

int main(void) {
    float row[5] = {0.0f, 1.0f, 4.0f, 1.0f, 0.0f};
    float next_row[5] = {0.0f, 1.0f, 2.0f, 9.0f, 2.0f};
    float phases[5] = {0};
    SpectralPeakEstimateInput input = {0};
    SpectralPeakEstimate out = {0};

    input.magsq_row = row;
    input.next_magsq_row = next_row;
    input.phase_row = phases;
    input.n_freqs = 5u;
    input.bin = 2u;
    input.curr_magsq = row[2];
    input.next_max_magsq = next_row[3];
    input.best_next_bin = 3;
    input.freq_step_omega = 10.0f;
    input.freq_step_df = 2.0f;
    input.inv_hop = 0.25f;
    input.interp_magsq = pass14_offset;
    input.type = SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;

    if (!spectral_peak_estimate(&input, &out)) return 1;
    if (fabsf(out.bin_offset - 0.25f) > 1.0e-6f) return 2;
    if (fabsf(out.next_bin_offset - (-0.25f)) > 1.0e-6f) return 3;
    if (fabsf(out.df - 1.0f) > 1.0e-6f) return 4;
    if (!(out.flags & SPECTRAL_PEAK_ESTIMATE_TEMPORAL_DF_REFINED)) return 5;
    if (!(out.flags & SPECTRAL_PEAK_ESTIMATE_NEXT_BIN_OFFSET_VALID)) return 6;

    input.next_magsq_row = NULL;
    if (!spectral_peak_estimate(&input, &out)) return 7;
    if (fabsf(out.df - 2.0f) > 1.0e-6f) return 8;
    if (out.flags & SPECTRAL_PEAK_ESTIMATE_TEMPORAL_DF_REFINED) return 9;

    next_row[4] = 20.0f;
    input.next_magsq_row = next_row;
    if (!spectral_peak_estimate(&input, &out)) return 10;
    if (fabsf(out.df - 2.0f) > 1.0e-6f) return 11;
    if (out.flags & SPECTRAL_PEAK_ESTIMATE_TEMPORAL_DF_REFINED) return 12;

    return 0;
}
'''
    with tempfile.TemporaryDirectory(prefix="spectral-pass14-df-") as tmp:
        tmp_path = Path(tmp)
        harness_c = tmp_path / "pass14_df_contract.c"
        exe = tmp_path / "pass14_df_contract"
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


