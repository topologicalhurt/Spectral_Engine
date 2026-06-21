#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_peak_model_validation_profiles_compile_and_validate() -> None:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc, "no C compiler available"

    harness = r'''
#include "spectral_peak_model.h"
#include <stdio.h>

int main(void) {
    SpectralPeakModel model;
    SpectralResolvedPeakModel resolved;
    SpectralWindowDescriptor interp_only = {
        SPECTRAL_WINDOW_HANN,
        "interp-only",
        "Interp Only",
        0,
        spectral_window_interp_magsq_parabolic,
        0
    };
    SpectralWindowDescriptor bad_window = {
        SPECTRAL_WINDOW_HANN,
        "bad",
        "Bad",
        spectral_window_hann,
        0,
        spectral_window_peak_magsq_log_parabolic
    };
    SpectralWindowDescriptor bad_rect = {
        SPECTRAL_WINDOW_RECTANGULAR,
        "bad-rect",
        "Bad Rect",
        spectral_window_rectangular,
        spectral_window_interp_magsq_parabolic,
        spectral_window_peak_magsq_log_parabolic
    };

    model = spectral_peak_model_default();
    if (spectral_peak_model_resolve(&model, &resolved) != SPECTRAL_OK) return 1;
    if (!resolved.window || resolved.window->type != SPECTRAL_WINDOW_HANN) return 2;
    if (resolved.interp_magsq != spectral_window_interp_magsq_parabolic) return 3;
    if (resolved.peak_magsq != spectral_window_peak_magsq_log_parabolic) return 4;
    if (resolved.amplitude_policy != SPECTRAL_PEAK_AMP_POLICY_INTERP_BOUNDED) return 5;

    model = spectral_peak_model_for_window_type(SPECTRAL_WINDOW_RECTANGULAR);
    if (spectral_peak_model_resolve(&model, &resolved) != SPECTRAL_OK) return 6;
    if (!resolved.window || resolved.window->type != SPECTRAL_WINDOW_RECTANGULAR) return 7;
    if (resolved.peak_magsq != spectral_window_peak_magsq_center) return 8;
    if (resolved.amplitude_policy != SPECTRAL_PEAK_AMP_POLICY_CENTER) return 9;

    model = spectral_peak_model_default();
    model.estimator = SPECTRAL_PEAK_ESTIMATOR_QUINN_SECOND;
    if (spectral_peak_model_resolve(&model, &resolved) != SPECTRAL_OK) return 10;
    if (!(resolved.capabilities & SPECTRAL_PEAK_MODEL_CAP_COMPLEX_TRIPLET)) return 11;
    if (!(resolved.capabilities & SPECTRAL_PEAK_MODEL_CAP_PHASE_ROW)) return 12;

    model = spectral_peak_model_for_window(&bad_window);
    if (spectral_peak_model_validate(&model) == SPECTRAL_OK) return 13;

    model = spectral_peak_model_for_window(&bad_rect);
    model.amplitude_policy = SPECTRAL_PEAK_AMP_POLICY_INTERP_BOUNDED;
    if (spectral_peak_model_validate(&model) == SPECTRAL_OK) return 14;

    model = spectral_peak_model_for_window(&interp_only);
    if (spectral_peak_model_resolve(&model, &resolved) != SPECTRAL_OK) return 15;
    if (resolved.amplitude_policy != SPECTRAL_PEAK_AMP_POLICY_CENTER) return 16;
    if (resolved.peak_magsq != spectral_window_peak_magsq_center) return 17;

    model.amplitude_policy = SPECTRAL_PEAK_AMP_POLICY_INTERP_BOUNDED;
    if (spectral_peak_model_validate(&model) == SPECTRAL_OK) return 18;

    return 0;
}
'''

    with tempfile.TemporaryDirectory(prefix="spectral-pass18-model-") as tmp:
        tmp_path = Path(tmp)
        harness_c = tmp_path / "pass18_peak_model.c"
        exe = tmp_path / "pass18_peak_model"
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
                str(ROOT / "spectral_engine/analysis/spectral_peak_model.c"),
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


def test_tracker_invalid_model_preserves_existing_profile() -> None:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc, "no C compiler available"

    harness = r'''
#include "spectral_peak_track_internal.h"
#include <stdio.h>

static float quarter_offset(float left, float center, float right) {
    (void)left;
    (void)center;
    (void)right;
    return 0.25f;
}

int main(void) {
    SpectralWindowDescriptor desc = {
        SPECTRAL_WINDOW_HANN,
        "quarter",
        "Quarter",
        0,
        quarter_offset,
        0
    };
    SpectralTracker* tracker = spectral_tracker_create(1, 33u, 48000, 64, 1, -120.0f, 4.0f);
    SpectralPeakModel model = spectral_peak_model_for_window(&desc);
    SpectralPeakModel invalid;

    if (!tracker) return 1;
    if (spectral_tracker_set_peak_model(tracker, &model) != SPECTRAL_OK) return 2;
    if (tracker->interp_magsq != quarter_offset) return 3;
    if (tracker->amplitude_policy != SPECTRAL_PEAK_AMP_POLICY_CENTER) return 4;

    invalid = spectral_peak_model_from_resolved(&tracker->peak_model);
    invalid.estimator = SPECTRAL_PEAK_ESTIMATOR_COUNT;
    if (spectral_tracker_set_peak_model(tracker, &invalid) == SPECTRAL_OK) return 5;
    if (tracker->peak_model.window != &desc) return 6;
    if (tracker->interp_magsq != quarter_offset) return 7;
    if (tracker->amplitude_policy != SPECTRAL_PEAK_AMP_POLICY_CENTER) return 8;

    spectral_tracker_destroy(tracker);
    return 0;
}
'''

    with tempfile.TemporaryDirectory(prefix="spectral-pass18-tracker-model-") as tmp:
        tmp_path = Path(tmp)
        harness_c = tmp_path / "pass18_tracker_model.c"
        exe = tmp_path / "pass18_tracker_model"
        harness_c.write_text(harness, encoding="utf-8")
        link_flags = ["-framework", "Accelerate"] if sys.platform == "darwin" else []
        subprocess.run(
            [
                cc,
                "-std=c11",
                "-I",
                str(ROOT / "spectral_engine/core"),
                "-I",
                str(ROOT / "spectral_engine/runtime"),
                "-I",
                str(ROOT / "spectral_engine/analysis"),
                "-I",
                str(ROOT / "spectral_engine/runtime"),
                "-I",
                str(ROOT / "third_party/simde"),
                str(ROOT / "spectral_engine/analysis/spectral_peak_track.c"),
                str(ROOT / "spectral_engine/analysis/spectral_peak_model.c"),
                str(ROOT / "spectral_engine/analysis/spectral_peak_interp.c"),
                str(ROOT / "spectral_engine/analysis/spectral_peak_estimator.c"),
                str(ROOT / "spectral_engine/core/spectral_fast_math.c"),
                str(ROOT / "spectral_engine/core/spectral_windows.c"),
                str(ROOT / "spectral_engine/core/spectral_common.c"),
                str(ROOT / "spectral_engine/core/spectral_log.c"),
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


