#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def compile_and_run(source: str, name: str) -> None:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc, "no C compiler available"

    with tempfile.TemporaryDirectory(prefix=f"spectral-pass20-{name}-") as tmp:
        tmp_path = Path(tmp)
        harness_c = tmp_path / f"{name}.c"
        exe = tmp_path / name
        harness_c.write_text(source, encoding="utf-8")
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


def test_phase_capability_contract_and_requires_phase_row() -> None:
    harness = r'''
#include "spectral_peak_model.h"

int main(void) {
    SpectralPeakModel model = spectral_peak_model_default();
    SpectralResolvedPeakModel resolved = {0};

    model.phase_policy = SPECTRAL_PEAK_PHASE_POLICY_OBSERVE;
    if (spectral_peak_model_resolve(&model, &resolved) != SPECTRAL_OK) return 1;
    if (!(resolved.capabilities & SPECTRAL_PEAK_MODEL_CAP_PHASE_ROW)) return 2;
    if (!(resolved.capabilities & SPECTRAL_PEAK_MODEL_CAP_NEXT_PHASE_ROW)) return 3;
    if (!spectral_peak_model_has_capability(&model, SPECTRAL_PEAK_MODEL_CAP_PHASE_ROW)) return 4;
    if (!spectral_peak_model_has_capability(&model, SPECTRAL_PEAK_MODEL_CAP_NEXT_PHASE_ROW)) return 5;

    model.phase_policy = SPECTRAL_PEAK_PHASE_POLICY_IGNORE;
    if (spectral_peak_model_resolve(&model, &resolved) != SPECTRAL_OK) return 6;
    if (resolved.capabilities & SPECTRAL_PEAK_MODEL_CAP_PHASE_DIAGNOSTIC) return 7;
    if (spectral_peak_model_has_capability(&model, SPECTRAL_PEAK_MODEL_CAP_NEXT_PHASE_ROW)) return 8;

    model.estimator = SPECTRAL_PEAK_ESTIMATOR_QUINN_SECOND;
    model.phase_policy = SPECTRAL_PEAK_PHASE_POLICY_IGNORE;
    if (spectral_peak_model_resolve(&model, &resolved) != SPECTRAL_OK) return 9;
    if (!(resolved.capabilities & SPECTRAL_PEAK_MODEL_CAP_PHASE_ROW)) return 10;
    if (!spectral_peak_model_has_capability(&model, SPECTRAL_PEAK_MODEL_CAP_PHASE_ROW)) return 11;

    return 0;
}
'''
    compile_and_run(harness, "phase_capability")


def test_void_setters_are_transactional_on_invalid_mutations() -> None:
    harness = r'''
#include "spectral_peak_track_internal.h"

static float custom_offset(float left, float center, float right) {
    (void)left;
    (void)center;
    (void)right;
    return 0.25f;
}

int main(void) {
    SpectralWindowDescriptor custom = {
        SPECTRAL_WINDOW_HANN,
        "custom",
        "Custom",
        0,
        custom_offset,
        0
    };
    SpectralWindowDescriptor bad = {
        SPECTRAL_WINDOW_HANN,
        "bad",
        "Bad",
        0,
        0,
        spectral_window_peak_magsq_log_parabolic
    };
    SpectralTracker* tracker = spectral_tracker_create(1, 33u, 48000, 64, 1, -120.0f, 4.0f);
    SpectralPeakModel model = spectral_peak_model_for_window(&custom);

    if (!tracker) return 1;
    if (spectral_tracker_set_peak_model(tracker, &model) != SPECTRAL_OK) return 2;
    if (tracker->peak_model.window != &custom) return 3;
    if (tracker->interp_magsq != custom_offset) return 4;
    if (tracker->amplitude_policy != SPECTRAL_PEAK_AMP_POLICY_CENTER) return 5;

    /* Invalid window mutation through legacy void setter must preserve custom. */
    spectral_tracker_set_window_descriptor(tracker, &bad);
    if (tracker->peak_model.window != &custom) return 6;
    if (tracker->interp_magsq != custom_offset) return 7;
    if (tracker->amplitude_policy != SPECTRAL_PEAK_AMP_POLICY_CENTER) return 8;

    /* Invalid enum mutations through legacy void setters must also preserve. */
    spectral_tracker_set_peak_estimator(tracker, SPECTRAL_PEAK_ESTIMATOR_COUNT);
    if (tracker->peak_model.window != &custom) return 9;
    if (tracker->interp_magsq != custom_offset) return 10;

    spectral_tracker_set_phase_policy(tracker, (SpectralPeakPhasePolicy)999);
    if (tracker->peak_model.window != &custom) return 11;
    if (tracker->interp_magsq != custom_offset) return 12;

    spectral_tracker_set_amplitude_policy(tracker, (SpectralPeakAmplitudePolicy)999);
    if (tracker->peak_model.window != &custom) return 13;
    if (tracker->interp_magsq != custom_offset) return 14;
    if (tracker->amplitude_policy != SPECTRAL_PEAK_AMP_POLICY_CENTER) return 15;

    spectral_tracker_destroy(tracker);
    return 0;
}
'''
    compile_and_run(harness, "transactional_setters")


