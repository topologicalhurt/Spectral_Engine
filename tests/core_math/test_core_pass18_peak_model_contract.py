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


def test_pass18_static_peak_model_wiring_present() -> None:
    manifest = (ROOT / "spectral_engine/cmake/source-manifest.cmake").read_text()
    model_h = (ROOT / "spectral_engine/analysis/spectral_peak_model.h").read_text()
    model_c = (ROOT / "spectral_engine/analysis/spectral_peak_model.c").read_text()
    track_h = (ROOT / "spectral_engine/analysis/spectral_peak_track.h").read_text()
    track_i = (ROOT / "spectral_engine/analysis/spectral_peak_track_internal.h").read_text()
    track_c = (ROOT / "spectral_engine/analysis/spectral_peak_track.c").read_text()
    audit = (ROOT / "tools/core_audit/core_static_audit.py").read_text()

    assert "spectral_peak_model.c" in manifest
    assert "SpectralPeakModel" in model_h
    assert "SpectralResolvedPeakModel" in model_h
    assert "spectral_peak_model_validate" in model_h
    assert "SPECTRAL_PEAK_MODEL_CAP_COMPLEX_TRIPLET" in model_h
    assert "spectral_peak_model_for_window_type" in model_c
    assert "SPECTRAL_WINDOW_RECTANGULAR" in model_c
    assert "spectral_window_peak_magsq_log_parabolic" in model_c

    assert '#include "spectral_peak_model.h"' in track_h
    assert "spectral_tracker_set_peak_model" in track_h
    assert "SpectralResolvedPeakModel peak_model;" in track_i
    assert "spectral_tracker_apply_peak_model" in track_c
    assert "spectral_peak_model_resolve" in track_c
    assert "tracker->peak_model = resolved;" in track_c
    assert "pass 18 peak model" in audit
