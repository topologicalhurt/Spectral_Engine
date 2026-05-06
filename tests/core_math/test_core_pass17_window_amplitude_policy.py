#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_window_descriptor_controls_amplitude_peak_height() -> None:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc, "no C compiler available"

    harness = r'''
#include "spectral_peak_estimator.h"
#include "spectral_windows.h"
#include <math.h>
#include <stdio.h>

int main(void) {
    float row[3] = {1.0f, 4.0f, 2.0f};
    float phases[3] = {0.0f, 0.0f, 0.0f};
    const SpectralWindowDescriptor* hann = spectral_window_descriptor(SPECTRAL_WINDOW_HANN);
    const SpectralWindowDescriptor* rect = spectral_window_descriptor(SPECTRAL_WINDOW_RECTANGULAR);
    SpectralPeakEstimateInput input = {0};
    SpectralPeakEstimate out_hann = {0};
    SpectralPeakEstimate out_rect = {0};

    if (!hann || !rect || !hann->peak_magsq || !rect->peak_magsq) return 1;

    input.magsq_row = row;
    input.phase_row = phases;
    input.n_freqs = 3u;
    input.bin = 1u;
    input.curr_magsq = row[1];
    input.next_max_magsq = row[1];
    input.best_next_bin = 1;
    input.freq_step_omega = 1.0f;
    input.freq_step_df = 1.0f;
    input.inv_hop = 1.0f;
    input.hop_float = 1.0f;
    input.type = SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;
    input.phase_policy = SPECTRAL_PEAK_PHASE_POLICY_IGNORE;
    input.amplitude_policy = SPECTRAL_PEAK_AMP_POLICY_INTERP_BOUNDED;

    input.interp_magsq = hann->interp_magsq;
    input.peak_magsq = hann->peak_magsq;
    if (!spectral_peak_estimate(&input, &out_hann)) return 2;
    if (!(out_hann.flags & SPECTRAL_PEAK_ESTIMATE_AMP_INTERP_VALID)) return 3;
    if (!(out_hann.amp > 2.0f)) return 4;

    input.interp_magsq = rect->interp_magsq;
    input.peak_magsq = rect->peak_magsq;
    if (!spectral_peak_estimate(&input, &out_rect)) return 5;
    if (!(out_rect.flags & SPECTRAL_PEAK_ESTIMATE_AMP_INTERP_VALID)) return 6;
    if (fabsf(out_rect.amp - 2.0f) > 1.0e-6f) return 7;

    return 0;
}
'''

    with tempfile.TemporaryDirectory(prefix="spectral-pass17-window-amp-") as tmp:
        tmp_path = Path(tmp)
        harness_c = tmp_path / "pass17_window_amp.c"
        exe = tmp_path / "pass17_window_amp"
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


def test_pass17_static_window_amplitude_policy_present() -> None:
    windows_h = (ROOT / "spectral_engine/core/spectral_windows.h").read_text()
    windows_c = (ROOT / "spectral_engine/core/spectral_windows.c").read_text()
    est_h = (ROOT / "spectral_engine/analysis/spectral_peak_estimator.h").read_text()
    est_c = (ROOT / "spectral_engine/analysis/spectral_peak_estimator.c").read_text()
    track_c = (ROOT / "spectral_engine/analysis/spectral_peak_track.c").read_text()

    assert "SpectralWindowPeakMagsqFn" in windows_h
    assert "peak_magsq;" in windows_h
    assert "spectral_window_peak_magsq_log_parabolic" in windows_c
    assert "spectral_window_peak_magsq_center" in windows_c
    assert "{ SPECTRAL_WINDOW_RECTANGULAR" in windows_c
    assert "spectral_window_peak_magsq_center" in windows_c

    assert "SpectralWindowPeakMagsqFn peak_magsq;" in est_h
    assert "spectral_peak_window_peak_magsq" in est_c
    assert "input->peak_magsq ? input->peak_magsq : spectral_window_peak_magsq_center" in est_c
    assert "tracker->peak_magsq" in track_c
