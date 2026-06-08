#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TWO_PI = 2.0 * math.pi
HANN_LOG_PARABOLIC_MAX_ERROR_BINS = 0.02


def hann(n: int, size: int) -> float:
    return 0.5 * (1.0 - math.cos(TWO_PI * n / (size - 1)))


def dft_bin_magsq(size: int, tone_bin: float, bin_index: int) -> float:
    re = 0.0
    im = 0.0
    for n in range(size):
        x = math.cos(TWO_PI * tone_bin * n / size) * hann(n, size)
        angle = TWO_PI * bin_index * n / size
        re += x * math.cos(angle)
        im -= x * math.sin(angle)
    return re * re + im * im


def log_power_parabolic_three_log(left: float, center: float, right: float) -> float:
    log_l = math.log(max(left, 1.0e-30))
    log_c = math.log(max(center, 1.0e-30))
    log_r = math.log(max(right, 1.0e-30))
    denom = log_l - 2.0 * log_c + log_r
    if abs(denom) < 1.0e-20:
        return 0.0
    p = 0.5 * (log_l - log_r) / denom
    return max(-0.5, min(0.5, p))


def log_power_parabolic_two_log(left: float, center: float, right: float) -> float:
    left = max(left, 1.0e-30)
    center = max(center, 1.0e-30)
    right = max(right, 1.0e-30)
    log_lc = math.log(left / center)
    log_rc = math.log(right / center)
    denom = log_lc + log_rc
    if abs(denom) < 1.0e-20:
        return 0.0
    p = 0.5 * (log_lc - log_rc) / denom
    return max(-0.5, min(0.5, p))


def log_power_parabolic(left: float, center: float, right: float) -> float:
    return log_power_parabolic_two_log(left, center, right)


def test_two_log_ratio_matches_three_log_formula() -> None:
    values = [1.0e-9, 1.0e-4, 0.1, 0.75, 1.0, 2.0, 16.0, 1.0e5]
    for left in values:
        for center in values:
            for right in values:
                three = log_power_parabolic_three_log(left, center, right)
                two = log_power_parabolic_two_log(left, center, right)
                assert abs(two - three) <= 1.0e-12


def test_hann_log_power_parabolic_bias_contract() -> None:
    size = 1024
    center_bin = 37
    offsets = [(-49 + i) / 100.0 for i in range(99)]
    max_abs_error = 0.0

    for offset in offsets:
        tone_bin = center_bin + offset
        left = dft_bin_magsq(size, tone_bin, center_bin - 1)
        center = dft_bin_magsq(size, tone_bin, center_bin)
        right = dft_bin_magsq(size, tone_bin, center_bin + 1)
        estimated = log_power_parabolic(left, center, right)
        max_abs_error = max(max_abs_error, abs(estimated - offset))

    assert max_abs_error <= HANN_LOG_PARABOLIC_MAX_ERROR_BINS


def test_c_estimator_callback_and_malformed_triplet_contract() -> None:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc, "no C compiler available"

    harness = r'''
#include "spectral_peak_estimator.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>

static float quarter_offset(float left, float center, float right) {
    (void)left;
    (void)center;
    (void)right;
    return 0.25f;
}

static float nan_offset(float left, float center, float right) {
    (void)left;
    (void)center;
    (void)right;
    return NAN;
}

int main(void) {
    float magsq[3] = {1.0f, 4.0f, 1.0f};
    float bad_magsq[3] = {NAN, 4.0f, 1.0f};
    float phases[3] = {0.0f, 0.0f, 0.0f};
    SpectralPeakEstimateInput input = {0};
    SpectralPeakEstimate out = {0};

    input.magsq_row = magsq;
    input.phase_row = phases;
    input.n_freqs = 3u;
    input.bin = 1u;
    input.curr_magsq = 4.0f;
    input.next_max_magsq = 4.0f;
    input.best_next_bin = 1;
    input.freq_step_omega = 2.0f;
    input.freq_step_df = 0.5f;
    input.inv_hop = 0.25f;
    input.interp_magsq = quarter_offset;
    input.type = SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;

    if (!spectral_peak_estimate(&input, &out)) return 1;
    if (fabsf(out.bin_offset - 0.25f) > 1.0e-6f) return 2;
    if (fabsf(out.omega - 2.5f) > 1.0e-6f) return 3;

    input.interp_magsq = nan_offset;
    if (spectral_peak_estimate(&input, &out)) return 4;

    input.magsq_row = bad_magsq;
    input.interp_magsq = quarter_offset;
    if (spectral_peak_estimate(&input, &out)) return 5;

    input.magsq_row = magsq;
    input.best_next_bin = INT_MIN;
    if (spectral_peak_estimate(&input, &out)) return 6;

    input.best_next_bin = 3;
    if (spectral_peak_estimate(&input, &out)) return 7;

    return 0;
}
    '''

    with tempfile.TemporaryDirectory(prefix="spectral-peak-estimator-") as tmp:
        tmp_path = Path(tmp)
        harness_c = tmp_path / "peak_estimator_contract.c"
        exe = tmp_path / "peak_estimator_contract"
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


def test_raw_tracker_descriptor_changes_single_shot_omega() -> None:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc, "no C compiler available"

    # n_fft = 64 (>= SPECTRAL_MIN_FFT_SIZE); n_freqs = 33; two frames each carrying a
    # symmetric (1,4,1) magnitude-squared peak at interior bin 8. The symmetric triplet
    # gives a parabolic sub-bin offset of exactly 0, so the default omega is
    # 8*(2*pi/64); the custom descriptor's constant quarter_offset adds +0.25 bin, so its
    # omega is 8.25*(2*pi/64). The offset is a FREQUENCY refinement, so the center-magsq-
    # driven amplitude must be identical for default and custom; the bad descriptor (null
    # offset and peak fns) must be rejected. omega/amp are derived independently of the
    # implementation, so this is a behavioural contract, not a change-detector.
    harness = r'''
#include "spectral_peak_track.h"
#include "spectral_windows.h"
#include "spectral_consts.h"
#include <math.h>

#define N_FFT    64
#define N_FREQS  33            /* N_FFT/2 + 1 */
#define N_FRAMES 2u
#define PEAK_BIN 8             /* interior trackable bin (not DC or Nyquist) */

static float quarter_offset(float left, float center, float right) {
    (void)left; (void)center; (void)right;
    return 0.25f;              /* constant sub-bin offset, independent of the triplet */
}

int main(void) {
    float magsq[N_FREQS * 2] = {0};
    float phase[N_FREQS * 2] = {0};
    for (unsigned f = 0; f < N_FRAMES; f++) {
        float* row = magsq + (size_t)f * N_FREQS;
        row[PEAK_BIN - 1] = 1.0f;
        row[PEAK_BIN]     = 4.0f;   /* local max -> the tracked peak */
        row[PEAK_BIN + 1] = 1.0f;
    }
    const float bin_step = SPECTRAL_TWO_PI / (float)N_FFT;   /* omega per bin = 2*pi/n_fft */
    double t_default = 0.0, t_custom = 0.0, t_bad = 0.0;

    SpectralWindowDescriptor desc = {
        SPECTRAL_WINDOW_HANN, "test-quarter", "Test Quarter", 0, quarter_offset, 0
    };
    SpectralWindowDescriptor bad_desc = {
        SPECTRAL_WINDOW_HANN, "bad", "Bad", 0, 0, 0
    };

    SegmentArray default_segments = spectral_track_peaks(
        magsq, phase, 4.0f, N_FRAMES, (unsigned)N_FREQS, 48000, N_FFT, 1, -120.0f, &t_default);
    SegmentArray custom_segments = spectral_track_peaks_with_window_descriptor(
        magsq, phase, 4.0f, N_FRAMES, (unsigned)N_FREQS, 48000, N_FFT, 1, -120.0f,
        &desc, SPECTRAL_PEAK_ESTIMATOR_AUTO, &t_custom);
    SegmentArray bad_segments = spectral_track_peaks_with_window_descriptor(
        magsq, phase, 4.0f, N_FRAMES, (unsigned)N_FREQS, 48000, N_FFT, 1, -120.0f,
        &bad_desc, SPECTRAL_PEAK_ESTIMATOR_AUTO, &t_bad);

    if (default_segments.count != 1u || !default_segments.segs) return 1;
    if (custom_segments.count != 1u || !custom_segments.segs) return 2;

    /* symmetric (1,4,1) -> parabolic offset 0 -> omega = bin * bin_step */
    if (fabsf(default_segments.segs[0].omega - (float)PEAK_BIN * bin_step) > 1.0e-4f) return 3;
    /* quarter_offset shifts the sub-bin position by +0.25 bin */
    if (fabsf(custom_segments.segs[0].omega - ((float)PEAK_BIN + 0.25f) * bin_step) > 1.0e-4f) return 4;

    /* Frequency offset must not move the (center-magsq-driven) amplitude. */
    float amp_default = default_segments.segs[0].amp;
    float amp_custom  = custom_segments.segs[0].amp;
    if (!(amp_default > 0.0f) || !isfinite(amp_default)) return 5;
    if (fabsf(amp_custom - amp_default) > 1.0e-4f * amp_default) return 6;

    /* Null offset and peak fns -> descriptor rejected -> no segments emitted. */
    if (bad_segments.count != 0u || bad_segments.segs) return 7;

    spectral_segment_array_free(&default_segments);
    spectral_segment_array_free(&custom_segments);
    spectral_segment_array_free(&bad_segments);
    return 0;
}
'''

    with tempfile.TemporaryDirectory(prefix="spectral-raw-tracker-descriptor-") as tmp:
        tmp_path = Path(tmp)
        harness_c = tmp_path / "raw_tracker_descriptor.c"
        exe = tmp_path / "raw_tracker_descriptor"
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


def _compile_and_run_estimator_flags(extra_cflags: list[str]) -> dict[tuple[int, int], dict[str, float]]:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc, "no C compiler available"

    harness = r'''
#include "spectral_peak_estimator.h"
#include <math.h>
#include <stdio.h>

static void run_case(int case_id, SpectralPeakEstimatorType type,
                     float left, float center, float right,
                     float phase_l, float phase_c, float phase_r,
                     float next_max, int best_next_bin) {
    float magsq[3] = {left, center, right};
    float phases[3] = {phase_l, phase_c, phase_r};
    SpectralPeakEstimateInput input = {0};
    SpectralPeakEstimate out = {0};
    int ok = 0;

    input.magsq_row = magsq;
    input.phase_row = phases;
    input.n_freqs = 3u;
    input.bin = 1u;
    input.curr_magsq = magsq[1];
    input.next_max_magsq = next_max;
    input.best_next_bin = best_next_bin;
    input.freq_step_omega = 2.0f;
    input.freq_step_df = 0.5f;
    input.inv_hop = 0.25f;
    input.hop_float = 4.0f;
    input.candan_correction = spectral_peak_candan_correction_for_n_freqs(input.n_freqs);
    input.type = type;

    ok = spectral_peak_estimate(&input, &out);
    printf("%d %d %d %.9g %.9g %.9g %.9g %.9g %u\n",
           case_id, (int)type, ok, out.bin_offset, out.amp, out.da, out.omega, out.df, out.flags);
}

int main(void) {
    const float cases[][8] = {
        {2.25f, 9.0f, 4.0f, 0.2f, 0.45f, 0.7f, 6.25f, 2.0f},
        {1.0f, 6.5f, 2.0f, -0.4f, 0.1f, 0.55f, 4.0f, 1.0f},
        {3.5f, 12.0f, 2.75f, 0.9f, 1.1f, 1.35f, 9.0f, 0.0f},
        {0.25f, 3.0f, 0.5f, -1.0f, -0.25f, 0.25f, 2.25f, 2.0f},
        {8.0f, 21.0f, 7.0f, 1.5f, 1.75f, 2.0f, 16.0f, 1.0f},
        {0.0001f, 0.004f, 0.0002f, -2.0f, -1.0f, -0.1f, 0.0025f, 0.0f},
        {5.0f, 20.0f, 13.0f, 0.0f, 0.6f, 1.1f, 18.0f, 2.0f},
        {1.25f, 10.0f, 1.75f, -0.7f, -0.2f, 0.4f, 7.0f, 1.0f}
    };
    const SpectralPeakEstimatorType methods[] = {
        SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC,
        SPECTRAL_PEAK_ESTIMATOR_MAG_PARABOLIC,
        SPECTRAL_PEAK_ESTIMATOR_JACOBSEN_COMPLEX,
        SPECTRAL_PEAK_ESTIMATOR_CANDAN_COMPLEX,
        SPECTRAL_PEAK_ESTIMATOR_QUINN_SECOND
    };
    for (int i = 0; i < 8; i++) {
        for (int m = 0; m < 5; m++) {
            run_case(i, methods[m],
                     cases[i][0], cases[i][1], cases[i][2],
                     cases[i][3], cases[i][4], cases[i][5],
                     cases[i][6], (int)cases[i][7]);
        }
    }
    return 0;
}
'''

    with tempfile.TemporaryDirectory(prefix="spectral-peak-estimator-flags-") as tmp:
        tmp_path = Path(tmp)
        harness_c = tmp_path / "peak_estimator_flags.c"
        exe = tmp_path / "peak_estimator_flags"
        harness_c.write_text(harness, encoding="utf-8")
        link_flags = ["-framework", "Accelerate"] if sys.platform == "darwin" else []
        subprocess.run(
            [
                cc,
                "-std=c11",
                *extra_cflags,
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
        completed = subprocess.run([str(exe)], check=True, cwd=ROOT, text=True, capture_output=True)

    rows: dict[tuple[int, int], dict[str, float]] = {}
    for line in completed.stdout.strip().splitlines():
        case_id, method, ok, offset, amp, da, omega, df, flags = line.split()
        rows[(int(case_id), int(method))] = {
            "ok": float(ok),
            "offset": float(offset),
            "amp": float(amp),
            "da": float(da),
            "omega": float(omega),
            "df": float(df),
            "flags": float(flags),
        }
    return rows


def test_approximation_flags_stay_within_peak_error_contract() -> None:
    exact = _compile_and_run_estimator_flags([])
    approx = _compile_and_run_estimator_flags(
        [
            "-DSPECTRAL_ENABLE_APPROX_PEAK_LOG=1",
            "-DSPECTRAL_ENABLE_APPROX_TRIG=1",
            "-DSPECTRAL_ENABLE_APPROX_INV_SQRT=1",
        ]
    )

    assert exact.keys() == approx.keys()
    for method, exact_row in exact.items():
        approx_row = approx[method]
        assert exact_row["ok"] == 1.0
        assert approx_row["ok"] == 1.0
        assert math.isfinite(approx_row["offset"])
        assert math.isfinite(approx_row["amp"])
        assert math.isfinite(approx_row["da"])
        assert math.isfinite(approx_row["omega"])
        assert math.isfinite(approx_row["df"])
        assert abs(approx_row["offset"] - exact_row["offset"]) <= 0.002
        assert abs(approx_row["amp"] - exact_row["amp"]) / exact_row["amp"] <= 1.0e-3
        assert abs(approx_row["da"] - exact_row["da"]) <= 1.0e-3
        assert approx_row["df"] == exact_row["df"]
        assert int(approx_row["flags"]) & 16 == int(exact_row["flags"]) & 16


def test_peak_estimator_benchmark_harness_reports_all_methods() -> None:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc, "no C compiler available"

    with tempfile.TemporaryDirectory(prefix="spectral-peak-estimator-bench-") as tmp:
        exe = Path(tmp) / "peak_estimator_bench"
        link_flags = ["-framework", "Accelerate"] if sys.platform == "darwin" else []
        subprocess.run(
            [
                cc,
                "-O2",
                "-std=c11",
                "-DSPECTRAL_PEAK_BENCH_BLOCKS=2",
                "-DSPECTRAL_PEAK_BENCH_ITERS=64",
                "-I",
                str(ROOT / "spectral_engine/core"),
                "-I",
                str(ROOT / "spectral_engine/analysis"),
                "-I",
                str(ROOT / "spectral_engine/runtime"),
                str(ROOT / "spectral_engine/analysis/spectral_peak_estimator.c"),
                str(ROOT / "spectral_engine/core/spectral_fast_math.c"),
                str(ROOT / "spectral_engine/core/spectral_windows.c"),
                str(ROOT / "tools/core_audit/peak_estimator_bench.c"),
                "-lm",
                *link_flags,
                "-o",
                str(exe),
            ],
            check=True,
            cwd=ROOT,
        )
        completed = subprocess.run([str(exe)], check=True, cwd=ROOT, text=True, capture_output=True)

    output = completed.stdout
    assert "p50_ns=" in output
    assert "p95_ns=" in output
    assert "max_offset_err=" in output
    assert "fallback_rate=" in output
    for name in [
        "log-parabolic",
        "mag-parabolic",
        "jacobsen-complex",
        "candan-complex",
        "quinn-second",
    ]:
        assert name in output
