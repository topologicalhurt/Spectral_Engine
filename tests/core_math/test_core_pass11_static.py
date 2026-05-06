#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

est_h = (ROOT / "spectral_engine/analysis/spectral_peak_estimator.h").read_text()
est_c = (ROOT / "spectral_engine/analysis/spectral_peak_estimator.c").read_text()
config_h = (ROOT / "spectral_engine/core/spectral_config.h").read_text()
windows_c = (ROOT / "spectral_engine/core/spectral_windows.c").read_text()
fast_math_c = (ROOT / "spectral_engine/core/spectral_fast_math.c").read_text()
fast_math_h = (ROOT / "spectral_engine/core/spectral_fast_math.h").read_text()
manifest = (ROOT / "spectral_engine/cmake/source-manifest.cmake").read_text()
track_h = (ROOT / "spectral_engine/analysis/spectral_peak_track.h").read_text()
track_internal_h = (ROOT / "spectral_engine/analysis/spectral_peak_track_internal.h").read_text()
track_c = (ROOT / "spectral_engine/analysis/spectral_peak_track.c").read_text()
interp_c = (ROOT / "spectral_engine/analysis/spectral_peak_interp.c").read_text()
interp_h = (ROOT / "spectral_engine/analysis/spectral_peak_interp.h").read_text()
pass11 = (ROOT / "docs/core_audit/PATCH_NOTES_PASS11.md").read_text()
contract_test = (ROOT / "tests/core_math/test_core_pass11_peak_estimator_contract.py").read_text()
bench_c = (ROOT / "tools/core_audit/peak_estimator_bench.c").read_text()

assert "spectral_peak_estimator.c" in manifest
assert "SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC" in est_h
assert "SPECTRAL_PEAK_ESTIMATOR_JACOBSEN_COMPLEX" in est_h
assert "SPECTRAL_PEAK_ESTIMATOR_CANDAN_COMPLEX" in est_h
assert "SPECTRAL_PEAK_ESTIMATOR_QUINN_SECOND" in est_h
assert "SpectralPeakEstimateInput" in est_h
assert "SpectralPeakEstimate" in est_h
assert "spectral_peak_estimate_validated" in est_h
assert "spectral_peak_candan_correction_for_n_freqs" in est_h
assert "float candan_correction;" in est_h
assert "SPECTRAL_ENABLE_APPROX_PEAK_LOG" in config_h

assert "https://doi.org/10.1109/PROC.1978.10837" in est_c
assert "https://doi.org/10.1109/78.295186" in est_c
assert "https://doi.org/10.1109/MSP.2007.361611" in est_c
assert "https://doi.org/10.1109/LSP.2011.2136378" in est_c
assert "spectral_peak_complex_offset_jacobsen" in est_c
assert "spectral_peak_offset_candan" in est_c
assert "spectral_peak_offset_quinn_second" in est_c
assert "spectral_peak_reconstruct_triplet" in est_c
assert "spectral_peak_sincosf" in est_c
assert "SPECTRAL_ENABLE_APPROX_TRIG" in est_c
assert "fast_sqrt(" in est_c
assert "fast_peak_log(" in est_c
assert "0x5f3759df" not in est_c
assert "#include <limits.h>" in est_c
assert "spectral_peak_best_next_valid" in est_c
assert "input->best_next_bin < 0" in est_c
assert "best_next + 1u < input->bin" in est_c
assert "spectral_peak_estimator_resolve_default" in est_c
assert "return SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;" in est_c
assert "raw_offset = interp(left, center, right);" in est_c
assert "if (!isfinite(raw_offset)) return 0;" in est_c
assert "SPECTRAL_PEAK_ESTIMATE_USED_FALLBACK" in est_c
assert "spectral_peak_estimate_validated" in est_c

assert "log_lc = fast_peak_log(left / center)" in windows_c
assert "log_rc = fast_peak_log(right / center)" in windows_c
assert "spectral_window_peak_logf" not in windows_c
assert "FAST_MATH_HOT float fast_peak_log(float x);" in fast_math_h
assert "float fast_peak_log(float x)" in fast_math_c
assert "SPECTRAL_ENABLE_APPROX_PEAK_LOG" in fast_math_c
assert "https://standards.ieee.org/standard/754-2019.html" in fast_math_c
assert "https://dlmf.nist.gov/4.37.E25" in fast_math_c
assert "https://dlmf.nist.gov/4.37.E31" in fast_math_c
assert "0x5f3759df" in fast_math_c

assert '#include "spectral_peak_estimator.h"' in track_h
assert "spectral_track_peaks_with_window_descriptor" in track_h
assert "spectral_tracker_set_peak_estimator" in track_h
assert "SpectralPeakEstimatorType peak_estimator;" in track_internal_h
assert "float peak_candan_correction;" in track_internal_h
assert "tracker->peak_estimator = SPECTRAL_PEAK_ESTIMATOR_DEFAULT;" in track_c
assert "tracker->peak_candan_correction = spectral_peak_candan_correction_for_n_freqs(n_freqs);" in track_c
assert "peak_model.estimator = estimator;" in track_c
assert "spectral_tracker_set_peak_model(tracker, &peak_model)" in track_c
assert "spectral_track_peaks_with_window_descriptor(" in track_c

assert '#include "spectral_peak_estimator.h"' in interp_c
assert "SpectralPeakEstimateInput estimate_input" in interp_c
assert "spectral_peak_estimate_validated(&estimate_input, &estimate)" in interp_c
assert "estimate_input.candan_correction = tracker->peak_candan_correction;" in interp_c
assert "seg->amp = estimate.amp;" in interp_c
assert "seg->omega = estimate.omega;" in interp_c
assert "spectral_window_interp_magsq_parabolic(left, curr, right)" not in interp_c
assert "estimator module" in interp_h.lower()

assert "0.02` bins" in pass11
assert "https://doi.org/10.1109/PROC.1978.10837" in pass11
assert "https://www.dsprelated.com/freebooks/sasp/quadratic_interpolation_spectral_peaks.html" in pass11
assert "two-log" in pass11
assert "endpoint_amp_scale_d = 1.0 / (double)metrics.sum" in windows_c
assert "endpoint_magsq_scale_d = endpoint_amp_scale_d * endpoint_amp_scale_d" in windows_c
assert "positive_amp_scale_d = 2.0 / (double)metrics.sum" in windows_c
assert "positive_magsq_scale_d = positive_amp_scale_d * positive_amp_scale_d" in windows_c
assert "test_approximation_flags_stay_within_peak_error_contract" in contract_test
assert "test_peak_estimator_benchmark_harness_reports_all_methods" in contract_test
assert "test_raw_tracker_descriptor_changes_single_shot_omega" in contract_test
assert "INT_MIN" in contract_test
assert "spectral_track_peaks_with_window_descriptor" in contract_test
assert "BenchExpected" in bench_c
assert "Correctness accounting is intentionally outside the timed loop" in bench_c
assert "spectral_peak_estimate_validated(&inputs[case_idx], &out)" in bench_c
assert "p50_ns=" in bench_c
assert "max_offset_err=" in bench_c
assert "fallback_rate=" in bench_c

print("pass11 static checks passed")
