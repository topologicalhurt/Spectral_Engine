#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

track_h = (ROOT / "spectral_engine/analysis/spectral_peak_track.h").read_text()
track_internal_h = (ROOT / "spectral_engine/analysis/spectral_peak_track_internal.h").read_text()
track_c = (ROOT / "spectral_engine/analysis/spectral_peak_track.c").read_text()
interp_c = (ROOT / "spectral_engine/analysis/spectral_peak_interp.c").read_text()
interp_h = (ROOT / "spectral_engine/analysis/spectral_peak_interp.h").read_text()
windows_c = (ROOT / "spectral_engine/core/spectral_windows.c").read_text()
fast_math_c = (ROOT / "spectral_engine/core/spectral_fast_math.c").read_text()
full_c = (ROOT / "spectral_engine/analysis/spectral_analysis_full.c").read_text()
fused_c = (ROOT / "spectral_engine/analysis/spectral_analysis_fused.c").read_text()
pass10 = (ROOT / "docs/core_audit/PATCH_NOTES_PASS10.md").read_text()
audit = (ROOT / "tools/core_audit/core_static_audit.py").read_text()

assert "spectral_tracker_set_window_descriptor" in track_h
assert "spectral_track_peaks_with_window_descriptor" in track_h
assert "SpectralWindowInterpMagsqFn interp_magsq;" in track_internal_h
assert "tracker->interp_magsq = spectral_window_interp_magsq_parabolic;" in track_c
assert "spectral_peak_model_for_window(" in track_c
assert "spectral_tracker_set_peak_model(tracker, &peak_model)" in track_c
assert "spectral_tracker_set_window_descriptor(tracker, spectral_window_descriptor(SPECTRAL_WINDOW_HANN));" in fused_c
assert "spectral_track_peaks_with_window_descriptor" in full_c

assert "left = row[cf - 1u];" in interp_c
assert "right = row[cf + 1u];" in interp_c
assert "!isfinite(left) || !isfinite(curr) || !isfinite(right)" in interp_c
assert "!isfinite(threshsq) || threshsq < 0.0f" in interp_c
assert "SpectralPeakEstimateInput estimate_input" in interp_c
assert "spectral_peak_estimate_validated(&estimate_input, &estimate)" in interp_c
assert "spectral_window_interp_magsq_parabolic(left, curr, right)" not in interp_c
assert "estimator module" in interp_h.lower()
assert "log_lc = fast_peak_log(left / center)" in windows_c
assert "log_rc = fast_peak_log(right / center)" in windows_c
assert "float fast_peak_log(float x)" in fast_math_c

assert "0.5 * (log(left) - log(right))" in pass10
assert "https://doi.org/10.1109/PROC.1978.10837" in pass10
assert "https://www.dsprelated.com/freebooks/sasp/quadratic_interpolation_spectral_peaks.html" in pass10
assert "https://doi.org/10.1109/78.295186" in pass10
assert "https://doi.org/10.1109/MSP.2007.361611" in pass10
assert "https://doi.org/10.1109/LSP.2011.2136378" in pass10

assert "floor(start / tile_size_d)" in audit
assert "ceil(end / tile_size_d)" in audit

print("pass10 static checks passed")
