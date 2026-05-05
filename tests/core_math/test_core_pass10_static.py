#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

track_h = (ROOT / "spectral_engine/analysis/spectral_peak_track.h").read_text()
track_internal_h = (ROOT / "spectral_engine/analysis/spectral_peak_track_internal.h").read_text()
track_c = (ROOT / "spectral_engine/analysis/spectral_peak_track.c").read_text()
interp_c = (ROOT / "spectral_engine/analysis/spectral_peak_interp.c").read_text()
interp_h = (ROOT / "spectral_engine/analysis/spectral_peak_interp.h").read_text()
fused_c = (ROOT / "spectral_engine/analysis/spectral_analysis_fused.c").read_text()
audit = (ROOT / "tools/core_audit/core_static_audit.py").read_text()

assert "spectral_tracker_set_window_descriptor" in track_h
assert "SpectralWindowInterpMagsqFn interp_magsq;" in track_internal_h
assert "tracker->interp_magsq = spectral_window_interp_magsq_parabolic;" in track_c
assert "spectral_tracker_set_window_descriptor(tracker, spectral_window_descriptor(SPECTRAL_WINDOW_HANN));" in track_c
assert "spectral_tracker_set_window_descriptor(tracker, spectral_window_descriptor(SPECTRAL_WINDOW_HANN));" in fused_c

assert "SpectralWindowInterpMagsqFn interp_magsq = tracker->interp_magsq" in interp_c
assert "p = interp_magsq(left, curr, right);" in interp_c
assert "spectral_window_interp_magsq_parabolic(left, curr, right)" not in interp_c
assert "if (!isfinite(p)) p = 0.0f;" in interp_c
assert "if (p > 0.5f) p = 0.5f;" in interp_c
assert "!isfinite(curr) || !isfinite(m0)" in interp_c
assert "window-aware" in interp_h.lower()

assert "floor(start / tile_size_d)" in audit
assert "ceil(end / tile_size_d)" in audit

print("pass10 static checks passed")
