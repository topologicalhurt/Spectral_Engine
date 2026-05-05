#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

windows_c = (ROOT / "spectral_engine/core/spectral_windows.c").read_text()
notes8 = (ROOT / "docs/core_audit/PATCH_NOTES_PASS8.md").read_text()
contract8 = (ROOT / "docs/core_audit/PASS8_STFT_WINDOW_CONTRACT.md").read_text()

assert "positive_amp_scale_d = 2.0 / (double)metrics.sum" in windows_c
assert "positive_magsq_scale_d = positive_amp_scale_d * positive_amp_scale_d" in windows_c
assert "isfinite(positive_amp_scale)" in windows_c
assert "isfinite(positive_magsq_scale)" in windows_c
assert "metrics.flags |= SPECTRAL_WINDOW_METRIC_POSITIVE_BIN_SCALE_VALID" in windows_c

assert "endpoint_amp_scale_d = 1.0 / (double)metrics.sum" in windows_c
assert "endpoint_magsq_scale_d = endpoint_amp_scale_d * endpoint_amp_scale_d" in windows_c
assert "isfinite(endpoint_amp_scale)" in windows_c
assert "isfinite(endpoint_magsq_scale)" in windows_c
assert "metrics.flags |= SPECTRAL_WINDOW_METRIC_ENDPOINT_BIN_SCALE_VALID" in windows_c

# The old stale uniform-scale documentation must be gone.
assert "FFT resources now carry an explicit `magsq_scale`" not in notes8
assert "applies the scale to every magnitude-squared row" not in notes8
assert "returned frame maximum" not in notes8 or "recomputed from scaled interior trackable bins" in notes8

assert "endpoint/interior-bin" in notes8
assert "trackable interior bins" in notes8
assert "DC/Nyquist" in contract8
assert "(1 / sum(window))^2" in contract8
assert "(2 / sum(window))^2" in contract8

print("pass9 static checks passed")
