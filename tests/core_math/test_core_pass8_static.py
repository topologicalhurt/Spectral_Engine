#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

windows_h = (ROOT / "spectral_engine/core/spectral_windows.h").read_text()
windows_c = (ROOT / "spectral_engine/core/spectral_windows.c").read_text()
internal_h = (ROOT / "spectral_engine/analysis/spectral_analysis_internal.h").read_text()
fft_c = (ROOT / "spectral_engine/analysis/spectral_analysis_fft.c").read_text()
full_c = (ROOT / "spectral_engine/analysis/spectral_analysis_full.c").read_text()
fused_c = (ROOT / "spectral_engine/analysis/spectral_analysis_fused.c").read_text()

assert "spectral_window_sum" in windows_h
assert "spectral_window_energy" in windows_h
assert "typedef void (*SpectralWindowGenerateFn)" in windows_h
assert "typedef float (*SpectralWindowInterpMagsqFn)" in windows_h
assert "SpectralWindowDescriptor" in windows_h
assert "SpectralWindowMetrics" in windows_h
assert "positive_bin_amp_scale" in windows_h
assert "positive_bin_magsq_scale" in windows_h
assert "endpoint_bin_amp_scale" in windows_h
assert "endpoint_bin_magsq_scale" in windows_h
assert "unsigned flags;" in windows_h
assert "SPECTRAL_WINDOW_METRIC_POSITIVE_BIN_SCALE_VALID" in windows_h
assert "SPECTRAL_WINDOW_METRIC_ENDPOINT_BIN_SCALE_VALID" in windows_h
assert "SPECTRAL_WINDOW_METRIC_ENBW_VALID" in windows_h
assert "spectral_window_descriptor(SpectralWindowType type)" in windows_h
assert "spectral_window_descriptor_at(size_t index)" in windows_h
assert "spectral_window_descriptor_count(void)" in windows_h
assert "spectral_window_find_by_id" in windows_h
assert "spectral_window_metrics" in windows_h
assert "spectral_window_positive_bin_magsq_scale" in windows_h
assert "spectral_window_endpoint_bin_magsq_scale" in windows_h
assert "static const SpectralWindowDescriptor spectral_window_descriptors[]" in windows_c
for window_id in ['"hann"', '"hamming"', '"blackman"', '"rectangular"']:
    assert window_id in windows_c
assert "desc->generate(window, length)" in windows_c
assert "metrics.flags |= SPECTRAL_WINDOW_METRIC_POSITIVE_BIN_SCALE_VALID" in windows_c
assert "SPECTRAL_WINDOW_METRIC_ENDPOINT_BIN_SCALE_VALID" in windows_c
assert "metrics.flags |= SPECTRAL_WINDOW_METRIC_ENBW_VALID" in windows_c
assert "vDSP_HANN_NORM" not in windows_c
assert "vDSP_HANN_DENORM" in windows_c
assert "float endpoint_bin_magsq_scale;" in internal_h
assert "float positive_bin_magsq_scale;" in internal_h
assert "spectral_fft_resources_set_magsq_scales" in internal_h
assert "res->endpoint_bin_magsq_scale = 1.0f;" in fft_c
assert "res->positive_bin_magsq_scale = 1.0f;" in fft_c
assert "spectral_fft_apply_magsq_scales" in fft_c
assert "spectral_fft_trackable_magsq_max" in fft_c
assert "magsq[0] *= endpoint_scale" in fft_c
assert "magsq[n_freqs - 1u] *= endpoint_scale" in fft_c
assert "magsq[i] *= positive_scale" in fft_c
assert "*frame_max = spectral_fft_trackable_magsq_max(magsq, n_freqs)" in fft_c
assert "*frame_max *= scale" not in fft_c
assert "for (size_t i = 0; i < n_freqs; i++)" not in fft_c
assert "void spectral_fft_resources_set_magsq_scales" in fft_c[:fft_c.find("int spectral_fft_resources_alloc")]
assert "static void spectral_fft_apply_magsq_scales" in fft_c[:fft_c.find("int spectral_fft_resources_alloc")]
assert "spectral_window_hann(window_func" not in full_c
assert "spectral_window_hann(window_func" not in fused_c
assert "spectral_window_generate(window_func, (size_t)n_fft, SPECTRAL_WINDOW_HANN)" in full_c
assert "spectral_window_generate(window_func, (size_t)n_fft, SPECTRAL_WINDOW_HANN)" in fused_c
assert "window_metrics = spectral_window_metrics(window_func, (size_t)n_fft)" in full_c
assert "window_metrics = spectral_window_metrics(window_func, (size_t)n_fft)" in fused_c
assert "window_metrics.endpoint_bin_magsq_scale" in full_c
assert "window_metrics.positive_bin_magsq_scale" in full_c
assert "window_metrics.endpoint_bin_magsq_scale" in fused_c
assert "window_metrics.positive_bin_magsq_scale" in fused_c

print("pass8 static checks passed")
