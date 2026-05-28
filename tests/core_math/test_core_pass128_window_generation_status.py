#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
windows_h = (ROOT / "spectral_engine/core/spectral_windows.h").read_text()
windows_c = (ROOT / "spectral_engine/core/spectral_windows.c").read_text()
analysis_c = (ROOT / "spectral_engine/analysis/spectral_analysis.c").read_text()

assert '#include "spectral_error.h"' in windows_h
assert '#include "spectral_contracts.h"' in windows_c
assert 'SpectralError spectral_window_generate(float* window, size_t length, SpectralWindowType type);' in windows_h
assert 'SpectralError spectral_window_generate(float* window, size_t length, SpectralWindowType type)' in windows_c
assert 'void spectral_window_generate' not in windows_h
assert 'void spectral_window_generate' not in windows_c
assert 'return SPECTRAL_ERR_PARAM;' in windows_c
assert 'return SPECTRAL_OK;' in windows_c
assert 'spectral_f32_span_finite(window, length)' in windows_c
assert 'spectral_window_generate(ctx->samples, n_fft, type) != SPECTRAL_OK' in analysis_c
print('pass128 ok')
