#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

fft = (ROOT / "spectral_engine/analysis/spectral_analysis_fft.c").read_text()
full = (ROOT / "spectral_engine/analysis/spectral_analysis_full.c").read_text()

assert "#include <limits.h>" in fft
assert "if (!res) return 0;" in fft
assert "memset(res, 0, sizeof(*res));\n\n    if (n_threads < 1" in fft
assert "n_fft > (size_t)INT_MAX" in fft
assert "SpectralFftResources res = {0};" in full

print("pass7 static checks passed")
