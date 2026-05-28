#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[2]
src = (root / "spectral_engine/synth/backends/arm/spectral_synth_arm32.c").read_text()
assert "spectral_arm32_segment_chirp_supported" in src
assert "seg->df_q15 == 0" in src
assert "!spectral_arm32_segment_chirp_supported(&data[i])" in src
