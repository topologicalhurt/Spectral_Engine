#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[2]
src = (root / "spectral_engine/synth/backends/arm/spectral_synth_arm32.c").read_text()
assert "spectral_arm32_segment_end_checked_u32" in src
assert "spectral_arm32_validate_segment_data" in src
assert "start < last_start || end < last_end" in src
assert "i - first_live + 1u" in src
assert "SPECTRAL_ARM32_MAX_ACTIVE" in src
assert "err = spectral_arm32_validate_segment_data" in src
