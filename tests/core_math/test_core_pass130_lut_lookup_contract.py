#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
lut_h = (ROOT / "spectral_engine/core/spectral_lut.h").read_text()

assert '#if SPECTRAL_OSC_LUT_BITS < 1 || SPECTRAL_OSC_LUT_BITS > 16' in lut_h
assert '#error "SPECTRAL_OSC_LUT_BITS must be in [1,16] for uq16_t phase lookup"' in lut_h
assert 'if (!lut) return Q15_ZERO;' in lut_h
assert '((uint32_t)phase_u16) >> (16u - SPECTRAL_OSC_LUT_BITS)' in lut_h
assert 'idx + 1u' in lut_h
assert '(uq16_t)(phase_u16 + (uq16_t)16384u)' in lut_h
print('pass130 ok')
