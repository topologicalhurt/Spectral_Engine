#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
q15_h = (ROOT / "spectral_engine/synth/math/spectral_q15.h").read_text()
q15_c = (ROOT / "spectral_engine/synth/math/spectral_q15.c").read_text()

assert 'static inline q15_t spectral_float_to_q15(float f)' in q15_h
assert 'static inline q31_t spectral_float_to_q31(float f)' in q15_h
assert 'if (!isfinite(f)) return Q15_ZERO;' in q15_h
assert 'if (!isfinite(f)) return (q31_t)0;' in q15_h
assert '#define FLOAT_TO_Q15(f) spectral_float_to_q15((float)(f))' in q15_h
assert '#define FLOAT_TO_Q31(f) spectral_float_to_q31((float)(f))' in q15_h
assert 'if (!isfinite(rad)) return Q15_ZERO;' in q15_h
assert 'if (!isfinite(n)) return Q15_ZERO;' in q15_h
assert 'if (!isfinite(o) || o <= 0.0f) return 0u;' in q15_h
assert q15_c.count('if (count > 0u && (!src || !dst)) return;') >= 4
print('pass129 ok')
