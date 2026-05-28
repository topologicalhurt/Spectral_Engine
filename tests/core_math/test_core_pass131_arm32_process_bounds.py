#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
arm_c = (ROOT / "spectral_engine/synth/backends/arm/spectral_synth_arm32.c").read_text()

assert 'static inline uint32_t spectral_arm32_segment_end_sat_u32' in arm_c
assert 'if (length > UINT32_MAX - start) return UINT32_MAX;' in arm_c
assert 'static inline void spectral_arm32_zero_output' in arm_c
assert 'num_samples > 256u' in arm_c
assert 'remaining = ctx->output_length - ctx->output_position;' in arm_c
assert 'if (num_samples > remaining) num_samples = remaining;' in arm_c
assert 'ctx->num_segments == 0u || !ctx->segments || !ctx->osc_lut' in arm_c
assert 'num_segments > 0u && (!data || !ctx->segments)' in arm_c
assert 'seg->start + seg->length' not in arm_c
assert 'seg_start + seg_length' not in arm_c
print('pass131 ok')
