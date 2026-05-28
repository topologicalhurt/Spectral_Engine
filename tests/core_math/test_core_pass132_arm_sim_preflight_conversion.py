#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
sim_c = (ROOT / "spectral_engine/synth/backends/sim/spectral_synth_simulation.c").read_text()

assert '#include "spectral_contracts.h"' in sim_c
assert 'SynthPreflight pf = synth_preflight_float' in sim_c
assert 'if (!pf.ok)' in sim_c
assert 'SYNTH_VALIDATE_FLOAT' not in sim_c
assert 'const SynthParams params = pf.params;' in sim_c
assert 'out_len > (size_t)UINT32_MAX' in sim_c
assert 'static int segment_to_sim' in sim_c
assert 'spectral_segment_valid_for_synth(src)' in sim_c
assert 'start_d > (double)UINT32_MAX' in sim_c
assert 'dst->start = (uint32_t)start_d;' in sim_c
assert 'sim_segs[i].start = (uint32_t)(sim_segs[i].start * stretch)' not in sim_c
print('pass132 ok')
