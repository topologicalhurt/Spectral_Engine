#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[2]
src = (root / "spectral_engine/synth/backends/sim/spectral_synth_simulation.c").read_text()
assert "uq32_t   phase_acc;" in src
assert "spectral_sim_phase_acc_from_q15" in src
assert "spectral_sim_phase_add_inc" in src
assert "spectral_sim_phase_add_scaled" in src
assert "spectral_sim_q31_add_scaled_sat" in src
assert "((q31_t)seg->phase_q15 + 32768) << 16" not in src
assert "phase += freq_inc" not in src
assert "p3 + freq_inc" not in src
