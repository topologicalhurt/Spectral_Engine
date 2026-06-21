"""Determinism must be invariant under user-exposed parameters.

The maintainer's requirement: the determinism guarantee (and therefore the WCET
that gates it) must hold across all parameters a user can vary — timbre,
frequency, pitch, stretch, amplitude — not just one fixture. This test PROVES
that the embedded per-voice cost is param-invariant, empirically:

  - The embedded synth path is SINE-ADDITIVE (the coupled oscillator; bandlimited
    /wavetable timbres are desktop-only). A "timbre" is a set of sine partials, so
    timbre/pitch/stretch all reduce to "N sine voices with some phase_inc/amp."
  - The coupled-step recurrence cost is independent of phase_inc value, amplitude,
    and saturation (same instructions every iteration), so the HOT LOOP is exactly
    invariant.
  - The ONE residual variation is the per-voice activation seed's trig range
    reduction (the seed angle = phase_inc·rad scales with frequency; a higher
    angle can take one extra reduction step). It is bounded by the angle range
    (<= 2*pi -> at most ~1 reduction), so it is at most a few instructions PER
    VOICE, one-time at activation — negligible vs the WCET safety margin, but
    real, so we pin the bound here rather than claim exact invariance.

If this bound is ever exceeded, a frequency- or amplitude-dependent code path has
entered the per-sample cost and the determinism guarantee is no longer
param-invariant — which must fail CI, not slip through.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from spectral_tools.performance.embedded import toolchain  # noqa: E402

# Activation-seed trig range reduction is bounded by one period; budget a few
# instructions per voice of headroom. Exceeding this means a param-dependent
# path leaked into the per-voice cost.
PARAM_INVARIANCE_INSN_BOUND_PER_VOICE = 4

VOICES = 64


def _have(*need: str) -> bool:
    try:
        toolchain.discover(ROOT, need=frozenset(need))
    except toolchain.ToolchainError:
        return False
    return True


def _process_insns(tc, spec, out_dir, tag) -> int:
    import re

    from spectral_tools.core.process import run
    from spectral_tools.performance.embedded import counts

    d = out_dir / tag
    d.mkdir(parents=True, exist_ok=True)
    plugin = counts._build_plugin(tc, out_dir)
    elf = counts._build_runner_elf(tc, spec, d)
    nm = run([tc.arm_nm, "-S", str(elf)], cwd=tc.repo_root, check=False)
    ranges = [
        (m.group(3), int(m.group(1), 16), int(m.group(1), 16) + int(m.group(2), 16))
        for m in re.finditer(
            r"^([0-9a-f]+)\s+([0-9a-f]+)\s+[Tt]\s+(\S+)$", nm.stdout, re.M)
    ]
    table, _cs, _ = counts._run_once(tc, elf, plugin, sorted(ranges), d / "c.txt")
    rows, _, _ = counts._parse_counts(table)
    process = next(r for r in rows if r.name == "spectral_arm32_process")
    return process.insns


@pytest.mark.skipif(not _have("qemu"),
                    reason="needs newlib arm-none-eabi-gcc + qemu")
def test_determinism_is_parameter_invariant(tmp_path):
    from spectral_tools.performance.embedded import fixture

    tc = toolchain.discover(ROOT, need=frozenset({"qemu"}))
    base = fixture.sustained_fixture(VOICES)

    # Same voice/sample structure, swept across the user-exposed parameter space.
    cases = {
        "base": base,
        "freq_low": replace(base, name="inv_low",
                            voices=tuple(replace(v, freq_hz=30.0) for v in base.voices)),
        "freq_high": replace(base, name="inv_high",
                             voices=tuple(replace(v, freq_hz=18000.0) for v in base.voices)),
        "freq_mixed": replace(base, name="inv_mix",
                              voices=tuple(replace(v, freq_hz=30.0 + i * 281.0)
                                           for i, v in enumerate(base.voices))),
        "amp_tiny": replace(base, name="inv_amp",
                            voices=tuple(replace(v, amp=0.001) for v in base.voices)),
    }
    insns = {tag: _process_insns(tc, spec, tmp_path, tag) for tag, spec in cases.items()}

    bound = PARAM_INVARIANCE_INSN_BOUND_PER_VOICE * VOICES
    lo, hi = min(insns.values()), max(insns.values())
    spread = hi - lo
    assert spread <= bound, (
        f"per-voice cost varies {spread} insns across user parameters "
        f"(> {bound} = {PARAM_INVARIANCE_INSN_BOUND_PER_VOICE}/voice) — a "
        f"frequency/amplitude-dependent path leaked into the per-voice cost; "
        f"determinism is no longer param-invariant: {insns}")

    # Amplitude must be EXACTLY invariant (no amp-dependent branch anywhere).
    assert insns["amp_tiny"] == insns["base"], (
        f"amplitude changed the instruction count ({insns['amp_tiny']} vs "
        f"{insns['base']}) — an amplitude-dependent branch exists")
