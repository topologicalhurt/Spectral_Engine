#!/usr/bin/env python3
"""Window-convention decision data (REVIEWER_HANDOFF_2 sec 4.4).

Compiles and runs harnesses/window_convention_sweep.c — symmetric (N-1, the
engine SSOT) vs periodic (DFT-even, proposed) — and asserts the *stable facts*
the maintainer's decision rests on, so the data cannot silently rot:

  - periodic hann/hamming satisfy exact COLA (float-noise ripple) at dividing
    hops; the symmetric forms miss by O(1/N)
  - blackman fails COLA at hop N/2 in BOTH conventions (its k=2 cosine term
    needs >= 4 overlapping phases) — the OLA constraint is hop <= N/4 either way
  - the coherent-gain shift of adopting periodic is ~ +1/N (the golden delta)
  - log-parabolic estimation accuracy is convention-INSENSITIVE (the periodic
    form does not measurably improve the engine's peak estimates)

The harness reports; it decides nothing. The convention choice is the
maintainer's (deferred deliberately, mandate D2 golden re-sign-off).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COLA_RE = re.compile(
    r"^COLA window=(?P<window>\S+) form=(?P<form>\S+) n=(?P<n>\d+) hop=(?P<hop>\d+) "
    r"gain=(?P<gain>[0-9.eE+-]+|nan) ripple=(?P<ripple>[0-9.eE+-]+|nan)$"
)
GAIN_RE = re.compile(
    r"^GAINRATIO window=(?P<window>\S+) n=(?P<n>\d+) sum_sym=(?P<sum_sym>[0-9.eE+-]+) "
    r"sum_per=(?P<sum_per>[0-9.eE+-]+) ratio=(?P<ratio>[0-9.eE+-]+)$"
)
ESTBIAS_RE = re.compile(
    r"^ESTBIAS window=(?P<window>\S+) form=(?P<form>\S+) n=(?P<n>\d+) "
    r"estimator=log-parabolic cases=(?P<cases>\d+) valid=(?P<valid>\d+) "
    r"rms_err=(?P<rms>[0-9.eE+-]+) max_abs_err=(?P<max>[0-9.eE+-]+)$"
)
SPECSHIFT_RE = re.compile(
    r"^SPECSHIFT window=(?P<window>\S+) n=(?P<n>\d+) cases=(?P<cases>\d+) "
    r"max_rel_magsq_delta=(?P<delta>[0-9.eE+-]+)$"
)


def _run_harness() -> str:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc, "no C compiler available"
    with tempfile.TemporaryDirectory(prefix="spectral-winconv-") as tmp:
        exe = Path(tmp) / "window_convention_sweep"
        link_flags = ["-framework", "Accelerate"] if sys.platform == "darwin" else []
        subprocess.run(
            [
                cc, "-O2", "-std=c11",
                "-I", str(ROOT / "spectral_engine/core"),
                "-I", str(ROOT / "spectral_engine/analysis"),
                "-I", str(ROOT / "spectral_engine/runtime"),
                str(ROOT / "spectral_engine/analysis/spectral_peak_estimator.c"),
                str(ROOT / "spectral_engine/core/spectral_fast_math.c"),
                str(ROOT / "spectral_engine/core/spectral_windows.c"),
                str(ROOT / "tests/core_math/harnesses/window_convention_sweep.c"),
                "-lm", *link_flags,
                "-o", str(exe),
            ],
            check=True, cwd=ROOT,
        )
        completed = subprocess.run(
            [str(exe)], check=True, text=True, capture_output=True, cwd=ROOT
        )
    return completed.stdout


def test_window_convention_decision_data() -> None:
    out = _run_harness()

    cola: dict[tuple[str, str, int, int], dict[str, float]] = {}
    gains: dict[tuple[str, int], float] = {}
    estbias: dict[tuple[str, str], dict[str, float]] = {}
    specshift: dict[str, float] = {}
    for line in out.splitlines():
        m = COLA_RE.match(line.strip())
        if m:
            assert m.group("gain") != "nan", f"COLA stats failed: {line}"
            cola[(m.group("window"), m.group("form"), int(m.group("n")), int(m.group("hop")))] = {
                "gain": float(m.group("gain")), "ripple": float(m.group("ripple")),
            }
            continue
        m = GAIN_RE.match(line.strip())
        if m:
            gains[(m.group("window"), int(m.group("n")))] = float(m.group("ratio"))
            continue
        m = ESTBIAS_RE.match(line.strip())
        if m:
            assert int(m.group("valid")) == int(m.group("cases")), f"estimator rejected cases: {line}"
            estbias[(m.group("window"), m.group("form"))] = {
                "rms": float(m.group("rms")), "max": float(m.group("max")),
            }
            continue
        m = SPECSHIFT_RE.match(line.strip())
        if m:
            specshift[m.group("window")] = float(m.group("delta"))

    windows = ("hann", "hamming", "blackman")
    sizes = (1024, 4096)

    # Full grid present: 3 windows x 2 forms x 2 sizes x 3 hops.
    assert len(cola) == 3 * 2 * 2 * 3
    assert set(gains) == {(w, n) for w in windows for n in sizes}
    assert set(estbias) == {(w, f) for w in windows for f in ("symmetric", "periodic")}
    assert set(specshift) == set(windows)

    for n in sizes:
        for w in ("hann", "hamming"):
            # Periodic raised-cosine (k<=1 terms): exact COLA at every dividing hop.
            for hop in (n // 2, n // 4, n // 8):
                assert cola[(w, "periodic", n, hop)]["ripple"] < 1e-5
            # Symmetric misses by O(1/N): nonzero, but bounded by ~4/N.
            sym = cola[(w, "symmetric", n, n // 2)]["ripple"]
            assert 1e-7 < sym < 4.0 / n
        # Blackman (k=2 term): hop N/2 fails in BOTH conventions...
        assert cola[("blackman", "symmetric", n, n // 2)]["ripple"] > 0.1
        assert cola[("blackman", "periodic", n, n // 2)]["ripple"] > 0.1
        # ...and is fine from hop N/4 down (periodic exact).
        assert cola[("blackman", "periodic", n, n // 4)]["ripple"] < 1e-5

        # Coherent-gain shift of adopting periodic ~ +1/N: the first-order
        # amplitude delta every golden would see.
        for w in windows:
            ratio = gains[(w, n)]
            assert 1.0 < ratio < 1.0 + 3.0 / n

    # The decisive estimation fact: log-parabolic accuracy is convention-
    # insensitive (relative RMS delta under 5%) — periodic does NOT buy
    # estimator accuracy, only exact COLA.
    for w in windows:
        sym, per = estbias[(w, "symmetric")], estbias[(w, "periodic")]
        assert abs(per["rms"] - sym["rms"]) / sym["rms"] < 0.05

    # Raw analysis bins DO move: the shift is real (> float noise) but small
    # (< 2% relative magsq at N=1024) — the scale of the golden re-sign-off.
    for w in windows:
        assert 1e-6 < specshift[w] < 0.02
