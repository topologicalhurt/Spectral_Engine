#!/usr/bin/env python3
"""F1 gate for the IFFT-synthesis frame-builder math (IFFT_SYNTHESIS_PLAN).

Compiles and runs harnesses/ifft_synth_sweep.c and pins the MEASURED error
floors (this harness is the measurement; double precision, N=512/hop 256,
periodic Hann):
  COLA exact; frame truncation floor K=8 ~ -55 dBFS, K=12 ~ -63 dBFS
  (decay ~1/d^3 -- measurement replaced the plan's -80 dB guess);
  interpolation negligible at O>=16; stream RMS -83..-92 dBFS.
Bounds below carry ~3 dB headroom; a breach means the frame-builder math
or the motif changed, and F2+ must re-derive its error budget.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FRAME_RE = re.compile(r"^FRAME k=(\d+) o=(\d+) max_err_dbfs=(-?[\d.]+)$")
STREAM_RE = re.compile(
    r"^STREAM k=(\d+) o=(\d+) partials=(\d+) max_err_dbfs=(-?[\d.]+) rms_err_dbfs=(-?[\d.]+)$")
COLA_RE = re.compile(r"^COLA min=([\d.]+) max=([\d.]+)$")


def _run() -> str:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("clang")
    assert cc
    with tempfile.TemporaryDirectory(prefix="spectral-ifft-") as tmp:
        exe = Path(tmp) / "ifft_sweep"
        subprocess.run([cc, "-O2", "-std=c11",
                        str(ROOT / "tests/core_math/harnesses/ifft_synth_sweep.c"),
                        "-lm", "-o", str(exe)], check=True, cwd=ROOT)
        return subprocess.run([str(exe)], check=True, text=True,
                              capture_output=True, cwd=ROOT).stdout


def test_ifft_frame_builder_error_floors():
    out = _run()
    frames, streams, cola = {}, {}, None
    for line in out.splitlines():
        if m := FRAME_RE.match(line.strip()):
            frames[(int(m[1]), int(m[2]))] = float(m[3])
        elif m := STREAM_RE.match(line.strip()):
            streams[(int(m[1]), int(m[2]), int(m[3]))] = (float(m[4]), float(m[5]))
        elif m := COLA_RE.match(line.strip()):
            cola = (float(m[1]), float(m[2]))

    # COLA: periodic Hann at 50% must be exact to double precision.
    assert cola is not None and abs(cola[0] - 1.0) < 1e-9 and abs(cola[1] - 1.0) < 1e-9

    # Frame truncation floors (measured values above, +3 dB headroom).
    assert frames[(4, 64)] < -40.0
    assert frames[(8, 64)] < -52.0
    assert frames[(12, 64)] < -59.0
    # Monotone in K: more motif taps must not get worse.
    assert frames[(12, 64)] < frames[(8, 64)] < frames[(4, 64)]
    # Interpolation negligible: O=16 within 1 dB of O=64.
    for k in (4, 8, 12):
        assert abs(frames[(k, 16)] - frames[(k, 64)]) < 1.0

    # Stream parity at K=8 (the F2 candidate operating point).
    for partials in (1, 16, 64):
        mx, rms = streams[(8, 64, partials)]
        assert mx < -60.0, f"stream max err too high at {partials} partials"
        assert rms < -80.0, f"stream rms err too high at {partials} partials"
