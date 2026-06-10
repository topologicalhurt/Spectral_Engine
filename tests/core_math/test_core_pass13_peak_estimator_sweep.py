#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


RESULT_RE = re.compile(
    r"^RESULT window=(?P<window>\S+) estimator=(?P<estimator>\S+) "
    r"snr_db=(?P<snr>[0-9.]+) cases=(?P<cases>\d+) valid=(?P<valid>\d+) "
    r"valid_rate=(?P<valid_rate>[0-9.]+) rms_err=(?P<rms>[0-9.eE+-]+) "
    r"max_abs_err=(?P<max>[0-9.eE+-]+) fallback_rate=(?P<fallback>[0-9.]+) "
    r"complex_rate=(?P<complex>[0-9.]+) avg_ns=(?P<avg_ns>[0-9.eE+-]+)$"
)


def parse_results(output: str) -> dict[tuple[str, str, float], dict[str, float]]:
    rows: dict[tuple[str, str, float], dict[str, float]] = {}
    for line in output.splitlines():
        match = RESULT_RE.match(line.strip())
        if not match:
            continue
        window = match.group("window")
        estimator = match.group("estimator")
        snr = float(match.group("snr"))
        rows[(window, estimator, snr)] = {
            "cases": float(match.group("cases")),
            "valid": float(match.group("valid")),
            "valid_rate": float(match.group("valid_rate")),
            "rms": float(match.group("rms")),
            "max": float(match.group("max")),
            "fallback": float(match.group("fallback")),
            "complex": float(match.group("complex")),
            "avg_ns": float(match.group("avg_ns")),
        }
    return rows


def test_peak_estimator_sweep_harness_compiles_and_reports_ground_truth_error() -> None:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    assert cc, "no C compiler available"

    with tempfile.TemporaryDirectory(prefix="spectral-pass13-sweep-") as tmp:
        exe = Path(tmp) / "peak_estimator_sweep"
        link_flags = ["-framework", "Accelerate"] if sys.platform == "darwin" else []
        subprocess.run(
            [
                cc,
                "-O2",
                "-std=c11",
                "-DSPECTRAL_PEAK_SWEEP_N=512",
                "-DSPECTRAL_PEAK_SWEEP_OFFSETS=21",
                "-DSPECTRAL_PEAK_SWEEP_ITERS=2",
                "-I",
                str(ROOT / "spectral_engine/core"),
                "-I",
                str(ROOT / "spectral_engine/analysis"),
                "-I",
                str(ROOT / "spectral_engine/runtime"),
                str(ROOT / "spectral_engine/analysis/spectral_peak_estimator.c"),
                str(ROOT / "spectral_engine/core/spectral_fast_math.c"),
                str(ROOT / "spectral_engine/core/spectral_windows.c"),
                str(ROOT / "tests/core_math/harnesses/peak_estimator_sweep.c"),
                "-lm",
                *link_flags,
                "-o",
                str(exe),
            ],
            check=True,
            cwd=ROOT,
        )
        completed = subprocess.run([str(exe)], check=True, text=True, capture_output=True, cwd=ROOT)

    rows = parse_results(completed.stdout)
    windows = {"hann", "hamming", "blackman", "rectangular"}
    estimators = {
        "log-parabolic",
        "mag-parabolic",
        "jacobsen-complex",
        "candan-complex",
        "quinn-second",
    }
    snrs = {120.0, 60.0, 30.0}

    assert len(rows) == len(windows) * len(estimators) * len(snrs)

    for window in windows:
        for estimator in estimators:
            for snr in snrs:
                row = rows[(window, estimator, snr)]
                assert row["cases"] > 0
                assert row["valid_rate"] > 0.90
                assert row["avg_ns"] >= 0.0

    # This is the one hard correctness guard: the default estimator must remain
    # accurate on the engine's current Hann-windowed, high-SNR signal model.
    hann_log = rows[("hann", "log-parabolic", 120.0)]
    assert hann_log["max"] <= 0.035
    assert hann_log["fallback"] == 0.0


