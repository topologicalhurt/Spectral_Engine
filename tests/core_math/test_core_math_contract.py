#!/usr/bin/env python3
"""Lightweight property tests for Spectral Engine core math contracts.

These tests do not compile the C code. They check the mathematical contracts that
source implementations must satisfy and also scan patched source text for the
most severe previously observed mistakes.
"""
from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TWO_PI = 2.0 * math.pi
PI = math.pi


def wrap_phase(p: float) -> float:
    return p - TWO_PI * math.floor(p / TWO_PI + 0.5)


def fade(x: float) -> float:
    return 0.5 * (1.0 + math.sin((x - 0.5) * PI))


def assert_close(name: str, got: float, want: float, tol: float = 1e-6) -> None:
    if abs(got - want) > tol:
        raise AssertionError(f"{name}: got {got}, want {want}, tol {tol}")


def test_phase_wrap() -> None:
    assert_close("wrap(0)", wrap_phase(0.0), 0.0)
    for k in range(-16, 17):
        assert_close(f"wrap({k}*2pi)", wrap_phase(k * TWO_PI), 0.0, 1e-5)
    assert -PI - 1e-6 <= wrap_phase(PI) <= PI + 1e-6
    assert_close("wrap(pi)", wrap_phase(PI), -PI, 1e-6)
    assert_close("wrap(-pi)", wrap_phase(-PI), -PI, 1e-6)


def test_fade_contract() -> None:
    assert_close("fade(0)", fade(0.0), 0.0, 1e-6)
    assert_close("fade(1)", fade(1.0), 1.0, 1e-6)
    prev = -1.0
    for i in range(0, 101):
        y = fade(i / 100.0)
        if y < prev - 1e-9:
            raise AssertionError("fade is not monotonic nondecreasing")
        if not (0.0 <= y <= 1.0):
            raise AssertionError(f"fade out of range: {y}")
        prev = y


def test_source_text() -> None:
    osc = ROOT / "spectral_engine/core/spectral_osc_formulas.h"
    env = ROOT / "spectral_engine/core/spectral_envelope.c"
    if osc.exists():
        text = osc.read_text(encoding="utf-8")
        forbidden = "norm - floorf(norm) - 0.5f"
        if forbidden in text:
            raise AssertionError(f"forbidden phase wrap pattern remains in {osc}")
        if "1.0f - spectral_fast_sin_inline" in text:
            raise AssertionError(f"inverted fade pattern remains in {osc}")
    if env.exists():
        text = env.read_text(encoding="utf-8")
        if "1.0f - fast_sin" in text:
            raise AssertionError(f"inverted fade pattern remains in {env}")


def main() -> int:
    test_phase_wrap()
    test_fade_contract()
    test_source_text()
    print("core math contract tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
