#!/usr/bin/env python3
"""Static audit for Spectral Engine core anti-patterns.

This is a grep-level tool, not a proof system. It is intended to catch the
canonical repeated mistakes described in docs/core_audit/AI_CANON.md.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

@dataclass(frozen=True)
class Rule:
    name: str
    needle: str
    severity: str
    message: str

RULES = [
    Rule(
        "bad_phase_wrap_floor_minus_half",
        "norm - floorf(norm) - 0.5f",
        "error",
        "phase normalization maps phase zero to -pi",
    ),
    Rule(
        "bad_phase_wrap_shader",
        "norm - floor(norm) - 0.5f",
        "error",
        "shader phase normalization maps phase zero to -pi",
    ),
    Rule(
        "inverted_fast_sin_fade",
        "1.0f - fast_sin((",
        "error",
        "fade ramp is likely inverted; fade-in starts at 1",
    ),
    Rule(
        "inverted_canonical_fade",
        "1.0f - spectral_fast_sin_inline((",
        "error",
        "canonical fade ramp is likely inverted",
    ),
    Rule(
        "empirical_boost_factor",
        "Empirical boost factor",
        "warn",
        "empirical interpolation constants require derivation and tests",
    ),
    Rule(
        "quake_inv_sqrt",
        "0x5f3759df",
        "warn",
        "Quake inverse sqrt must be opt-in and error-bounded",
    ),
    Rule(
        "overcommit_assumption",
        "Linux overcommit guarantees",
        "error",
        "kernel memory behavior must not rely on Linux overcommit",
    ),
    Rule(
        "zero_cost_claim",
        "zero-cost",
        "warn",
        "zero-cost performance claims must be measured or removed",
    ),
    Rule(
        "near_exact_claim",
        "near-exact",
        "warn",
        "near-exact numerical claims must include error bounds",
    ),
]

SCAN_DIRS = [
    "spectral_engine/core",
    "spectral_engine/analysis",
    "spectral_engine/synth",
]


def iter_files(root: Path) -> Iterable[Path]:
    for rel in SCAN_DIRS:
        d = root / rel
        if not d.exists():
            continue
        for path in d.rglob("*"):
            if path.suffix.lower() in {".c", ".h", ".cu", ".m", ".mm"}:
                yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    failures = 0
    warnings = 0
    for path in iter_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for rule in RULES:
            if rule.needle in text:
                rel = path.relative_to(root)
                print(f"{rule.severity.upper()}: {rel}: {rule.name}: {rule.message}")
                if rule.severity == "error":
                    failures += 1
                else:
                    warnings += 1
    print(f"static audit complete: {failures} errors, {warnings} warnings")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
