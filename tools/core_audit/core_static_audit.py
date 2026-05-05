#!/usr/bin/env python3
"""Static audit checks for high-confidence Spectral Engine core invariants."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

FAILURES: list[str] = []


def read(rel: str) -> str:
    p = ROOT / rel
    if not p.exists():
        FAILURES.append(f"missing {rel}")
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def require(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


def main() -> int:
    require(not any(ROOT.glob(".spectral_core_audit_backups_*")), "audit backup directories must not be committed")

    osc = read("spectral_engine/core/spectral_osc_formulas.h")
    require("#define SPECTRAL_OSC_FORMULAS_VERSION 3" in osc, "oscillator formula version must be 3 after pass 2")
    require("return sinf(x);" in osc, "spectral_fast_sin_inline must use exact sinf by default")
    require("SPECTRAL_ENABLE_APPROX_TRIG" in osc, "approx trig must be explicitly gated")

    fm = read("spectral_engine/core/spectral_fast_math.c")
    require("return atan2f(y, x);" in fm, "fast_atan2 must be exact by default")

    vo = read("spectral_engine/core/spectral_vector_ops.c")
    require("#if !SPECTRAL_ENABLE_APPROX_ATAN2" in vo, "vector phase extraction must gate approximate atan2")

    fused = read("spectral_engine/analysis/spectral_analysis_fused.c")
    require("float* phase_prev" in fused and "float* phase_next" in fused, "fused analysis must rotate separate phase rows")
    region_start = fused.find("Global Maximum Discovery")
    region_end = fused.find("tracker = spectral_tracker_create")
    if region_start >= 0 and region_end >= 0:
        require("SPECTRAL_CUSTOM_FAST_MATH_MODE" not in fused[region_start:region_end],
                "fused max discovery must not be disabled by fast-math mode")
    else:
        require(False, "fused analysis max-discovery region not found")

    synth = read("spectral_engine/synth/backends/cpu/spectral_synth_cpu.c")
    for fn in ["segment_fn_wavetable_float", "segment_fn_native_timbre", "segment_fn_native_wavetable"]:
        pos = synth.find(f"static void {fn}")
        require(pos >= 0, f"missing {fn}")
        body = synth[pos:synth.find("}\n", pos) + 2] if pos >= 0 else ""
        require("fade_envelope" in body, f"{fn} must apply segment fade envelope")

    wavetable = read("spectral_engine/core/spectral_wavetable.c")
    require('#include "spectral_osc_formulas.h"' in wavetable, "wavetable builtins must include canonical oscillator formulas")
    require("2.0 * t - 1.0" not in wavetable, "wavetable saw must not use independent phase convention")
    require("if (!bank) return SPECTRAL_SAMPLE_ZERO;" in wavetable, "wavetable timbre lookup must guard NULL bank")

    metal = read("spectral_engine/synth/backends/gpu/metal/spectral_synth_metal.m")
    require("SPECTRAL_OSC_FORMULAS_VERSION == 3" in metal, "Metal oscillator formula guard must be version 3")
    require("SPECTRAL_METAL_FAST_MATH" in metal, "Metal fast math must be explicitly gated")

    header = read("spectral_engine/analysis/spectral_peak_track.h")
    require("VM overcommit" not in header, "peak tracker public comment must not advertise VM overcommit")

    if FAILURES:
        for f in FAILURES:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("core static audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
