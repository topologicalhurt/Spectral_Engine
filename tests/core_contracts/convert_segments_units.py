#!/usr/bin/env python3
"""End-to-end unit contract of the convert_segments tool.

Runs the REAL binary on a generated SPEC (.bin) file and asserts every
encoded SPQ field against an independent Python mirror of the named
boundary conversions (a sanctioned, labeled independent-verification
implementation — the C truth is spectral_q.h). This pins the UNITS of
every field: the kernel applies amp_q15 + da_q15*offset per SAMPLE, so a
converter that rescales da/df (the escaped defect class: a 1/1000 "per
millisecond" scaling quantized typical fades to da_q15 == 0) fails here
on the exact field that moved.

Standalone (stdlib only) for CTest. Exit codes: 0 pass, 77 skip (binary
not built), 1 failure.

Usage: convert_segments_units.py <convert_segments-binary>
"""

from __future__ import annotations

import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SKIP = 77

# SPEC format (spectral_segment_parser.h): 28-byte header + 64-byte Segments.
SPEC_MAGIC = b"SPEC"
SPEC_VERSION = 2
# SPQ format: 32-byte header + 16-byte SpectralSegmentQ15 (chirp build).
SPQ_MAGIC = 0x31515053
Q15_RECORD = struct.Struct("<IHHhhhh")
SAMPLE_RATE = 48000

# Test segments chosen to exercise the defect class: |da|,|df| in the
# per-sample magnitude range analysis actually emits (1e-5..1e-3), where a
# wrong 1/1000 rescale lands the Q15 encoding on exactly 0.
SEGMENTS = (
    # start, length, phase,  omega,   df,      amp,  da
    (0.0,    1000.0,  0.30,  0.0576,  0.0,     0.80, -2.0e-4),
    (480.0,   512.0, -1.10,  0.5236,  3.0e-5,  0.50,  4.0e-4),
    (960.0,  2048.0,  2.50,  0.1234, -1.0e-5,  0.25, -5.0e-5),
)


def f32(x: float) -> float:
    """Round a Python double to float32 (the converter works in float)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def float_to_q15(f: float) -> int:
    f = f32(f)
    if f != f:
        return 0
    if f >= 1.0:
        return 32767
    if f <= -1.0:
        return -32768
    return int(f32(f * f32(32768.0)))   # C cast truncates toward zero


def omega_to_q88(omega: float) -> int:
    o = f32(omega)
    if not math.isfinite(o) or o <= 0.0:
        return 0
    if o > 255.0:
        o = f32(o / 4.0)
    if o > 255.0:
        o = 255.0
    return int(f32(o * 256.0))


def phase_rad_to_q15(rad: float) -> int:
    if not math.isfinite(rad):
        return 0
    two_pi = f32(2.0 * math.pi)
    n = f32(f32(math.fmod(f32(rad), two_pi)) / two_pi)
    if n < 0.0:
        n = f32(n + 1.0)
    if n >= 1.0:
        n = f32(n - 1.0)
    return int(f32((n - 0.5) * 65536.0))


def write_spec(path: Path) -> None:
    header = struct.pack("<4sIIffII", SPEC_MAGIC, SPEC_VERSION, SAMPLE_RATE,
                         1.0, 0.0, len(SEGMENTS), 0)
    body = b""
    for start, length, phase, omega, df, amp, da in SEGMENTS:
        body += struct.pack("<7f", start, length, phase, omega, df, amp, da)
        body += b"\x00" * 36   # union pad to the 64-byte Segment
    path.write_bytes(header + body)


def main() -> int:
    binary = Path(sys.argv[1])
    if not binary.exists():
        print(f"SKIP: convert_segments not built ({binary})")
        return SKIP

    fails = 0
    with tempfile.TemporaryDirectory(prefix="spectral-spq-") as tmp:
        spec, spq = Path(tmp) / "in.bin", Path(tmp) / "out.spq"
        write_spec(spec)
        proc = subprocess.run([str(binary), str(spec), str(spq)],
                              capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            print(proc.stdout, proc.stderr, sep="\n")
            print("FAIL: converter exited nonzero")
            return 1

        raw = spq.read_bytes()
        magic, _ver, sr, count, _outlen, flags, _r0, _r1 = struct.unpack_from(
            "<8I", raw)
        if magic != SPQ_MAGIC or sr != SAMPLE_RATE or count != len(SEGMENTS):
            print(f"FAIL: header magic={magic:#x} sr={sr} count={count}")
            return 1
        if flags != 1:
            print(f"SKIP: compact (no-chirp) build, flags={flags} — "
                  "field layout differs; units pinned on the chirp build")
            return SKIP

        for i, (start, length, phase, omega, df, amp, da) in enumerate(SEGMENTS):
            got = Q15_RECORD.unpack_from(raw, 32 + i * Q15_RECORD.size)
            want = (int(start), int(length), omega_to_q88(omega),
                    phase_rad_to_q15(phase), float_to_q15(amp),
                    float_to_q15(da), float_to_q15(df))
            names = ("start", "length", "freq_q88", "phase_q15",
                     "amp_q15", "da_q15", "df_q15")
            for name, g, w in zip(names, got, want):
                tol = 1 if name in ("freq_q88", "phase_q15", "amp_q15",
                                    "da_q15", "df_q15") else 0
                if abs(g - w) > tol:
                    print(f"FAIL: seg {i} {name}: got {g}, want {w}")
                    fails += 1
            # The defect class explicitly: a fade this size must NOT vanish.
            if abs(f32(da)) >= 1e-4 and got[5] == 0:
                print(f"FAIL: seg {i} da_q15 == 0 for da={da} — fade erased")
                fails += 1

    if fails:
        return 1
    print(f"PASS: {len(SEGMENTS)} segments, all fields unit-exact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
