#!/usr/bin/env python3
"""GPU (Metal) vs CPU backend parity over the real CLI pipeline.

Runs the desktop binary twice per timbre — backend forced to CPU, then to
Metal — on the same input, and bounds the difference of the rendered float
WAVs. Covers every GPU-eligible timbre (0..SPECTRAL_GPU_MAX_TIMBRE = parabola),
so the Metal df (chirp) + da (amp-ramp) path is exercised across the full
surface, not just two timbres. Measured floors on the 440 Hz fixture (all six):
worst-case max 2.4e-7 (-132.5 dBFS) / RMS 6.4e-8 (-143.9 dBFS); per-timbre
max|diff| 1.2e-7..2.4e-7, RMS 1.6e-8..6.4e-8. The bounds below carry ~30 dB
headroom, far below audibility yet far above any real formula or dispatch
defect (those land at -40..-70 dBFS).

Standalone (stdlib only) so CTest can run it without pytest. Exit codes:
0 = pass, 77 = skipped (binary not built, or no Metal device — e.g. a
virtualized CI host; the binary's fallback to CPU is detected and treated
as a skip, never a silent pass), 1 = parity failure.

Usage: gpu_backend_parity.py <desktop-binary> <input-wav>
"""

from __future__ import annotations

import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SKIP = 77
TIMBRES = (0, 1, 2, 3, 4, 5)  # all GPU-eligible: sine,saw,square,triangle,asin,parabola (<=SPECTRAL_GPU_MAX_TIMBRE)
MAX_DIFF_BOUND = 1.0e-5     # ~-100 dBFS
RMS_DIFF_BOUND = 1.0e-6     # ~-120 dBFS
CPU, METAL = "1", "2"


def load_f32_wav(path: Path) -> tuple[float, ...]:
    riff = path.read_bytes()
    if riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a WAV")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(riff):
        cid = riff[pos:pos + 4]
        size = struct.unpack("<I", riff[pos + 4:pos + 8])[0]
        body = riff[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError(f"{path}: missing fmt/data chunk")
    if fmt[0] != 3 or fmt[5] != 32:
        raise ValueError(f"{path}: expected float32 WAV, got {fmt}")
    return struct.unpack(f"<{len(data) // 4}f", data)


def render(binary: Path, wav: Path, timbre: int, backend: str,
           workdir: Path) -> tuple[str, Path]:
    """One CLI render in an isolated cwd; returns (stdout, output wav)."""
    run_dir = workdir / f"t{timbre}_b{backend}" / "d"
    run_dir.mkdir(parents=True)
    proc = subprocess.run(
        [str(binary), str(wav), str(timbre),
         "1.0", "0", "1024", "256", "-70.0", "1", backend],
        cwd=run_dir, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr, sep="\n")
        raise RuntimeError(f"render failed (timbre {timbre}, backend {backend})")
    # The output WAV is named after the input stem (unique per input), not a fixed
    # out_c.wav; run_dir is fresh so exactly one .wav is produced — glob for it.
    wavs = sorted((run_dir / "output").glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"render produced no .wav (timbre {timbre}, backend {backend})")
    return proc.stdout, wavs[0]


def main() -> int:
    binary, wav = Path(sys.argv[1]), Path(sys.argv[2])
    if not binary.exists():
        print(f"SKIP: desktop binary not built ({binary})")
        return SKIP

    with tempfile.TemporaryDirectory(prefix="spectral-gpu-parity-") as tmp:
        for timbre in TIMBRES:
            gpu_out, gpu_wav = render(binary, wav, timbre, METAL, Path(tmp))
            if "Backend used: Metal" not in gpu_out:
                print("SKIP: no Metal device (binary fell back to CPU)")
                return SKIP
            cpu_out, cpu_wav = render(binary, wav, timbre, CPU, Path(tmp))
            if "Backend used: CPU" not in cpu_out:
                raise RuntimeError("CPU run did not report CPU backend")

            a, b = load_f32_wav(cpu_wav), load_f32_wav(gpu_wav)
            if len(a) != len(b):
                print(f"FAIL timbre {timbre}: length {len(a)} vs {len(b)}")
                return 1
            mx = max(abs(x - y) for x, y in zip(a, b))
            rms = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))
            print(f"timbre {timbre}: max|diff| {mx:.3e}  rms {rms:.3e}")
            if mx > MAX_DIFF_BOUND or rms > RMS_DIFF_BOUND:
                print(f"FAIL timbre {timbre}: bounds max<{MAX_DIFF_BOUND:g} "
                      f"rms<{RMS_DIFF_BOUND:g}")
                return 1
    print("PASS: Metal output matches CPU within bounds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
