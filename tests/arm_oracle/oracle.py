#!/usr/bin/env python3
"""Interim ARM-synthesis behavior oracle (Campaign-2, ULTRAPLAN Phase A).

Captures deterministic golden hashes of the simulate build
(synth_arm32_simulation, the host-side Q15 ARM oracle) over a fixed
fixture x argument matrix, so the Phase A refactors (arch split, NEON
removal, A1 fixes) can be shown behavior-preserving before the compiled
CTest harness (Phase D / Phase G) exists.

Scope and limits:
  - Hashes are environment-specific (depend on this compiler/arch build of
    the sim binary). They are a *same-machine before/after* regression
    anchor, NOT a cross-machine golden. Re-capture after rebuilding the sim
    on a different toolchain.
  - Superseded by the tolerance-based compiled parity harness in Phase D.

Usage:
  python tests/arm_oracle/oracle.py gen        # (re)generate input fixtures
  python tests/arm_oracle/oracle.py capture    # run matrix, write goldens.json
  python tests/arm_oracle/oracle.py check       # re-run matrix, compare to goldens.json
  python tests/arm_oracle/oracle.py diff A B     # max/RMS PCM diff of two float WAVs
"""
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
GOLDENS = HERE / "goldens.json"
SIM = ROOT / "build/bin/spectral_arm64_native_simulation"
DEFAULT_OUT = ROOT / "output/out_c.wav"
CACHE_DIRS = (ROOT / "output/cache", ROOT / "output/pgo")

SR = 44100
NFRAMES = SR // 2  # 0.5 s

# (fixture, timbre, stretch, pitch, n_fft, hop, db_thresh)
# silence.wav is generated but intentionally excluded: a 0-segment input is
# rejected by the CLI as an "input error" (nothing to render) before synthesis,
# so it exercises no ARM-synth behavior.
MATRIX = [
    ("sine_bin.wav",    0, 1.0, 0, 1024, 256, -70),
    ("sine_offbin.wav", 0, 1.0, 0, 1024, 256, -70),
    ("two_tone.wav",    0, 1.0, 0, 1024, 256, -70),
    ("sine_bin.wav",    0, 1.5, 0, 1024, 256, -70),
    ("sine_offbin.wav", 0, 1.5, 0, 1024, 256, -70),
    ("two_tone.wav",    0, 1.5, 0, 1024, 256, -70),
]


def _write_pcm16(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        buf = bytearray()
        for s in samples:
            v = int(round(max(-1.0, min(1.0, s)) * 32767.0))
            buf += struct.pack("<h", v)
        w.writeframes(bytes(buf))


def _sine(freq, amp=0.5):
    k = 2.0 * math.pi * freq / SR
    return [amp * math.sin(k * n) for n in range(NFRAMES)]


def gen():
    _write_pcm16(FIXTURES / "silence.wav", [0.0] * NFRAMES)
    bin_freq = 10 * SR / 1024.0          # bin-centered for n_fft=1024
    off_freq = 10.5 * SR / 1024.0        # halfway between bins
    _write_pcm16(FIXTURES / "sine_bin.wav", _sine(bin_freq))
    _write_pcm16(FIXTURES / "sine_offbin.wav", _sine(off_freq))
    two = [0.4 * math.sin(2 * math.pi * bin_freq * n / SR)
           + 0.4 * math.sin(2 * math.pi * bin_freq * 2.5 * n / SR)
           for n in range(NFRAMES)]
    _write_pcm16(FIXTURES / "two_tone.wav", two)
    print("generated 4 fixtures in", FIXTURES)


def _clean_cache():
    for d in CACHE_DIRS:
        shutil.rmtree(d, ignore_errors=True)


def _run_case(case):
    fixture, timbre, stretch, pitch, nfft, hop, thresh = case
    inp = FIXTURES / fixture
    if not inp.exists():
        raise SystemExit("missing fixture %s; run 'gen' first" % inp)
    if not SIM.exists():
        raise SystemExit("missing sim binary %s; run 'make simulate' first" % SIM)
    _clean_cache()
    if DEFAULT_OUT.exists():
        DEFAULT_OUT.unlink()
    cmd = [str(SIM), str(inp), str(timbre), str(stretch), str(pitch),
           str(nfft), str(hop), str(thresh)]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    m = re.search(r"Wrote output:\s*(\S+)", r.stdout)
    outp = (ROOT / m.group(1)) if m else DEFAULT_OUT
    if r.returncode != 0 or not outp.exists():
        raise SystemExit("sim failed for %s (rc=%d)\n%s"
                         % (case, r.returncode, r.stderr[-800:]))
    return outp.read_bytes()


def _data_bytes(data):
    """Return raw bytes of the 'data' chunk payload (audio only), else None.

    We hash THIS, not the whole file: the engine writes a WAV PEAK chunk whose
    timeStamp field tracks wall-clock time, so whole-file hashes differ between
    runs in different seconds even when the audio is byte-identical.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    i = 12
    while i + 8 <= len(data):
        cid = data[i:i + 4]
        sz = struct.unpack("<I", data[i + 4:i + 8])[0]
        if cid == b"data":
            return data[i + 8:i + 8 + sz]
        i += 8 + sz + (sz & 1)
    return None


def _float_pcm(data):
    """Return float32 samples from a fmt=3 32-bit WAV, else None."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    i, fmt, bits = 12, None, None
    while i + 8 <= len(data):
        cid = data[i:i + 4]
        sz = struct.unpack("<I", data[i + 8 - 4:i + 8])[0]
        body = data[i + 8:i + 8 + sz]
        if cid == b"fmt " and len(body) >= 16:
            fmt = struct.unpack("<H", body[0:2])[0]
            bits = struct.unpack("<H", body[14:16])[0]
        elif cid == b"data":
            if fmt == 3 and bits == 32:
                n = sz // 4
                return list(struct.unpack("<%df" % n, body[:n * 4]))
            return None
        i += 8 + sz + (sz & 1)
    return None


def _key(c):
    return "%s|t%d|s%g|p%d|n%d|h%d|d%d" % c


def _summ(data):
    body = _data_bytes(data)
    pcm = _float_pcm(data)
    return {
        "sha256": hashlib.sha256(body if body is not None else data).hexdigest(),
        "pcm_bytes": (len(body) if body is not None else len(data)),
        "nsamp": (len(pcm) if pcm is not None else None),
        "peak": (max((abs(x) for x in pcm), default=0.0)
                 if pcm is not None else None),
    }


def capture():
    g = {}
    for c in MATRIX:
        g[_key(c)] = _summ(_run_case(c))
        print("captured", _key(c), g[_key(c)]["sha256"][:12])
    GOLDENS.write_text(json.dumps(g, indent=2) + "\n")
    print("wrote", GOLDENS, "(%d cases)" % len(g))


def check():
    if not GOLDENS.exists():
        raise SystemExit("no goldens.json; run 'capture' first")
    g = json.loads(GOLDENS.read_text())
    fails = 0
    for c in MATRIX:
        k = _key(c)
        cur = _summ(_run_case(c))
        exp = g.get(k)
        if exp is None:
            print("NEW   ", k, "(not in goldens)")
            fails += 1
        elif cur["sha256"] == exp["sha256"]:
            print("OK    ", k)
        else:
            note = "peak %.6g->%.6g nsamp %s->%s" % (
                exp.get("peak") if exp.get("peak") is not None else float("nan"),
                cur["peak"] if cur["peak"] is not None else float("nan"),
                exp.get("nsamp"), cur["nsamp"])
            print("FAIL  ", k, "hash differs;", note)
            fails += 1
    if fails:
        raise SystemExit("%d/%d cases differ from goldens" % (fails, len(MATRIX)))
    print("all %d cases match goldens" % len(MATRIX))


def diff():
    a, b = Path(sys.argv[2]), Path(sys.argv[3])
    pa, pb = _float_pcm(a.read_bytes()), _float_pcm(b.read_bytes())
    if pa is None or pb is None:
        raise SystemExit("could not parse float PCM from one of the inputs")
    identical = a.read_bytes() == b.read_bytes()
    n = min(len(pa), len(pb))
    maxd = max((abs(pa[i] - pb[i]) for i in range(n)), default=0.0)
    rms = math.sqrt(sum((pa[i] - pb[i]) ** 2 for i in range(n)) / n) if n else 0.0
    print("byte-identical:", identical)
    print("nsamp: %d vs %d" % (len(pa), len(pb)))
    print("max abs diff: %.9g" % maxd)
    print("rms diff:     %.9g" % rms)


def main():
    cmds = {"gen": gen, "capture": capture, "check": check, "diff": diff}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        raise SystemExit(__doc__)
    if sys.argv[1] == "diff" and len(sys.argv) < 4:
        raise SystemExit("usage: oracle.py diff A.wav B.wav")
    cmds[sys.argv[1]]()


if __name__ == "__main__":
    main()
