"""Firmware OS-contract purity gate for the Daisy / Cortex-M7 engine TUs (arch-05).

The layer law (test_layering.py) checks include DIRECTION, not freedom from host OS
contract: a firmware TU may legally include a kernel header that, on the host, pulls
stdio / a heap. The include-DAG check is structurally blind to it (kernel->kernel is a
legal edge; system includes are ignored). This gate closes that gap.

For each TU in SPECTRAL_SOURCES_DAISY_ENGINE it recompiles the unit in REAL-FIRMWARE mode
-- SPECTRAL_EMBEDDED=1 with the host-sim markers (SPECTRAL_EMBEDDED_SIMULATION /
SPECTRAL_USE_EMBEDDED_SYNTH) stripped, so SPECTRAL_IS_EMBEDDED_SIM resolves to 0 exactly
as the arm-none-eabi daisy build (daisy-config.cmake) does -- and asserts the object's
UNDEFINED symbols (nm -u: what the unit actually CALLS, not what a header merely declares,
so <stdlib.h>/<stdio.h> prototypes never false-trip) name no host file I/O, libc heap,
host file-system shim, or OS / threading primitive. A bare-metal target links none of these.

Runs on the host via the embedded_arm compile_commands entries -- no Daisy hardware or
libDaisy needed. The OS-contract gating keys on SPECTRAL_EMBEDDED / SPECTRAL_IS_EMBEDDED_SIM
(not on the arch built-in), so the host firmware-mode object is faithful for this check.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "spectral_engine" / "cmake" / "source-manifest.cmake"
COMPILE_DB = ROOT / "build" / "compile_commands.json"

# Host OS / libc contract a bare-metal firmware unit must not depend on. Deliberately
# omits memcpy/memset (freestanding builtins) and printf/snprintf (formatting is
# retargetable and heap-free on embedded) to keep the gate to the unambiguous classes:
# file I/O, the host-FS shim, the libc heap, and OS / threading primitives.
FORBIDDEN = re.compile(
    r"^(?:"
    r"fopen|freopen|fclose|fread|fwrite|fseek|ftell|fgets|fputs|fprintf|fscanf|sscanf|"  # stdio file I/O
    r"spectral_fs_[a-z_]+|"                                                               # host-FS shim (not in fw link)
    r"malloc|calloc|realloc|free|aligned_alloc|posix_memalign|"                           # libc heap
    r"mmap|munmap|sysconf|sbrk|getenv|system|fork|"                                       # OS
    r"pthread_[a-z_]+|omp_[a-z_]+|setlocale|gettimeofday|clock_gettime"                   # threads / locale / time
    r")$"
)

# Markers that the host-sim builds add and the real arm-none-eabi firmware does NOT
# (daisy-config.cmake defines neither) -> stripping them models real firmware.
_HOST_SIM_DEFINES = ("-DSPECTRAL_EMBEDDED_SIMULATION=1", "-DSPECTRAL_USE_EMBEDDED_SYNTH=1")


def _daisy_engine_basenames() -> list[str]:
    text = MANIFEST.read_text(encoding="utf-8")
    m = re.search(r"set\(SPECTRAL_SOURCES_DAISY_ENGINE(.*?)\)", text, re.DOTALL)
    assert m, "SPECTRAL_SOURCES_DAISY_ENGINE not found in source-manifest.cmake"
    return sorted(set(re.findall(r"([A-Za-z0-9_]+\.c)", m.group(1))))


def _embedded_command(db: list[dict], basename: str) -> str | None:
    for e in db:
        if e["file"].endswith("/" + basename):
            cmd = e.get("command") or " ".join(e.get("arguments", []))
            if "-DSPECTRAL_EMBEDDED_SIMULATION" in cmd:
                return cmd
    return None


def _firmware_mode_args(cmd: str) -> list[str]:
    """Drop the host-sim markers (and the -c/-o sink) so the object models real firmware."""
    args = shlex.split(cmd)
    out: list[str] = []
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a == "-o":
            skip = True
            continue
        if a == "-c" or a in _HOST_SIM_DEFINES:
            continue
        out.append(a)
    return out + ["-USPECTRAL_EMBEDDED_SIMULATION", "-USPECTRAL_USE_EMBEDDED_SYNTH"]


@pytest.mark.skipif(not COMPILE_DB.exists(),
                    reason="no build/compile_commands.json; configure the build first")
@pytest.mark.skipif(shutil.which("nm") is None, reason="nm unavailable")
def test_daisy_engine_tus_carry_no_host_os_contract():
    db = json.loads(COMPILE_DB.read_text(encoding="utf-8"))
    basenames = _daisy_engine_basenames()
    assert basenames, "parsed no firmware engine TUs"

    impurities: dict[str, list[str]] = {}
    checked = 0
    for base in basenames:
        cmd = _embedded_command(db, base)
        if cmd is None:
            continue  # this TU not in the configured build's embedded_arm set
        args = _firmware_mode_args(cmd)
        with tempfile.TemporaryDirectory() as td:
            obj = Path(td) / (base + ".o")
            r = subprocess.run(args + ["-c", "-o", str(obj)],
                               capture_output=True, text=True)
            assert r.returncode == 0, f"{base}: firmware-mode compile failed:\n{r.stderr}"
            nm = subprocess.run(["nm", "-u", str(obj)], capture_output=True, text=True)
        undef = {ln.split()[-1].lstrip("_") for ln in nm.stdout.splitlines() if ln.strip()}
        bad = sorted(s for s in undef if FORBIDDEN.match(s))
        if bad:
            impurities[base] = bad
        checked += 1

    assert checked, ("no embedded_arm compile entries for the firmware engine set; "
                     "configure/build the embedded_arm target first")
    assert not impurities, (
        "firmware engine TUs reference host OS/libc contract -- guard the host-only code under\n"
        "  #if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM   (arch-05):\n  "
        + "\n  ".join(f"{name}: {syms}" for name, syms in sorted(impurities.items())))
