"""Build-flag hygiene gate (kernel-hardening VI/VII).

Pins the arch-gating of x86-only SIMD-ISA flags. The bug this catches: `-mavx2`
and `-mno-avx512f` were applied to every host including arm64, where they are
`-Wunused-command-line-argument` noise (88 warnings on `desktop`, 362 on
`tests_all`). The fix (host-config.cmake) emits those flags only on x86.

The test configures the real CMake project (export compile commands) and reads
the resolved per-TU flags, so it asserts what the compiler ACTUALLY receives,
not what a string in a .cmake file says. Fail-on-bug: drop the arch-gate and the
arm64 host re-acquires the flags -> this test fails.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

X86_ISA_FLAGS = ("-mavx2", "-mno-avx512f")


def _is_x86() -> bool:
    m = platform.machine().lower()
    return m in ("x86_64", "amd64", "i386", "i486", "i586", "i686")


def _configure_and_read_flags(tmp_path: Path) -> list[str]:
    """Configure the project with compile-command export and return every
    compiler argument seen across all TUs (deduped is unnecessary — we only
    membership-test)."""
    if shutil.which("cmake") is None:
        pytest.skip("cmake not on PATH")
    build = tmp_path / "b"
    cfg = subprocess.run(
        ["cmake", "-S", str(ROOT), "-B", str(build),
         "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"],
        capture_output=True, text=True,
    )
    if cfg.returncode != 0:
        pytest.skip(f"configure failed in this environment:\n{cfg.stderr[-2000:]}")
    ccdb = build / "compile_commands.json"
    if not ccdb.exists():
        pytest.skip("compile_commands.json not generated (generator lacks support)")
    args: list[str] = []
    for entry in json.loads(ccdb.read_text()):
        if "arguments" in entry:
            args.extend(entry["arguments"])
        else:
            args.extend(entry.get("command", "").split())
    return args


def test_x86_isa_flags_absent_on_non_x86(tmp_path):
    """On a non-x86 host (this arm64 box), the x86 SIMD-ISA flags must NOT reach
    any TU — that is the warning fix and the arch-gate contract."""
    if _is_x86():
        pytest.skip("x86 host: the flags are expected PRESENT here (see the x86 test)")
    args = _configure_and_read_flags(tmp_path)
    present = sorted({f for f in X86_ISA_FLAGS if f in args})
    assert not present, (
        f"x86-only ISA flags leaked onto a non-x86 host: {present}. The arch-gate "
        "in cmake/host-config.cmake regressed; these are -Wunused-command-line-"
        "argument noise on arm64."
    )


def test_x86_isa_flags_present_on_x86(tmp_path):
    """On an x86 host, AVX2 must be on and AVX-512 capped off (the intended
    x86 profile). Skipped off-x86."""
    if not _is_x86():
        pytest.skip("non-x86 host: x86 ISA flags are intentionally absent")
    args = _configure_and_read_flags(tmp_path)
    missing = sorted({f for f in X86_ISA_FLAGS if f not in args})
    assert not missing, (
        f"x86 host is missing its intended SIMD-ISA flags: {missing}. The x86 "
        "profile must keep AVX2 on and -mno-avx512f (downclock cap)."
    )


def _tu_args(tmp_path: Path, output_substr: str, file_suffix: str) -> list[str]:
    """The compiler arguments for one TU (matched by output dir + source suffix)."""
    if shutil.which("cmake") is None:
        pytest.skip("cmake not on PATH")
    build = tmp_path / "b"
    cfg = subprocess.run(
        ["cmake", "-S", str(ROOT), "-B", str(build),
         "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"],
        capture_output=True, text=True,
    )
    if cfg.returncode != 0:
        pytest.skip(f"configure failed in this environment:\n{cfg.stderr[-2000:]}")
    ccdb = build / "compile_commands.json"
    if not ccdb.exists():
        pytest.skip("compile_commands.json not generated")
    for entry in json.loads(ccdb.read_text()):
        if output_substr in entry.get("output", "") and entry["file"].endswith(file_suffix):
            return entry.get("arguments") or entry.get("command", "").split()
    pytest.skip(f"TU not found: {output_substr} / {file_suffix}")
    return []


def test_arm_correctness_harness_is_firmware_faithful(tmp_path):
    """The arm_core_test correctness harness exercises FIRMWARE fixed-point code,
    which ships precise (SAFE_MATH default) and without ThinLTO. Compiling it with
    the host's -ffast-math + -flto=thin was unfaithful and caused a rare
    nondeterministic ThinLTO miscompile (plan §VIII). Pin the firmware-faithful,
    reproducible flags. Fail-on-bug: drop the override and the harness re-acquires
    -flto=thin / fast-math semantics."""
    args = _tu_args(tmp_path, "arm_core_test.dir", "spectral_synth_arm32.c")
    assert "-fno-lto" in args, (
        "arm_core_test must compile -fno-lto: ThinLTO codegen on this fixed-point "
        "path is non-reproducible and rarely miscompiles (plan §VIII)."
    )
    assert "-fno-fast-math" in args, (
        "arm_core_test must compile -fno-fast-math to match the firmware's precise "
        "(SAFE_MATH) semantics it is meant to validate."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
