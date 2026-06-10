"""Toolchain discovery for the M7 measurement stack.

The target flags are NOT defined here: they are parsed at runtime from
spectral_engine/cmake/options.cmake (SPECTRAL_DAISY_CPU/FPU/FLOAT_ABI/
OPTIMIZE) — the CMake defaults the Daisy production build uses are the
single source of truth, and a missing/renamed variable raises instead of
silently measuring a different build (C-truth rule: Python derives facts
from the C/CMake artifacts, it does not restate them).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ...core.utils import find_repo_root

NATIVE_DIR = Path(__file__).resolve().parent / "native"

ARM_GCC = "arm-none-eabi-gcc"
ARM_NM = "arm-none-eabi-nm"
QEMU_SYSTEM_ARM = "qemu-system-arm"
LLVM_MCA_CANDIDATES = ("/opt/homebrew/opt/llvm/bin/llvm-mca", "llvm-mca")

OPTIONS_CMAKE_RELPATH = "spectral_engine/cmake/options.cmake"
_DAISY_FLAG_VARS = ("SPECTRAL_DAISY_CPU", "SPECTRAL_DAISY_FPU", "SPECTRAL_DAISY_FLOAT_ABI")

M7_DEFINES = (
    "-DSPECTRAL_EMBEDDED=1",
    "-DSPECTRAL_ARM_M7=1",
    "-DSPECTRAL_HAS_DUAL_MAC=1",
)


def daisy_target_flags(repo_root: Path) -> tuple[str, ...]:
    """Target + optimization flags read from the Daisy CMake defaults.

    ``-mthumb`` is appended here: M-profile is Thumb-only and the Daisy build
    inherits it from libDaisy's make rules rather than options.cmake.
    """
    options_path = repo_root / OPTIONS_CMAKE_RELPATH
    text = options_path.read_text(encoding="utf-8")
    flags: list[str] = []
    for var in _DAISY_FLAG_VARS:
        match = re.search(rf'set\({var} "([^"]+)"', text)
        if match is None:
            raise ToolchainError(
                f"{var} not found in {OPTIONS_CMAKE_RELPATH} — the Daisy flag "
                "contract moved; fix the parse, do not hardcode flags"
            )
        flags.append(match.group(1))
    flags.append("-mthumb")
    opt = re.search(r'set\(SPECTRAL_DAISY_OPTIMIZE "([^"]+)"', text)
    if opt is None:
        raise ToolchainError(f"SPECTRAL_DAISY_OPTIMIZE not found in {OPTIONS_CMAKE_RELPATH}")
    flags.append(f"-O{opt.group(1)}")
    return tuple(flags)

ENGINE_INCLUDE_SUBDIRS = (
    "spectral_engine",
    "spectral_engine/core",
    "spectral_engine/synth",
    "spectral_engine/synth/math",
    "spectral_engine/synth/api",
    "spectral_engine/runtime",
    "spectral_engine/analysis",
)

KERNEL_TU_RELPATHS = (
    "spectral_engine/synth/backends/arm/spectral_synth_arm32.c",
    "spectral_engine/synth/math/spectral_q15.c",
    "spectral_engine/core/spectral_lut.c",
)


class ToolchainError(RuntimeError):
    """A required tool is missing or unusable; message says how to get it."""


@dataclass(frozen=True, slots=True)
class Toolchain:
    repo_root: Path
    arm_gcc: str
    target_flags: tuple[str, ...]
    arm_nm: str | None = None
    llvm_mca: str | None = None
    qemu_system_arm: str | None = None
    glib_cflags: list[str] = field(default_factory=list)
    glib_libs: list[str] = field(default_factory=list)
    qemu_include_dir: Path | None = None

    def cflags(self, *, extra_includes: tuple[Path, ...] = ()) -> list[str]:
        """Full compile flags for the real M7 TUs (freestanding, stub libc)."""
        flags: list[str] = [*self.target_flags, "-ffreestanding"]
        flags += ["-isystem", str(NATIVE_DIR / "fs_include")]
        flags += list(M7_DEFINES)
        for sub in ENGINE_INCLUDE_SUBDIRS:
            flags += ["-I", str(self.repo_root / sub)]
        for inc in extra_includes:
            flags += ["-I", str(inc)]
        return flags

    def kernel_tus(self) -> list[Path]:
        return [self.repo_root / rel for rel in KERNEL_TU_RELPATHS]


def _which_or_none(name: str) -> str | None:
    return shutil.which(name)


def _find_llvm_mca() -> str | None:
    for candidate in LLVM_MCA_CANDIDATES:
        path = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if path:
            return path
    return None


def _qemu_include_dir(qemu_bin: str) -> Path | None:
    """qemu-plugin.h ships next to the qemu prefix (brew: Cellar/qemu/include)."""
    real = Path(qemu_bin).resolve()
    for prefix in (real.parent.parent, Path("/opt/homebrew")):
        header = prefix / "include" / "qemu-plugin.h"
        if header.is_file():
            return header.parent
    return None


def _pkg_config(args: list[str]) -> list[str]:
    try:
        proc = subprocess.run(
            ["pkg-config", *args], capture_output=True, text=True, timeout=15, check=False
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    return proc.stdout.split()


def discover(repo_root: Path | None = None, *, need: frozenset[str] = frozenset()) -> Toolchain:
    """Locate the toolchain. ``need`` ⊆ {"mca", "qemu"}: raise with an
    actionable message if a needed optional tool is absent; tools outside
    ``need`` are returned as None and callers degrade explicitly."""
    root = repo_root or find_repo_root(Path(__file__))

    arm_gcc = _which_or_none(ARM_GCC)
    if arm_gcc is None:
        raise ToolchainError(
            "arm-none-eabi-gcc not found (brew install arm-none-eabi-gcc); "
            "required for every embedded measurement"
        )

    mca = _find_llvm_mca()
    if "mca" in need and mca is None:
        raise ToolchainError("llvm-mca not found (brew install llvm); required for cycle modeling")

    qemu = _which_or_none(QEMU_SYSTEM_ARM)
    qemu_inc: Path | None = None
    glib_cflags: list[str] = []
    glib_libs: list[str] = []
    if qemu is not None:
        qemu_inc = _qemu_include_dir(qemu)
        glib_cflags = _pkg_config(["--cflags", "glib-2.0"])
        glib_libs = _pkg_config(["--libs", "glib-2.0"])
    if "qemu" in need:
        if qemu is None:
            raise ToolchainError("qemu-system-arm not found (brew install qemu); required for dynamic counts")
        if qemu_inc is None:
            raise ToolchainError("qemu-plugin.h not found near qemu install; cannot build the counts plugin")
        if not glib_cflags:
            raise ToolchainError("pkg-config glib-2.0 failed; glib headers required to build the counts plugin")

    return Toolchain(
        repo_root=root,
        arm_gcc=arm_gcc,
        target_flags=daisy_target_flags(root),
        arm_nm=_which_or_none(ARM_NM),
        llvm_mca=mca,
        qemu_system_arm=qemu,
        glib_cflags=glib_cflags,
        glib_libs=glib_libs,
        qemu_include_dir=qemu_inc,
    )
