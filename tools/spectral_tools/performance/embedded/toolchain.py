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
    try:
        text = options_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ToolchainError(f"cannot read {OPTIONS_CMAKE_RELPATH}: {exc}") from exc
    flags: list[str] = []
    for var in _DAISY_FLAG_VARS:
        match = re.search(rf'set\({var} "([^"]+)"', text)
        if match is None:
            raise ToolchainError(
                f"{var} not found in {OPTIONS_CMAKE_RELPATH} — the Daisy flag "
                "contract moved; fix the parse, do not hardcode flags"
            )
        # A cmake value may hold several flags; split so argv stays well-formed.
        flags.extend(match.group(1).split())
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
    "spectral_engine/synth/api",
    "spectral_engine/runtime",
    "spectral_engine/analysis",
)

KERNEL_TU_RELPATHS = (
    "spectral_engine/arch/arm/spectral_synth_arm32.c",
    "spectral_engine/core/spectral_q15.c",
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
        """Full compile flags for the real M7 TUs.

        Headers come from the toolchain's real newlib (discovery requires it);
        -ffreestanding is kept deliberately — measured byte-identical census
        vs hosted on the production TU, and the QEMU runner genuinely is
        freestanding (own startup, no libc init)."""
        flags: list[str] = [*self.target_flags, "-ffreestanding"]
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


def _has_newlib(gcc: str) -> bool:
    """True if the gcc resolves a real libc for the M7 multilib. A bare
    toolchain echoes the literal name back; newlib returns an absolute path."""
    try:
        proc = subprocess.run(
            [gcc, "-mcpu=cortex-m7", "-mthumb", "-print-file-name=libc_nano.a"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except OSError:
        return False
    candidate = proc.stdout.strip()
    return proc.returncode == 0 and "/" in candidate and Path(candidate).is_file()


def _local_toolchain_gccs(repo_root: Path) -> list[str]:
    """Bootstrap-installed toolchains under tools/toolchains/, newest first."""
    toolchains_dir = repo_root / "tools" / "toolchains"
    if not toolchains_dir.is_dir():
        return []
    hits = sorted(toolchains_dir.glob("*/bin/arm-none-eabi-gcc"), reverse=True)
    return [str(p) for p in hits if p.is_file()]


def find_arm_gcc(repo_root: Path) -> str:
    """Pick the arm-none-eabi-gcc to measure with: must have newlib.

    Preference: bootstrap-installed toolchain (tools/toolchains/) over PATH —
    the brew formula ships no libc and is rejected by the newlib probe. The
    stub-header workaround is gone on purpose: measurements must compile
    against the same C library family the Daisy firmware links (newlib)."""
    candidates = [*_local_toolchain_gccs(repo_root)]
    path_gcc = _which_or_none(ARM_GCC)
    if path_gcc:
        candidates.append(path_gcc)
    for gcc in candidates:
        if _has_newlib(gcc):
            return gcc
    if candidates:
        raise ToolchainError(
            f"arm-none-eabi-gcc found ({candidates[0]}) but it has no newlib "
            "(brew's is bare). Run: python3 -m spectral_tools.testing.benchmark_workflow "
            "m7-bootstrap   (downloads the sha-pinned xPack toolchain into tools/toolchains/)"
        )
    raise ToolchainError(
        "no arm-none-eabi-gcc found. Run: python3 -m spectral_tools.testing."
        "benchmark_workflow m7-bootstrap   (or install a newlib-capable toolchain)"
    )


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

    arm_gcc = find_arm_gcc(root)

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

    arm_nm = str(Path(arm_gcc).parent / ARM_NM)
    if not Path(arm_nm).is_file():
        arm_nm = _which_or_none(ARM_NM)

    return Toolchain(
        repo_root=root,
        arm_gcc=arm_gcc,
        target_flags=daisy_target_flags(root),
        arm_nm=arm_nm,
        llvm_mca=mca,
        qemu_system_arm=qemu,
        glib_cflags=glib_cflags,
        glib_libs=glib_libs,
        qemu_include_dir=qemu_inc,
    )


# --- bootstrap: reproducible no-sudo toolchain acquisition --------------------

# xPack GNU Arm Embedded GCC (newlib 4.5.0 included). Version-pinned; the
# darwin-arm64 sha256 was verified against the per-file .sha asset published
# with the release. Other platforms verify against the fetched .sha asset
# (same release tag, HTTPS).
XPACK_VERSION = "15.2.1-1.1"
XPACK_BASE_URL = (
    "https://github.com/xpack-dev-tools/arm-none-eabi-gcc-xpack/releases/download/"
    f"v{XPACK_VERSION}/xpack-arm-none-eabi-gcc-{XPACK_VERSION}-"
)
# sha256 pinned per platform, fetched once from the published .sha assets and
# recorded here. The bootstrap NEVER fetches the expected hash at runtime: a
# server-supplied digest for a server-supplied tarball is self-certifying and
# provides zero protection against a compromised release asset (the exact
# supply-chain threat the hash check exists to defend). Unpinned platform =
# fail closed. To add one: fetch <url>.sha out of band and paste it here.
XPACK_SHA256 = {
    "darwin-arm64": "574082d35e49a2bcbdc355836b2a3ae5e5bb3b9456c9f5e37177db2ab4aad870",
    "darwin-x64": "5be906a5194c3173e145a58048e5607ffff947773237802b93e55e23c415f1b6",
    "linux-arm64": "67980c7990eba7bb7ffdf39699102effd70889f5ac427be19a8c8a6c5fab2972",
    "linux-x64": "da6a49ad4003944b823c6c93702a8787c922ab34bd7e918ec0eaf6933a9b1ff6",
}


def _xpack_platform() -> str:
    import platform as _platform

    system = {"Darwin": "darwin", "Linux": "linux"}.get(_platform.system())
    machine = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64"}.get(_platform.machine())
    if system is None or machine is None:
        raise ToolchainError(
            f"no xPack toolchain for {_platform.system()}/{_platform.machine()}; "
            "install a newlib-capable arm-none-eabi toolchain manually"
        )
    return f"{system}-{machine}"


def bootstrap(repo_root: Path | None = None, *, force: bool = False) -> Path:
    """Download + sha-verify + extract the pinned xPack toolchain into
    tools/toolchains/ (gitignored). Returns the extracted toolchain root."""
    import hashlib
    import tarfile
    import urllib.request

    root = repo_root or find_repo_root(Path(__file__))
    dest_dir = root / "tools" / "toolchains"
    extracted = dest_dir / f"xpack-arm-none-eabi-gcc-{XPACK_VERSION}"
    if extracted.is_dir() and not force:
        if _has_newlib(str(extracted / "bin" / ARM_GCC)):
            return extracted
        raise ToolchainError(f"{extracted} exists but looks broken; rerun with force")

    plat = _xpack_platform()
    expected = XPACK_SHA256.get(plat)
    if expected is None:
        # Fail closed: never trust a runtime-fetched digest for this platform.
        raise ToolchainError(
            f"no pinned sha256 for platform {plat!r}; refusing to bootstrap. "
            f"Fetch {XPACK_BASE_URL}{plat}.tar.gz.sha out of band, verify it, and add "
            "it to XPACK_SHA256 in toolchain.py."
        )

    url = f"{XPACK_BASE_URL}{plat}.tar.gz"
    dest_dir.mkdir(parents=True, exist_ok=True)
    tarball = dest_dir / f"xpack-{XPACK_VERSION}-{plat}.tar.gz"

    if not url.startswith("https://"):  # belt-and-suspenders: pinned constant is https
        raise ToolchainError(f"refusing non-https toolchain URL: {url}")
    try:
        urllib.request.urlretrieve(url, tarball)  # noqa: S310 — pinned https release URL
    except OSError as exc:
        tarball.unlink(missing_ok=True)   # no partial file a later run could trust
        raise ToolchainError(f"toolchain download failed: {exc}") from exc

    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    if digest != expected:
        tarball.unlink()
        raise ToolchainError(
            f"toolchain tarball sha256 mismatch (got {digest}, want {expected}) — refusing"
        )

    with tarfile.open(tarball) as tar:
        tar.extractall(dest_dir, filter="data")
    tarball.unlink()
    if not _has_newlib(str(extracted / "bin" / ARM_GCC)):
        raise ToolchainError(f"extracted toolchain at {extracted} failed the newlib probe")
    return extracted
