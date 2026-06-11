"""ctypes bridge to the C path canonicalization and FNV-1a hashing in
spectral_resource_fs.c.

This module loads ``libspectral_resource_bridge`` at import time.  If the
shared library is not found the import **raises** ``RuntimeError`` — the
calling tool must build the CMake target ``spectral_resource_bridge`` first.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

from ..core.utils import find_repo_root

# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

_LIB_BASE_NAME = "spectral_resource_bridge"
_CANONICAL_PATH_SIZE = 1024  # Must match SPECTRAL_CANONICAL_PATH_SIZE in the header.


def _find_library() -> Path:
    """Search well-known build output locations relative to the repository root."""
    repo_root = find_repo_root(Path(__file__))

    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    candidates = [
        repo_root / "build" / f"lib{_LIB_BASE_NAME}{suffix}",
        repo_root / "build" / "spectral_engine" / f"lib{_LIB_BASE_NAME}{suffix}",
        repo_root / "build" / "lib" / f"lib{_LIB_BASE_NAME}{suffix}",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeError(
        f"Native bridge library lib{_LIB_BASE_NAME}{suffix} not found.\n"
        f"You must build it before running any Python tool that depends on it:\n"
        f"  cmake --build build --target {_LIB_BASE_NAME}\n"
        f"Searched:\n" + "\n".join(f"  {p}" for p in candidates)
    )


def _load_library() -> ctypes.CDLL:
    lib_path = _find_library()
    lib = ctypes.CDLL(str(lib_path))

    # uint32_t spectral_resource_file_id_from_path(const char* path);
    lib.spectral_resource_file_id_from_path.argtypes = [ctypes.c_char_p]
    lib.spectral_resource_file_id_from_path.restype = ctypes.c_uint32

    # size_t spectral_resource_path_canonical(
    #     const char* path, char* out, size_t out_size);
    lib.spectral_resource_path_canonical.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.spectral_resource_path_canonical.restype = ctypes.c_size_t

    return lib


_lib = _load_library()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def file_id_from_path(raw_path: bytes) -> int:
    """Compute FNV-1a 32-bit file ID for a resource path.

    The C function ``spectral_resource_file_id_from_path`` canonicalizes its
    input before hashing, so pass the RAW relative path — exactly what the
    embedded runtime passes. Canonicalization is NOT idempotent (phase 1 strips
    the \\x01 RLE sentinel phase 5 emits), so passing an already-canonical path
    double-canonicalizes and diverges from the runtime id on repeated-byte runs.
    """
    return _lib.spectral_resource_file_id_from_path(raw_path)


def compress_path(relative_posix_path: str) -> bytes:
    """Canonicalize a resource-relative POSIX path through all five C phases.

    Returns the canonical byte string (identical to the old Python
    ``compress_path`` output).

    Raises ``RuntimeError`` if the canonical form exceeds the fixed
    buffer size (SPECTRAL_CANONICAL_PATH_SIZE).
    """
    raw = relative_posix_path.encode("utf-8")
    buf = ctypes.create_string_buffer(_CANONICAL_PATH_SIZE)
    length = _lib.spectral_resource_path_canonical(raw, buf, _CANONICAL_PATH_SIZE)
    if length >= _CANONICAL_PATH_SIZE:
        raise RuntimeError(
            f"Canonical path exceeds buffer size ({length} >= {_CANONICAL_PATH_SIZE}): "
            f"{relative_posix_path!r}"
        )
    return buf.raw[:length]
