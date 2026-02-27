"""Pure-Python reference implementations of the C path canonicalization and
FNV-1a hashing used by the resource hash table.

These are **not** used in production — the native ctypes bridge
(``generators.native_bridge``) calls the authoritative C implementation.

The reference implementations are preserved here for parity testing:
comparing the Python output against the C output guarantees that the two
are byte-identical for any input.
"""

from __future__ import annotations

from itertools import groupby

# ---------------------------------------------------------------------------
# Constants — must match spectral_resource_fs.h / .c
# ---------------------------------------------------------------------------

FNV1A_32_OFFSET_BASIS = 2166136261
FNV1A_32_PRIME = 16777619
FNV1A_32_MASK = 0xFFFFFFFF
FNV1A_32_MODULUS = 1 << 32
FNV1A_32_PRIME_INVERSE = pow(FNV1A_32_PRIME, -1, FNV1A_32_MODULUS)

UTF8_ENCODING = "utf-8"
SPECTRAL_CANONICAL_PATH_SIZE = 1024
SPECTRAL_CANONICAL_MAX_BYTES = SPECTRAL_CANONICAL_PATH_SIZE - 1
SPECTRAL_CANONICAL_MAX_COMPONENTS = 256

RLE_SENTINEL_BYTE = 0x01
MAX_RLE_CHUNK = 255
BYTE_SLASH = ord("/")
BYTE_BACKSLASH = ord("\\")
BYTE_SPACE = ord(" ")
BYTE_DOT = ord(".")
HEX_DIGITS = b"0123456789abcdef"


# ---------------------------------------------------------------------------
# FNV-1a
# ---------------------------------------------------------------------------

def file_id_from_path(canonical_path: bytes) -> int:
    """Compute FNV-1a (32-bit) over a canonical path string.

    Callers must pass the result of compress_path(), not the raw relative path,
    so that the ID matches what the C runtime computes via
    spectral_resource_file_id_from_path() in spectral_resource_fs.c.
    """
    digest = FNV1A_32_OFFSET_BASIS
    for byte in canonical_path:
        digest ^= byte
        digest = (digest * FNV1A_32_PRIME) & FNV1A_32_MASK
    return digest


def decode_fnv1a32_offset_basis(canonical_path: bytes, digest: int) -> int:
    """Reverse FNV-1a over known bytes to recover the initial offset basis."""
    value = digest & FNV1A_32_MASK
    for byte in reversed(canonical_path):
        value = (value * FNV1A_32_PRIME_INVERSE) & FNV1A_32_MASK
        value ^= byte
    return value


# ---------------------------------------------------------------------------
# Phase 1–3: lowercase, strip controls, normalize separators
# ---------------------------------------------------------------------------

def _ascii_lower_byte(value: int) -> int:
    """Lowercase ASCII byte values only (matches C canonicalization)."""
    if 0x41 <= value <= 0x5A:
        return value + (0x61 - 0x41)
    return value


def _phase1to3_normalize_path_bytes(relative_path: str) -> bytearray:
    """Phase 1-3: UTF-8 bytes, control-byte strip, ASCII-lower, slash normalize.

    This mirrors spectral_path_p1() in C:
    - encode path to bytes
    - strip control bytes: 0x00-0x1F and 0x7F
    - lowercase ASCII only
    - normalize '\\\\' to '/'
    - truncate at SPECTRAL_CANONICAL_MAX_BYTES
    """
    scratch = bytearray()
    for raw_byte in relative_path.encode(UTF8_ENCODING):
        if raw_byte <= 0x1F or raw_byte == 0x7F:
            continue
        normalized = _ascii_lower_byte(raw_byte)
        if normalized == BYTE_BACKSLASH:
            normalized = BYTE_SLASH
        if len(scratch) >= SPECTRAL_CANONICAL_MAX_BYTES:
            break
        scratch.append(normalized)
    return scratch


# ---------------------------------------------------------------------------
# Phase 4: component-wise resolution
# ---------------------------------------------------------------------------

def _find_component_end(scratch: bytearray, start_idx: int, scratch_len: int) -> int:
    end_idx = start_idx
    while end_idx < scratch_len and scratch[end_idx] != BYTE_SLASH:
        end_idx += 1
    return end_idx


def _trim_trailing_byte(scratch: bytearray, start_idx: int, end_idx: int, byte_value: int) -> int:
    trimmed_end = end_idx
    while trimmed_end > start_idx and scratch[trimmed_end - 1] == byte_value:
        trimmed_end -= 1
    return trimmed_end


def _is_single_dot_component(scratch: bytearray, start_idx: int, end_idx: int) -> bool:
    return end_idx - start_idx == 1 and scratch[start_idx] == BYTE_DOT


def _is_parent_component(scratch: bytearray, start_idx: int, end_idx: int) -> bool:
    return (
        end_idx - start_idx == 2
        and scratch[start_idx] == BYTE_DOT
        and scratch[start_idx + 1] == BYTE_DOT
    )


def _pop_last_component(resolved: bytearray, component_starts: list[int]) -> None:
    if not component_starts:
        return
    start = component_starts.pop()
    del resolved[start:]


def _append_component(
    resolved: bytearray,
    component_starts: list[int],
    scratch: bytearray,
    start_idx: int,
    end_idx: int,
) -> None:
    if end_idx <= start_idx:
        return
    if len(component_starts) >= SPECTRAL_CANONICAL_MAX_COMPONENTS:
        return

    component_starts.append(len(resolved))

    if len(component_starts) > 1 and len(resolved) < SPECTRAL_CANONICAL_MAX_BYTES:
        resolved.append(BYTE_SLASH)

    room = SPECTRAL_CANONICAL_MAX_BYTES - len(resolved)
    if room <= 0:
        return

    resolved.extend(scratch[start_idx : min(end_idx, start_idx + room)])


def _phase4_resolve_components(scratch: bytearray) -> bytearray:
    """Phase 4: component-wise resolution with NTFS compatibility quirks.

    Rules (must match C):
    - trim trailing spaces before ".." detection
    - '..' pops the last accepted component
    - skip empty and '.'
    - trim trailing dots
    - cap component count and output length
    """
    resolved = bytearray()
    component_starts: list[int] = []
    scratch_len = len(scratch)
    idx = 0

    while idx <= scratch_len:
        end_idx = _find_component_end(scratch, idx, scratch_len)
        end_space = _trim_trailing_byte(scratch, idx, end_idx, BYTE_SPACE)

        if end_space == idx or _is_single_dot_component(scratch, idx, end_space):
            idx = end_idx + 1
            continue

        if _is_parent_component(scratch, idx, end_space):
            _pop_last_component(resolved, component_starts)
            idx = end_idx + 1
            continue

        end_dot = _trim_trailing_byte(scratch, idx, end_space, BYTE_DOT)
        _append_component(resolved, component_starts, scratch, idx, end_dot)
        idx = end_idx + 1

    return resolved


# ---------------------------------------------------------------------------
# Phase 5: generalized RLE
# ---------------------------------------------------------------------------

def _append_rle_token(out: bytearray, value: int, chunk: int) -> bool:
    """Append token bytes as 0x01 <byte> <hex_hi> <hex_lo>."""
    if len(out) + 4 > SPECTRAL_CANONICAL_MAX_BYTES:
        return False
    out.extend(
        (
            RLE_SENTINEL_BYTE,
            value,
            HEX_DIGITS[(chunk >> 4) & 0x0F],
            HEX_DIGITS[chunk & 0x0F],
        )
    )
    return True


def _append_literal_byte(out: bytearray, value: int) -> bool:
    if len(out) + 1 > SPECTRAL_CANONICAL_MAX_BYTES:
        return False
    out.append(value)
    return True


def _phase5_generalized_rle(resolved: bytearray) -> bytes:
    """Phase 5: generalized RLE with \\x01 sentinel and lowercase hex nibbles."""
    out = bytearray()
    for current, grouped in groupby(resolved):
        run = sum(1 for _ in grouped)
        while run > 0:
            chunk = MAX_RLE_CHUNK if run > MAX_RLE_CHUNK else run
            if chunk >= 2:
                if not _append_rle_token(out, current, chunk):
                    break
            elif not _append_literal_byte(out, current):
                break
            run -= chunk

    return bytes(out)


# ---------------------------------------------------------------------------
# Public API — mirrors resource_hashes.compress_path / file_id_from_path
# ---------------------------------------------------------------------------

def compress_path(relative_posix_path: str) -> bytes:
    """Canonicalize a resource-relative POSIX path; pure-Python reference.

    This must produce byte-identical output to the C implementation in
    ``spectral_resource_path_canonical()``.
    """
    scratch = _phase1to3_normalize_path_bytes(relative_posix_path)
    resolved = _phase4_resolve_components(scratch)
    return _phase5_generalized_rle(resolved)
