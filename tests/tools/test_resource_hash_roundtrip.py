"""Resource file-id round-trip: the committed embedded table must agree with
the C runtime, and the path canonicalization that underlies both must be
handled correctly (it is NOT idempotent).

Regression for a confirmed defect: the generator computed embedded file_ids as
FNV(canon(canon(path))) by passing already-compressed bytes into
spectral_resource_file_id_from_path (which canonicalizes again), while the
embedded runtime computes FNV(canon(raw_path)). Canon is not idempotent —
phase 1 strips the \\x01 RLE sentinel phase 5 emits — so the two diverged for
any path with a repeated-byte run (e.g. the committed testing/sin_440hz.wav),
and spectral_resource_fs_find_by_id(file_id_from_path(path)) silently returned
NULL.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

TABLE = ROOT / "spectral_engine/core/spectral_hash_resources_xx32_xx3.c"

# Both blocks of the generated table, correlated by the shared size field.
_EMBEDDED_RE = re.compile(
    r"\{\s*UINT32_C\((0x[0-9a-fA-F]+)\),\s*UINT64_C\(0x[0-9a-fA-F]+\),\s*\(size_t\)(\d+)u\s*\}"
)
_HOST_RE = re.compile(
    r'\{\s*"([^"]+)",\s*UINT64_C\(0x[0-9a-fA-F]+\),\s*\(size_t\)(\d+)u\s*\}'
)


def _bridge():
    try:
        from spectral_tools.generators import native_bridge
    except RuntimeError as exc:  # bridge dylib not built
        pytest.skip(f"native resource bridge unavailable: {exc}")
    return native_bridge


def _committed_table() -> list[tuple[str, int]]:
    """Return [(relative_path, embedded_file_id)] correlated by size."""
    text = TABLE.read_text(encoding="utf-8")
    by_size_id = {int(size): int(fid, 16) for fid, size in _EMBEDDED_RE.findall(text)}
    by_size_path = {int(size): path for path, size in _HOST_RE.findall(text)}
    assert by_size_id and by_size_path, "failed to parse the generated resource table"
    assert by_size_id.keys() == by_size_path.keys(), "embedded/host table size sets diverge"
    return [(by_size_path[s], by_size_id[s]) for s in sorted(by_size_id)]


def test_committed_file_ids_match_runtime():
    """Each committed embedded file_id == what the C runtime computes from the
    RAW path. This is exactly the find_by_id(file_id_from_path(path)) contract."""
    nb = _bridge()
    for rel_path, committed_id in _committed_table():
        runtime_id = nb.file_id_from_path(rel_path.encode("utf-8"))
        assert runtime_id == committed_id, (
            f"{rel_path}: committed table file_id {committed_id:#010x} != "
            f"runtime spectral_resource_file_id_from_path {runtime_id:#010x} "
            "(double-canonicalization regression?)"
        )


def test_canonicalization_is_not_idempotent_root_cause():
    """The root-cause guard: re-canonicalizing a canonical path changes its id
    whenever an RLE run was emitted. If this ever becomes equal, canon has been
    made idempotent and the 'pass the raw path' rule in the generator can be
    revisited — until then, passing canonical bytes double-canons and breaks
    the round-trip above."""
    nb = _bridge()
    from spectral_tools.generators.resource_hashes import compress_path

    # A path with a repeated-byte run ('44') -> canon emits the \x01 sentinel.
    resources = ROOT / "resources"
    sample = resources / "testing/sin_440hz.wav"
    if not sample.exists():
        pytest.skip("sample resource not present")
    canonical = compress_path(sample, resources)
    raw_id = nb.file_id_from_path("testing/sin_440hz.wav".encode("utf-8"))
    double_id = nb.file_id_from_path(canonical)  # canon(canonical) — the old bug
    assert raw_id != double_id, (
        "canon appears idempotent for a repeated-run path; the double-canon bug "
        "would no longer be observable and this guard needs revisiting"
    )
