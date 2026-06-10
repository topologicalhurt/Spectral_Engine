"""Typed loader/serializer for the third-party dependency manifest.

``third_party/libs.yaml`` is the SSOT for every vendored dependency. Schema
(version 1):

    version: 1
    dependencies:
      simde:
        kind: submodule            # submodule | subtree   (required)
        url: https://github.com/simd-everywhere/simde      (required)
        path: third_party/simde    # repo-relative          (required)
        ref: master                # branch (subtree) / branch to track (submodule)
        sync: true                 # false = bypassed by bulk sync/pull ops
        sparse: [simde]            # submodule-only: sparse-checkout cones
        track: both                # subtree-only: both|pull|push|none
        notes: free text

Validation is strict: unknown keys, kind-mismatched keys (``track`` on a
submodule, ``sparse`` on a subtree), duplicate paths, and paths escaping the
repo all raise ``ManifestError``. Serialization is deterministic (sorted
keys, stable field order) so manifest diffs stay reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MANIFEST_RELPATH = Path("third_party/libs.yaml")
MANIFEST_VERSION = 1

KIND_SUBMODULE = "submodule"
KIND_SUBTREE = "subtree"
VALID_KINDS = (KIND_SUBMODULE, KIND_SUBTREE)

TRACK_VALUES = ("both", "pull", "push", "none")
DEFAULT_REF = "main"

_COMMON_KEYS = {"kind", "url", "path", "ref", "sync", "notes"}
_KIND_KEYS = {
    KIND_SUBMODULE: _COMMON_KEYS | {"sparse"},
    KIND_SUBTREE: _COMMON_KEYS | {"track"},
}
_REQUIRED_KEYS = {"kind", "url", "path"}


class ManifestError(ValueError):
    """Manifest content violates the schema; message names the entry."""


@dataclass(frozen=True, slots=True)
class DependencySpec:
    name: str
    kind: str
    url: str
    path: str                       # repo-relative, normalized (no ./, no trailing /)
    ref: str = DEFAULT_REF
    sync: bool = True               # False = bypassed by bulk operations
    sparse: tuple[str, ...] = field(default_factory=tuple)   # submodule-only
    track: str = "both"             # subtree-only
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "url": self.url,
            "path": self.path,
            "ref": self.ref,
        }
        if self.kind == KIND_SUBMODULE and not self.sync:
            data["sync"] = False
        if self.kind == KIND_SUBMODULE and self.sparse:
            data["sparse"] = list(self.sparse)
        if self.kind == KIND_SUBTREE and self.track != "both":
            data["track"] = self.track
        if self.notes:
            data["notes"] = self.notes
        return data


@dataclass(frozen=True, slots=True)
class Manifest:
    path: Path
    entries: tuple[DependencySpec, ...]

    def of_kind(self, kind: str) -> tuple[DependencySpec, ...]:
        return tuple(entry for entry in self.entries if entry.kind == kind)

    def submodules(self) -> tuple[DependencySpec, ...]:
        return self.of_kind(KIND_SUBMODULE)

    def subtrees(self) -> tuple[DependencySpec, ...]:
        return self.of_kind(KIND_SUBTREE)

    def by_path(self, path: str) -> DependencySpec | None:
        normalized = _normalize_path(path)
        for entry in self.entries:
            if entry.path == normalized:
                return entry
        return None


def _normalize_path(path: str) -> str:
    path = path.strip()
    while path.startswith("./"):
        path = path[2:]
    return path.rstrip("/")


def _validate_path(name: str, path: str) -> str:
    normalized = _normalize_path(path)
    if not normalized:
        raise ManifestError(f"{name}: empty path")
    parts = Path(normalized).parts
    if Path(normalized).is_absolute() or ".." in parts:
        raise ManifestError(f"{name}: path must be repo-relative without '..' (got {path!r})")
    return normalized


def _parse_entry(name: str, raw: Any) -> DependencySpec:
    if not isinstance(raw, dict):
        raise ManifestError(f"{name}: entry must be a mapping, got {type(raw).__name__}")
    kind = raw.get("kind")
    if kind not in VALID_KINDS:
        raise ManifestError(f"{name}: kind must be one of {VALID_KINDS}, got {kind!r}")

    allowed = _KIND_KEYS[kind]
    unknown = set(raw) - allowed
    if unknown:
        raise ManifestError(
            f"{name}: unknown key(s) {sorted(unknown)} for kind={kind} "
            f"(allowed: {sorted(allowed)})"
        )
    missing = _REQUIRED_KEYS - set(raw)
    if missing:
        raise ManifestError(f"{name}: missing required key(s) {sorted(missing)}")

    url = raw["url"]
    if not isinstance(url, str) or ("://" not in url and "@" not in url):
        raise ManifestError(f"{name}: url does not look like a git URL: {url!r}")

    sync = raw.get("sync", True)
    if not isinstance(sync, bool):
        raise ManifestError(f"{name}: sync must be a boolean, got {sync!r}")
    if kind == KIND_SUBTREE and sync is False:
        raise ManifestError(
            f"{name}: subtree entries express bypass via 'track: none' "
            "(engine-native), not 'sync: false' — one bypass mechanism per kind"
        )

    ref = raw.get("ref", DEFAULT_REF)
    if not isinstance(ref, str) or not ref.strip():
        raise ManifestError(f"{name}: ref must be a non-empty string")

    sparse_raw = raw.get("sparse", [])
    if not isinstance(sparse_raw, list) or not all(isinstance(s, str) for s in sparse_raw):
        raise ManifestError(f"{name}: sparse must be a list of path strings")

    track = raw.get("track", "both")
    if track not in TRACK_VALUES:
        raise ManifestError(f"{name}: track must be one of {TRACK_VALUES}, got {track!r}")

    return DependencySpec(
        name=name,
        kind=kind,
        url=url,
        path=_validate_path(name, str(raw["path"])),
        ref=ref.strip(),
        sync=sync,
        sparse=tuple(sparse_raw),
        track=track,
        notes=str(raw.get("notes", "")),
    )


def load(repo_root: Path, *, allow_missing: bool = False) -> Manifest:
    manifest_path = repo_root / MANIFEST_RELPATH
    if not manifest_path.is_file():
        if allow_missing:
            return Manifest(path=manifest_path, entries=())
        raise ManifestError(f"manifest not found: {manifest_path}")

    try:
        doc = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError(f"{MANIFEST_RELPATH}: invalid YAML: {exc}") from exc
    if doc is None:
        return Manifest(path=manifest_path, entries=())
    if not isinstance(doc, dict):
        raise ManifestError(f"{MANIFEST_RELPATH}: top level must be a mapping")

    version = doc.get("version")
    if version != MANIFEST_VERSION:
        raise ManifestError(
            f"{MANIFEST_RELPATH}: unsupported manifest version {version!r} "
            f"(this tool reads version {MANIFEST_VERSION})"
        )
    deps = doc.get("dependencies", {})
    if not isinstance(deps, dict):
        raise ManifestError(f"{MANIFEST_RELPATH}: dependencies must be a mapping of name -> entry")
    unknown_top = set(doc) - {"version", "dependencies"}
    if unknown_top:
        raise ManifestError(f"{MANIFEST_RELPATH}: unknown top-level key(s) {sorted(unknown_top)}")

    entries: list[DependencySpec] = []
    seen_paths: dict[str, str] = {}
    for name in sorted(deps):
        entry = _parse_entry(str(name), deps[name])
        if entry.path in seen_paths:
            raise ManifestError(
                f"{entry.name}: duplicate path {entry.path!r} (also used by {seen_paths[entry.path]})"
            )
        seen_paths[entry.path] = entry.name
        entries.append(entry)

    return Manifest(path=manifest_path, entries=tuple(entries))


def save(repo_root: Path, entries: tuple[DependencySpec, ...] | list[DependencySpec]) -> Path:
    """Deterministic serialization: names sorted, stable field order."""
    manifest_path = repo_root / MANIFEST_RELPATH
    doc = {
        "version": MANIFEST_VERSION,
        "dependencies": {entry.name: entry.as_dict() for entry in sorted(entries, key=lambda e: e.name)},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "# Third-party dependency manifest — managed by `python -m spectral_tools.vendor`\n"
        "# Schema + rules: tools/spectral_tools/vendor/manifest.py (ADR-0003)\n"
        + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return manifest_path


def replace_entries(
    repo_root: Path, *, kind: str, new_entries: list[DependencySpec]
) -> Path:
    """Replace all entries of ``kind`` while preserving the other kind —
    the write path the subtree engine uses so it can never clobber
    submodule declarations (and vice versa)."""
    if kind not in VALID_KINDS:
        raise ManifestError(f"unknown kind {kind!r}")
    for entry in new_entries:
        if entry.kind != kind:
            raise ManifestError(
                f"{entry.name}: kind={entry.kind} passed to replace_entries(kind={kind})"
            )
    current = load(repo_root, allow_missing=True)
    kept = [entry for entry in current.entries if entry.kind != kind]
    return save(repo_root, [*kept, *new_entries])
