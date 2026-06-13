"""Include-direction law for the kernel/arch/drivers layout (KERNEL_LAYOUT_PLAN).

Parses every engine TU/header's #include directives, resolves in-repo
headers by basename, and fails on any edge the layer law forbids:

  kernel   (core/ + runtime/)  -> kernel only, EXCEPT the dispatch-registry
                                  allowlist below (the static vtable must
                                  name the driver symbols).
  arch/X   -> kernel + arch/X       (one ISA never includes another)
  drivers/X-> kernel + drivers/X    (one driver never includes another)
  analysis -> kernel + analysis
  cmd      -> unconstrained         (userland, top of the DAG)
  api/     -> kernel + arch + api   (board support binds an arch)

This is a dependency-DAG contract on real include edges, not source-text
prose. Third-party and system includes are ignored. Header basenames must
be unique across the engine — resolution relies on it (and ambiguity is
its own smell), so a duplicate basename fails too.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "spectral_engine"
API = ROOT / "api"

INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)

# (including file relative to repo root, included basename) pairs that are
# deliberate, documented exceptions to the law.
ALLOWLIST = {
    # The backend dispatch registry: the ONE core->driver edge. The static
    # vtable table must name driver symbols; everything else uses the vtable.
    ("spectral_engine/core/spectral_backend.c", "spectral_metal.h"),
    ("spectral_engine/core/spectral_backend.c", "spectral_cuda.h"),
}

SOURCE_SUFFIXES = {".c", ".h", ".m", ".cu", ".inc"}


def _layer(path: Path) -> str:
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if parts[0] == "api":
        return "api"
    assert parts[0] == "spectral_engine", rel
    top = parts[1]
    if top in ("core", "runtime"):
        return "kernel"
    if top in ("arch", "drivers"):
        return f"{top}/{parts[2]}"   # arch/arm, drivers/metal, ...
    return top                       # analysis, cmd


def _allowed(src_layer: str, dst_layer: str) -> bool:
    if src_layer == dst_layer or dst_layer == "kernel":
        return True
    if src_layer == "cmd":
        return True
    if src_layer == "api":
        return dst_layer.startswith("arch/") or dst_layer == "api"
    return False


def _engine_files() -> list[Path]:
    files = [p for base in (ENGINE, API) for p in base.rglob("*")
             if p.suffix in SOURCE_SUFFIXES and p.is_file()]
    assert files, "engine tree not found"
    return files


def _header_owners(files: list[Path]) -> dict[str, Path]:
    owners: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        if f.suffix in (".h", ".inc"):
            owners[f.name].append(f)
    dupes = {n: ps for n, ps in owners.items() if len(ps) > 1}
    assert not dupes, f"duplicate header basenames break resolution: {dupes}"
    return {n: ps[0] for n, ps in owners.items()}


def test_include_direction_is_law():
    files = _engine_files()
    owners = _header_owners(files)
    violations = []
    for f in files:
        src_layer = _layer(f)
        rel = str(f.relative_to(ROOT))
        for name in INCLUDE_RE.findall(f.read_text(encoding="utf-8", errors="replace")):
            base = name.rsplit("/", 1)[-1]
            target = owners.get(base)
            if target is None:
                continue   # third-party / system / generated-out-of-tree
            dst_layer = _layer(target)
            if (rel, base) in ALLOWLIST:
                continue
            if not _allowed(src_layer, dst_layer):
                violations.append(f"{rel} [{src_layer}] -> {base} [{dst_layer}]")
    assert not violations, (
        "include-direction violations (fix the edge or amend the law + "
        "allowlist DELIBERATELY):\n  " + "\n  ".join(sorted(violations)))
