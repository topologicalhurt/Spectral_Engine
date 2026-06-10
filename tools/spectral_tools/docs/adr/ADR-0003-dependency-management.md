# ADR-0003: Vendored-Dependency Management (Manifest, Submodules, Subtrees)

- Status: Accepted
- Date: 2026-06-10
- Decision makers: maintainer (topologicalhurt)

## Context

`third_party/` migrated from git subtrees to submodules (simde, xxHash); the
old `libs.txt` (CSV-ish lines) had no notion of dependency kind — its 4th
field collided with the subtree engine's track-mask parsing the moment
`submodule` appeared in it — and there was no submodule tooling at all. The
subtree engine (`spectral_tools.subtrees`, ~2k lines incl. dry-run DAG
projection and parallel remote probing) predates the migration and was bound
to the txt format.

## Decisions

### 1. One YAML manifest, explicit kinds

`third_party/libs.yaml` is the SSOT for every vendored dependency.
Each entry declares `kind: submodule | subtree` plus `url`, `path`, `ref`,
and kind-specific fields. Validation is strict (unknown keys, kind-mismatched
fields, duplicate paths, escaping paths all raise); serialization is
deterministic. Loader/serializer: `spectral_tools.vendor.manifest`.

### 2. Strict kind separation, one front door

- `spectral_tools.vendor.submodules` manages `kind: submodule` (status,
  verify, sync, add, remove). All state is derived from git
  (`.gitmodules`, `git submodule status`) and *compared* to the manifest —
  never duplicated (C-truth rule, ADR-0002 §4).
- The existing subtree engine keeps `kind: subtree`: its manifest I/O
  boundary now reads/writes the YAML through
  `manifest.replace_entries(kind=subtree)`, which structurally cannot
  clobber submodule declarations (and vice versa).
- `verify` is the separation gate: a git submodule with no manifest entry,
  or a manifest subtree path that git tracks as a gitlink, is an error.
- CLI front door: `python -m spectral_tools.vendor`
  (`list`, `submodule <op>`, `subtree <op>` — the latter forwards verbatim
  to the subtree engine).

### 3. Bypass semantics — one mechanism per kind

"Don't touch this dependency in bulk operations" is expressed as:

- submodules: `sync: false` (skipped by `submodule sync` unless the path is
  named explicitly);
- subtrees: the engine-native `track: none|pull|push` mask.

The validator rejects `sync: false` on a subtree entry so the two mechanisms
can never coexist on one entry. `sparse:` (submodule-only) declares
sparse-checkout cones for pulling only the needed subdirectories of a large
upstream (repo-size control).

### 4. Subtree engine: kept, boundary-hardened

The engine's internals (split/probe/dry-run machinery) are retained; the
rigor pass was applied at its boundaries (manifest I/O, kind separation,
message accuracy). Deeper internal hardening of the 800-line manager is
tracked work, not silently claimed.

## Consequences

- Adding a dependency is one manifest entry + `vendor submodule sync` (or a
  subtree `add`); CI/tests can `verify` consistency cheaply.
- The two vendoring strategies cannot corrupt each other's declarations by
  construction, and mixing kinds at one path is detected.
- `pyyaml` joins the pinned build-venv requirements.
