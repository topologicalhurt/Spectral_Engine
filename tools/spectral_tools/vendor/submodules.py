"""Git-submodule management over the dependency manifest (ADR-0003).

State is derived from git (`.gitmodules`, `git submodule status`, the
worktree) and compared against the manifest — never duplicated. Operations:

- ``status``  — per-entry presence/init/sha/drift, plus manifest⇄git checks.
- ``verify``  — strict bidirectional consistency (the kind-separation gate:
  a git submodule with no ``kind: submodule`` manifest entry is an error,
  as is a manifest subtree path that git knows as a submodule).
- ``sync``    — make git match the manifest: add missing submodules,
  init/update present ones, apply sparse-checkout cones. Entries with
  ``sync: false`` are bypassed unless explicitly selected by path.
- ``remove``  — deinit + git rm + manifest update.

Every operation supports ``dry_run`` and returns house-format report rows.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.process import run
from . import manifest as manifest_mod
from .manifest import KIND_SUBMODULE, DependencySpec, Manifest, ManifestError


class SubmoduleError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GitSubmoduleState:
    """One row of `git submodule status` + .gitmodules config."""
    path: str
    sha: str | None
    initialized: bool
    url: str | None
    branch: str | None


def _git(repo_root: Path, args: list[str], *, check: bool = True) -> str:
    result = run(["git", *args], cwd=repo_root, check=False)
    if check and result.returncode != 0:
        raise SubmoduleError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout


def read_gitmodules(repo_root: Path) -> dict[str, dict[str, str]]:
    """Parse .gitmodules into {path: {url, branch, name}} — git's own record."""
    gitmodules = repo_root / ".gitmodules"
    if not gitmodules.is_file():
        return {}
    parser = configparser.ConfigParser()
    parser.read_string(gitmodules.read_text(encoding="utf-8"))
    by_path: dict[str, dict[str, str]] = {}
    for section in parser.sections():
        if not section.startswith("submodule "):
            continue
        name = section.split(" ", 1)[1].strip('"')
        data = dict(parser[section])
        path = data.get("path")
        if path:
            by_path[path] = {**data, "name": name}
    return by_path


def git_submodule_states(repo_root: Path) -> dict[str, GitSubmoduleState]:
    """`git submodule status` joined with .gitmodules, keyed by path.

    Status line prefix: '-' = not initialized, '+' = checked-out sha differs
    from the recorded gitlink, 'U' = merge conflicts, ' ' = clean.
    """
    by_path = read_gitmodules(repo_root)
    states: dict[str, GitSubmoduleState] = {}
    out = _git(repo_root, ["submodule", "status"], check=False)
    for line in out.splitlines():
        if not line.strip():
            continue
        prefix, rest = line[0], line[1:]
        parts = rest.split()
        if len(parts) < 2:
            continue
        sha, path = parts[0], parts[1]
        config = by_path.get(path, {})
        states[path] = GitSubmoduleState(
            path=path,
            sha=None if prefix == "-" else sha,
            initialized=prefix != "-",
            url=config.get("url"),
            branch=config.get("branch"),
        )
    # .gitmodules entries that `submodule status` did not list (e.g. gitlink
    # removed but config left behind) still need surfacing.
    for path, config in by_path.items():
        states.setdefault(
            path,
            GitSubmoduleState(path=path, sha=None, initialized=False,
                              url=config.get("url"), branch=config.get("branch")),
        )
    return states


def _entry_row(entry: DependencySpec, state: GitSubmoduleState | None) -> dict[str, Any]:
    issues: list[str] = []
    if state is None:
        issues.append("declared in manifest but absent from .gitmodules (run sync)")
    else:
        if not state.initialized:
            issues.append("not initialized (run sync)")
        if state.url and state.url != entry.url:
            issues.append(f"url drift: .gitmodules={state.url} manifest={entry.url}")
        if state.branch and state.branch != entry.ref:
            issues.append(f"branch drift: .gitmodules={state.branch} manifest={entry.ref}")
    return {
        "name": entry.name,
        "path": entry.path,
        "kind": entry.kind,
        "ref": entry.ref,
        "sync": entry.sync,
        "sha": state.sha if state else None,
        "initialized": bool(state and state.initialized),
        "issues": issues,
    }


def status(repo_root: Path, mani: Manifest | None = None) -> list[dict[str, Any]]:
    mani = mani or manifest_mod.load(repo_root)
    states = git_submodule_states(repo_root)
    rows = [_entry_row(entry, states.get(entry.path)) for entry in mani.submodules()]
    declared_paths = {entry.path for entry in mani.submodules()}
    for path, state in sorted(states.items()):
        if path not in declared_paths:
            rows.append({
                "name": f"<undeclared:{path}>",
                "path": path,
                "kind": KIND_SUBMODULE,
                "ref": state.branch,
                "sync": None,
                "sha": state.sha,
                "initialized": state.initialized,
                "issues": ["git submodule with no manifest entry — declare it or remove it"],
            })
    return rows


def verify(repo_root: Path, mani: Manifest | None = None) -> list[str]:
    """Strict bidirectional manifest⇄git consistency. Returns problems
    (empty = consistent). This is the kind-separation gate."""
    mani = mani or manifest_mod.load(repo_root)
    states = git_submodule_states(repo_root)
    problems: list[str] = []

    for row in status(repo_root, mani):
        problems.extend(f"{row['path']}: {issue}" for issue in row["issues"])

    # A manifest SUBTREE path must never be a git submodule (and a subtree
    # path that git tracks as a gitlink means the kinds have been mixed).
    for entry in mani.subtrees():
        if entry.path in states:
            problems.append(
                f"{entry.path}: declared kind=subtree but git tracks it as a "
                "submodule — kinds must not mix"
            )
    return problems


def _apply_sparse(repo_root: Path, entry: DependencySpec, *, dry_run: bool) -> list[str]:
    if not entry.sparse:
        return []
    cmd = ["git", "-C", entry.path, "sparse-checkout", "set", "--cone", *entry.sparse]
    if dry_run:
        return [f"[dry-run] {' '.join(cmd)}"]
    _git(repo_root, cmd[1:])
    return [f"sparse-checkout: {', '.join(entry.sparse)}"]


def sync(
    repo_root: Path,
    *,
    mani: Manifest | None = None,
    only_paths: list[str] | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Make git match the manifest. ``sync: false`` entries are bypassed in
    bulk runs but are synced when named explicitly in ``only_paths``."""
    mani = mani or manifest_mod.load(repo_root)
    states = git_submodule_states(repo_root)
    selected = list(mani.submodules())
    if only_paths:
        wanted = {manifest_mod._normalize_path(p) for p in only_paths}
        selected = [e for e in selected if e.path in wanted]
        missing = wanted - {e.path for e in selected}
        if missing:
            raise SubmoduleError(f"path(s) not in manifest as submodules: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for entry in selected:
        actions: list[str] = []
        if not entry.sync and not only_paths:
            rows.append({"name": entry.name, "path": entry.path,
                         "result": "bypassed (sync: false)", "actions": []})
            continue

        state = states.get(entry.path)
        if state is None:
            cmd = ["submodule", "add", "-b", entry.ref, entry.url, entry.path]
            if dry_run:
                actions.append(f"[dry-run] git {' '.join(cmd)}")
            else:
                _git(repo_root, cmd)
                actions.append(f"added submodule @ {entry.ref}")
        elif not state.initialized:
            cmd = ["submodule", "update", "--init", "--", entry.path]
            if dry_run:
                actions.append(f"[dry-run] git {' '.join(cmd)}")
            else:
                _git(repo_root, cmd)
                actions.append("initialized")
        else:
            actions.append("present")
        actions.extend(_apply_sparse(repo_root, entry, dry_run=dry_run))
        rows.append({"name": entry.name, "path": entry.path, "result": "ok", "actions": actions})
    return rows


def remove(repo_root: Path, path: str, *, dry_run: bool = False) -> list[str]:
    """Deinit + git rm + manifest update for one submodule path."""
    mani = manifest_mod.load(repo_root)
    entry = mani.by_path(path)
    if entry is None or entry.kind != KIND_SUBMODULE:
        raise SubmoduleError(f"{path}: not a manifest submodule entry")
    steps = [
        ["git", "submodule", "deinit", "-f", "--", entry.path],
        ["git", "rm", "-f", "--", entry.path],
    ]
    log: list[str] = []
    if dry_run:
        log.extend(f"[dry-run] {' '.join(step)}" for step in steps)
        log.append(f"[dry-run] drop {entry.name} from {manifest_mod.MANIFEST_RELPATH}")
        log.append(f"[dry-run] note: .git/modules/<name> is left for git to gc")
        return log
    for step in steps:
        _git(repo_root, step[1:])
        log.append(" ".join(step))
    remaining = [e for e in mani.entries if e is not entry]
    manifest_mod.save(repo_root, remaining)
    log.append(f"dropped {entry.name} from {manifest_mod.MANIFEST_RELPATH}")
    return log


def add(
    repo_root: Path,
    *,
    name: str,
    url: str,
    path: str,
    ref: str,
    sparse: tuple[str, ...] = (),
    dry_run: bool = False,
) -> DependencySpec:
    """Declare + materialize a new submodule (manifest first, then git)."""
    mani = manifest_mod.load(repo_root, allow_missing=True)
    entry = DependencySpec(name=name, kind=KIND_SUBMODULE, url=url,
                           path=manifest_mod._normalize_path(path), ref=ref, sparse=sparse)
    if mani.by_path(entry.path) is not None:
        raise SubmoduleError(f"{entry.path}: already declared in the manifest")
    if any(e.name == name for e in mani.entries):
        raise SubmoduleError(f"{name}: name already declared in the manifest")
    if not dry_run:
        manifest_mod.save(repo_root, [*mani.entries, entry])
        sync(repo_root, only_paths=[entry.path])
    return entry
