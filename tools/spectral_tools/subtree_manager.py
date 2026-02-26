"""Subtree management implementation."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import getpid
from pathlib import Path
from typing import Iterable

from .constants import short_ref
from .console import Console
from .process import run
from .subtree_models import (
    ALLOW_SUBTREE_SPLIT_FALLBACK,
    CLI_PROG,
    COUNT_FETCH_DEPTH,
    COUNT_FETCH_FILTER,
    DEFAULT_BRANCH,
    DEFAULT_PRINT_BRANCH,
    DRY_RUN_REF_PREFIX,
    HELP_ALIASES,
    LIBS_RELATIVE_PATH,
    LIST_ALIASES,
    MAX_PARALLEL_REMOTE_QUERIES,
    REMOVE_ALIAS,
    STATUS_ALIASES,
    SUBTREE_DIR_TRAILER,
    SUBTREE_HELP_ACTION,
    SUBTREE_LIST_ACTION,
    SUBTREE_PULL_ACTION,
    SUBTREE_PUSH_ACTION,
    SUBTREE_REMOVE_ACTION,
    SUBTREE_SPLIT_TRAILER,
    SUBTREE_STATUS_ACTION,
    UPDATE_PY_RELATIVE_PATH,
    DryRunStepReport,
    LibEntry,
    RunResult,
    StatusEntryReport,
    Summary,
    SyncProbe,
)
from .subtree_reporting import render_dry_run_report, render_status_report
from .utils import find_repo_root, read_utf8_lf

# Default timeout for all git subprocess calls (seconds).
# Network ops (ls-remote, fetch, push, pull) can hang on DNS/SSH failures;
# this caps the maximum wait.
GIT_DEFAULT_TIMEOUT_SEC = 120


class SubtreeManager:
    def __init__(self, repo_root: Path, libs_file: Path, ignored_paths: Iterable[Path]) -> None:
        self.repo_root = repo_root.resolve()
        self.libs_file = libs_file.resolve()
        self.console = Console()
        self._stash_ref = ""
        self._remote_head_cache: dict[tuple[str, str], tuple[str | None, str]] = {}
        self._local_split_cache: dict[str, tuple[str | None, str]] = {}
        self._count_fetch_cache: dict[tuple[str, str], tuple[str | None, str]] = {}

        self._ignored_rel: list[str] = []
        for path in ignored_paths:
            path = path.resolve()
            try:
                rel = str(path.relative_to(self.repo_root))
            except ValueError:
                continue
            self._ignored_rel.append(rel)

    def _find_stash_ref_by_marker(self, marker: str) -> str:
        code, out, _ = self._git(["stash", "list", "--format=%gd%x00%gs"])
        if code != 0 or not out:
            return ""
        for line in out.splitlines():
            if "\x00" not in line:
                continue
            ref, subject = line.split("\x00", 1)
            if marker in subject:
                return ref
        return ""

    @staticmethod
    def _trim(value: str) -> str:
        return value.strip()

    @staticmethod
    def _normalize(path: str) -> str:
        path = path.strip()
        while path.startswith("./"):
            path = path[2:]
        return path.rstrip("/")

    def _resolve_local_path(self, raw_path: str) -> tuple[str, Path]:
        norm = self._normalize(raw_path)
        if not norm or norm in {".", ".."}:
            raise ValueError("path cannot be empty, '.' or '..'")
        local = Path(norm)
        if local.is_absolute():
            raise ValueError("path must be relative to repository root")
        if ".." in local.parts:
            raise ValueError("path traversal '..' is not allowed")

        full = (self.repo_root / local).resolve()
        try:
            full.relative_to(self.repo_root)
        except ValueError as exc:
            raise ValueError("path resolves outside repository root") from exc
        return norm, full

    def die(self, message: str) -> None:
        print(f"Error: {message}", file=sys.stderr)
        raise SystemExit(1)

    def warn(self, message: str) -> None:
        print(f"Warning: {message}", file=sys.stderr)

    def _git(self, args: list[str], *, timeout_sec: int = GIT_DEFAULT_TIMEOUT_SEC) -> tuple[int, str, str]:
        result = run(
            ["git", "-C", str(self.repo_root), *args],
            cwd=self.repo_root,
            timeout_sec=timeout_sec,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr

    def _current_head(self) -> str:
        code, out, _ = self._git(["rev-parse", "--short=12", "HEAD"])
        if code != 0 or not out.strip():
            return "HEAD"
        return out.strip()

    @staticmethod
    def _dry_run_ref(step_index: int) -> str:
        return f"{DRY_RUN_REF_PREFIX}-{step_index:03d}"

    def _fetch_remote_head_uncached(self, repo: str, branch: str) -> tuple[str | None, str]:
        # Fast path: resolve remote branch tip without downloading history.
        code, out, err = self._git(["ls-remote", "--heads", "--exit-code", repo, branch])
        if code != 0:
            return None, f"remote head lookup failed: {err.strip() or 'unknown error'}"

        remote_head: str | None = None
        for line in out.splitlines():
            fields = line.split()
            if fields:
                remote_head = fields[0].strip()
                break
        if not remote_head:
            return None, "remote head lookup returned no commit"

        return remote_head, ""

    def _prefetch_remote_heads(self, keys: list[tuple[str, str]]) -> dict[tuple[str, str], tuple[str | None, str]]:
        unique_keys = sorted(set(keys))
        missing_keys = [key for key in unique_keys if key not in self._remote_head_cache]

        if missing_keys:
            worker_count = min(MAX_PARALLEL_REMOTE_QUERIES, len(missing_keys))
            # Thread-safety: workers do read-only ls-remote calls; cache writes happen only on main thread.
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                future_map = {
                    pool.submit(self._fetch_remote_head_uncached, repo, branch): (repo, branch)
                    for repo, branch in missing_keys
                }
                for future in as_completed(future_map):
                    key = future_map[future]
                    try:
                        self._remote_head_cache[key] = future.result()
                    except Exception as exc:
                        self._remote_head_cache[key] = (None, f"remote head lookup failed: {exc}")

        return {key: self._remote_head_cache.get(key, (None, "remote head lookup unavailable")) for key in unique_keys}

    def _commit_exists(self, commit: str) -> bool:
        if not commit:
            return False
        code, _, _ = self._git(["cat-file", "-e", f"{commit}^{{commit}}"])
        return code == 0

    def _is_ancestor(self, older: str, newer: str) -> bool | None:
        code, _, _ = self._git(["merge-base", "--is-ancestor", older, newer])
        if code == 0:
            return True
        if code == 1:
            return False
        return None

    def _fetch_remote_for_count(self, repo: str, branch: str) -> tuple[str | None, str]:
        key = (repo, branch)
        if key in self._count_fetch_cache:
            return self._count_fetch_cache[key]

        filtered_args = [
            "fetch",
            "--quiet",
            "--no-tags",
            f"--depth={COUNT_FETCH_DEPTH}",
            f"--filter={COUNT_FETCH_FILTER}",
            repo,
            branch,
        ]
        code, _, err = self._git(filtered_args)
        note = ""
        if code != 0:
            fallback_args = ["fetch", "--quiet", "--no-tags", f"--depth={COUNT_FETCH_DEPTH}", repo, branch]
            code, _, err = self._git(fallback_args)
            if code != 0:
                value = (None, f"shallow fetch failed: {err.strip() or 'unknown error'}")
                self._count_fetch_cache[key] = value
                return value
            note = "count fetch fallback used (remote does not support filter)"

        code, out, err = self._git(["rev-parse", "FETCH_HEAD"])
        if code != 0 or not out.strip():
            value = (None, f"unable to resolve FETCH_HEAD after shallow fetch: {err.strip() or 'unknown error'}")
            self._count_fetch_cache[key] = value
            return value

        value = (out.strip(), note)
        self._count_fetch_cache[key] = value
        return value

    def _local_subtree_split_from_log(self, path: str) -> str | None:
        code, out, _ = self._git(
            ["log", "-n", "1", "--format=%B", "--grep", f"{SUBTREE_DIR_TRAILER} {path}", "--fixed-strings"]
        )
        if code != 0 or not out.strip():
            return None

        for raw_line in out.splitlines():
            line = raw_line.strip()
            if line.startswith(SUBTREE_SPLIT_TRAILER):
                commit = line.split(":", 1)[1].strip()
                if commit:
                    return commit
        return None

    def _discover_local_split(self, path: str, *, present: bool) -> tuple[str | None, str]:
        if path in self._local_split_cache:
            return self._local_split_cache[path]
        if not present:
            value = (None, "local subtree is missing")
            self._local_split_cache[path] = value
            return value

        split_from_log = self._local_subtree_split_from_log(path)
        if split_from_log:
            value = (split_from_log, "local split resolved from subtree metadata")
            self._local_split_cache[path] = value
            return value

        if not ALLOW_SUBTREE_SPLIT_FALLBACK:
            value = (None, "local split trailer not found (slow subtree split fallback disabled)")
            self._local_split_cache[path] = value
            return value

        code, out, err = self._git(["subtree", "split", f"--prefix={path}", "HEAD"])
        if code != 0 or not out.strip():
            value = (None, f"unable to derive local split: {err.strip() or 'unknown error'}")
            self._local_split_cache[path] = value
            return value

        split = out.strip().splitlines()[-1].strip()
        value = (split if split else None, "local split resolved via git subtree split")
        self._local_split_cache[path] = value
        return value

    def _count_commits(self, older: str, newer: str) -> int | None:
        code, out, _ = self._git(["rev-list", "--count", f"{older}..{newer}"])
        if code != 0:
            return None
        try:
            return int(out.strip())
        except ValueError:
            return None

    def _probe_sync(
        self,
        *,
        path: str,
        repo: str,
        branch: str,
        present: bool,
        remote_lookup: tuple[str | None, str] | None = None,
    ) -> SyncProbe:
        local_split, local_note = self._discover_local_split(path, present=present)
        if remote_lookup is None:
            remote_head, remote_note = self._fetch_remote_head_uncached(repo, branch)
        else:
            remote_head, remote_note = remote_lookup

        pull_commits: int | None = None
        push_commits: int | None = None

        notes: list[str] = []
        if local_note:
            notes.append(local_note)
        if remote_note:
            notes.append(remote_note)

        if local_split and remote_head:
            compare_target = remote_head
            if not self._commit_exists(compare_target):
                fetched_head, fetch_note = self._fetch_remote_for_count(repo, branch)
                if fetch_note:
                    notes.append(fetch_note)
                if fetched_head:
                    compare_target = fetched_head
                    if fetched_head != remote_head:
                        notes.append(
                            f"remote moved during status check ({short_ref(remote_head)} -> {short_ref(fetched_head)})"
                        )

            if self._commit_exists(compare_target):
                local_is_ancestor = self._is_ancestor(local_split, compare_target)
                remote_is_ancestor = self._is_ancestor(compare_target, local_split)

                if local_is_ancestor is True and remote_is_ancestor is True:
                    pull_commits = 0
                    push_commits = 0
                elif local_is_ancestor is True and remote_is_ancestor is False:
                    pull_commits = self._count_commits(local_split, compare_target)
                    push_commits = 0
                elif local_is_ancestor is False and remote_is_ancestor is True:
                    pull_commits = 0
                    push_commits = self._count_commits(compare_target, local_split)
                elif local_is_ancestor is False and remote_is_ancestor is False:
                    notes.append("local and remote histories diverged (non fast-forward)")
                else:
                    pull_commits = self._count_commits(local_split, compare_target)
                    push_commits = self._count_commits(compare_target, local_split)
            else:
                notes.append(
                    f"remote commit {short_ref(remote_head)} not available locally after bounded fetch; "
                    "exact commit deltas unavailable"
                )

        if local_split and remote_head and pull_commits is None and "diverged" not in "; ".join(notes):
            notes.append("unable to compute pull commit delta")
        if local_split and remote_head and push_commits is None and "diverged" not in "; ".join(notes):
            notes.append("unable to compute push commit delta")

        return SyncProbe(
            local_split=local_split,
            remote_head=remote_head,
            pull_commits=pull_commits,
            push_commits=push_commits,
            note="; ".join(notes),
        )

    def require_git_repo(self) -> None:
        code, _, _ = self._git(["rev-parse", "--git-dir"])
        if code != 0:
            self.die(f"Not a git repository: {self.repo_root}")

    def _exclude_specs(self) -> list[str]:
        return [f":(exclude){rel}" for rel in self._ignored_rel]

    def has_changes(self) -> bool:
        excludes = self._exclude_specs()

        code, _, _ = self._git(["diff", "--cached", "--quiet", "--", ".", *excludes])
        if code != 0:
            return True

        code, _, _ = self._git(["diff", "--quiet", "--", ".", *excludes])
        if code != 0:
            return True

        code, out, _ = self._git(["ls-files", "--others", "--exclude-standard"])
        if code != 0:
            return True

        ignored = set(self._ignored_rel)
        for line in out.splitlines():
            line = line.strip()
            if line and line not in ignored:
                return True

        return False

    def require_clean_tree(self) -> None:
        if self.has_changes():
            self.die(
                "Working tree is not clean. Commit or stash your changes first.\n"
                "  git stash push --include-untracked"
            )

    def _collect_dirty_paths(self) -> list[str]:
        code, out, _ = self._git(["status", "--porcelain=v1", "-z"])
        if code != 0 or not out:
            return []

        dirty: list[str] = []
        records = out.split("\x00")
        idx = 0
        while idx < len(records):
            rec = records[idx]
            idx += 1
            if not rec:
                continue
            xy = rec[:2]
            path = rec[3:]
            if path:
                dirty.append(path)
            if xy[:1] in {"R", "C"} and idx < len(records):
                extra = records[idx]
                idx += 1
                if extra:
                    dirty.append(extra)
        return dirty

    def _prepare_subtree_clean_tree(self) -> None:
        dirty = self._collect_dirty_paths()
        if not dirty:
            return

        allowed = set(self._ignored_rel)
        for path in dirty:
            if path not in allowed:
                self.die(f"Working tree has modifications in '{path}'. Commit or stash before running subtree.")

        allowed_dirty = sorted({path for path in dirty if path in allowed})
        if not allowed_dirty:
            return

        message = f"subtree-temp-{getpid()}-{time.time_ns()}"
        args = ["stash", "push", "-u", "-m", message, "--", *allowed_dirty]
        code, _, _ = self._git(args)
        if code != 0:
            self.die("Unable to stash temporary subtree metadata changes.")

        self._stash_ref = self._find_stash_ref_by_marker(message)
        if not self._stash_ref:
            self.die("Unable to identify temporary stash entry. Restore manually via 'git stash list' and retry.")

    def _restore_subtree_clean_tree(self) -> bool:
        if not self._stash_ref:
            return True
        code, _, _ = self._git(["stash", "pop", "-q", self._stash_ref])
        self._stash_ref = ""
        if code != 0:
            self.warn("stash pop had conflicts; resolve manually.")
            return False
        return True

    def parse_lib_entry(self, line: str) -> LibEntry:
        parts = [self._trim(part) for part in line.split(",", 2)]
        repo = parts[0] if len(parts) >= 1 else ""
        local_path = parts[1] if len(parts) >= 2 else ""
        branch = parts[2] if len(parts) >= 3 else ""
        branch = branch or DEFAULT_BRANCH
        return LibEntry(raw=line, repo=repo, local_path=self._normalize(local_path), branch=branch)

    def read_libs_file(self) -> list[str]:
        if not self.libs_file.exists():
            self.die(f"libs.txt not found at {self.libs_file}")

        entries: list[str] = []
        for line in read_utf8_lf(self.libs_file).splitlines():
            line = self._trim(line)
            if not line or line.startswith("#"):
                continue
            entries.append(line)
        return entries

    def validate_entries(self, lines: list[str]) -> list[LibEntry]:
        entries: list[LibEntry] = []
        seen_paths: set[str] = set()
        errors = 0

        for line in lines:
            entry = self.parse_lib_entry(line)
            entries.append(entry)

            if not entry.repo:
                print(f"  {line}: missing repo URL", file=sys.stderr)
                errors += 1
                continue
            if not entry.local_path:
                print(f"  {line}: missing local path", file=sys.stderr)
                errors += 1
                continue
            try:
                self._resolve_local_path(entry.local_path)
            except ValueError as exc:
                print(f"  {line}: invalid local path ({exc})", file=sys.stderr)
                errors += 1
                continue

            if "://" not in entry.repo and "@" not in entry.repo:
                self.warn(f"'{entry.repo}' doesn't look like a URL (no :// or @)")

            if entry.local_path in seen_paths:
                self.warn(f"duplicate path '{entry.local_path}' in libs.txt")
            seen_paths.add(entry.local_path)

        if errors > 0:
            self.die(f"{errors} invalid entries in libs.txt")

        return entries

    def match_filter(self, target: str, filters: list[str]) -> bool:
        if not filters:
            return True
        target_norm = self._normalize(target)
        for flt in filters:
            flt_norm = self._normalize(flt)
            if not flt_norm:
                continue
            if target_norm == flt_norm or target_norm.startswith(f"{flt_norm}/"):
                return True
        return False

    def _run_subtree(self, args: list[str]) -> bool:
        code, out, err = self._git(["subtree", *args])
        if code == 0:
            return True

        print(f"\n{self.console.red('git subtree failed:')}", file=sys.stderr)
        merged = "\n".join(part for part in (out.strip(), err.strip()) if part)
        if merged:
            print(merged, file=sys.stderr)
        return False

    def print_summary(self, summary: Summary) -> None:
        print("")
        for path, result in zip(summary.paths, summary.results, strict=True):
            if result is RunResult.OK:
                view = self.console.green("ok")
            elif result is RunResult.FAILED:
                view = self.console.red("FAILED")
            else:
                view = self.console.yellow("skipped")
            print(f"  {path:<40} {view}")

        print("")
        print(
            f"{self.console.green(str(summary.ok_count))} succeeded, "
            f"{self.console.red(str(summary.fail_count))} failed, "
            f"{self.console.yellow(str(summary.skip_count))} skipped"
        )

    def do_pull(self, *, dry_run: bool, filters: list[str], print_branch: bool = DEFAULT_PRINT_BRANCH) -> int:
        lines = self.read_libs_file()
        if not lines:
            print("No entries in libs.txt")
            return 0

        entries = self.validate_entries(lines)
        summary = Summary()
        dry_steps: list[DryRunStepReport] = []
        head_start = self._current_head()
        projected_tree: list[str] = [head_start]
        projected_index = 0
        remote_lookup_map: dict[tuple[str, str], tuple[str | None, str]] = {}

        if dry_run:
            remote_keys: list[tuple[str, str]] = []
            for entry in entries:
                if not entry.repo or not entry.local_path:
                    continue
                if not self.match_filter(entry.local_path, filters):
                    continue
                _, full = self._resolve_local_path(entry.local_path)
                if full.exists() and not full.is_dir():
                    continue
                remote_keys.append((entry.repo, entry.branch))
            remote_lookup_map = self._prefetch_remote_heads(remote_keys)

        if not dry_run:
            self._prepare_subtree_clean_tree()

        restore_ok = True
        try:
            for entry in entries:
                if not entry.repo or not entry.local_path:
                    continue
                if not self.match_filter(entry.local_path, filters):
                    continue

                _, full = self._resolve_local_path(entry.local_path)
                if full.exists() and not full.is_dir():
                    if dry_run:
                        print(
                            f"  {self.console.yellow('[dry-run]')} "
                            f"{entry.local_path} exists but is not a directory, would fail"
                        )
                        dry_steps.append(
                            DryRunStepReport(
                                path=entry.local_path,
                                repo=entry.repo,
                                branch=entry.branch,
                                operation="pull",
                                status=RunResult.FAILED,
                                commit_before=projected_tree[-1],
                                commit_after=projected_tree[-1],
                                commit_message="",
                                projected_tree=tuple(projected_tree),
                                remote_head=None,
                                delta_commits=None,
                                note="local path exists but is not a directory",
                            )
                        )
                        summary.record_fail(entry.local_path)
                    else:
                        print(
                            f"  {self.console.red('[fail]')} "
                            f"{entry.local_path} exists but is not a directory"
                        )
                        summary.record_fail(entry.local_path)
                    continue

                if dry_run:
                    operation = "pull" if full.is_dir() else "add"
                    probe = self._probe_sync(
                        path=entry.local_path,
                        repo=entry.repo,
                        branch=entry.branch,
                        present=full.is_dir(),
                        remote_lookup=remote_lookup_map.get((entry.repo, entry.branch)),
                    )
                    delta_commits = probe.pull_commits
                    projected_before = projected_tree[-1]
                    projected_after = projected_before
                    commit_message = ""
                    if operation in {"pull", "add"}:
                        projected_index += 1
                        projected_after = self._dry_run_ref(projected_index)
                        projected_tree.append(projected_after)
                        commit_message = (
                            f"Update subtree {entry.local_path}"
                            if operation == "pull"
                            else f"Add subtree {entry.local_path}"
                        )

                    if full.is_dir():
                        print(
                            f"  {self.console.yellow('[dry-run]')} would pull "
                            f"{self.console.bold(entry.local_path)} from {entry.repo} ({entry.branch}) "
                            f"-> {projected_after} "
                            f"(remote={short_ref(probe.remote_head)} "
                            f"ff={delta_commits if delta_commits is not None else 'unknown'})"
                        )
                    else:
                        print(
                            f"  {self.console.yellow('[dry-run]')} would add "
                            f"{self.console.bold(entry.local_path)} from {entry.repo} ({entry.branch}) "
                            f"-> {projected_after} "
                            f"(remote={short_ref(probe.remote_head)})"
                        )
                    dry_steps.append(
                        DryRunStepReport(
                            path=entry.local_path,
                            repo=entry.repo,
                            branch=entry.branch,
                            operation=operation,
                            status=RunResult.OK,
                            commit_before=projected_before,
                            commit_after=projected_after,
                            commit_message=commit_message,
                            projected_tree=tuple(projected_tree),
                            remote_head=probe.remote_head,
                            delta_commits=delta_commits,
                            note=probe.note,
                        )
                    )
                    summary.record_ok(entry.local_path)
                    continue

                if full.is_dir():
                    print(f"  pull  {self.console.bold(entry.local_path)} <- {entry.branch} ... ", end="", flush=True)
                    ok = self._run_subtree(
                        [
                            "pull",
                            f"--prefix={entry.local_path}",
                            entry.repo,
                            entry.branch,
                            "--squash",
                            "-m",
                            f"Update subtree {entry.local_path}",
                        ]
                    )
                else:
                    print(f"  add   {self.console.bold(entry.local_path)} <- {entry.branch} ... ", end="", flush=True)
                    ok = self._run_subtree(
                        [
                            "add",
                            f"--prefix={entry.local_path}",
                            entry.repo,
                            entry.branch,
                            "--squash",
                        ]
                    )

                if ok:
                    print(self.console.green("ok"))
                    summary.record_ok(entry.local_path)
                else:
                    print(self.console.red("FAILED"))
                    summary.record_fail(entry.local_path)
        finally:
            if not dry_run:
                restore_ok = self._restore_subtree_clean_tree()

        if dry_run and print_branch:
            print("")
            print(
                render_dry_run_report(
                    action=SUBTREE_PULL_ACTION,
                    filters=filters,
                    head_start=head_start,
                    steps=dry_steps,
                )
            )
        if not dry_run:
            self.print_summary(summary)

        return 1 if summary.fail_count > 0 or not restore_ok else 0

    def do_push(self, *, dry_run: bool, filters: list[str], print_branch: bool = DEFAULT_PRINT_BRANCH) -> int:
        lines = self.read_libs_file()
        if not lines:
            print("No entries in libs.txt")
            return 0

        entries = self.validate_entries(lines)
        summary = Summary()
        dry_steps: list[DryRunStepReport] = []
        head_start = self._current_head()
        projected_tree: list[str] = [head_start]
        remote_lookup_map: dict[tuple[str, str], tuple[str | None, str]] = {}

        if dry_run:
            remote_keys: list[tuple[str, str]] = []
            for entry in entries:
                if not entry.repo or not entry.local_path:
                    continue
                if not self.match_filter(entry.local_path, filters):
                    continue
                _, full = self._resolve_local_path(entry.local_path)
                if full.is_dir():
                    remote_keys.append((entry.repo, entry.branch))
            remote_lookup_map = self._prefetch_remote_heads(remote_keys)

        if not dry_run:
            self._prepare_subtree_clean_tree()

        restore_ok = True
        try:
            for entry in entries:
                if not entry.repo or not entry.local_path:
                    continue
                if not self.match_filter(entry.local_path, filters):
                    continue

                _, full = self._resolve_local_path(entry.local_path)
                if full.exists() and not full.is_dir():
                    if dry_run:
                        print(
                            f"  {self.console.yellow('[dry-run]')} "
                            f"{entry.local_path} exists but is not a directory, would fail"
                        )
                        dry_steps.append(
                            DryRunStepReport(
                                path=entry.local_path,
                                repo=entry.repo,
                                branch=entry.branch,
                                operation="push",
                                status=RunResult.FAILED,
                                commit_before=projected_tree[-1],
                                commit_after=projected_tree[-1],
                                commit_message="",
                                projected_tree=tuple(projected_tree),
                                remote_head=None,
                                delta_commits=None,
                                note="local path exists but is not a directory",
                            )
                        )
                        summary.record_fail(entry.local_path)
                    else:
                        print(
                            f"  push  {entry.local_path} ... "
                            f"{self.console.red('FAILED')} (exists but is not a directory)"
                        )
                        summary.record_fail(entry.local_path)
                    continue

                if dry_run:
                    if not full.is_dir():
                        print(
                            f"  {self.console.yellow('[dry-run]')} {entry.local_path} does not exist, would skip"
                        )
                        dry_steps.append(
                            DryRunStepReport(
                                path=entry.local_path,
                                repo=entry.repo,
                                branch=entry.branch,
                                operation="push",
                                status=RunResult.SKIPPED,
                                commit_before=projected_tree[-1],
                                commit_after=projected_tree[-1],
                                commit_message="",
                                projected_tree=tuple(projected_tree),
                                remote_head=None,
                                delta_commits=None,
                                note="local subtree is not present",
                            )
                        )
                        summary.record_skip(entry.local_path)
                        continue

                    probe = self._probe_sync(
                        path=entry.local_path,
                        repo=entry.repo,
                        branch=entry.branch,
                        present=True,
                        remote_lookup=remote_lookup_map.get((entry.repo, entry.branch)),
                    )
                    delta_commits = probe.push_commits
                    if full.is_dir():
                        print(
                            f"  {self.console.yellow('[dry-run]')} would push "
                            f"{self.console.bold(entry.local_path)} to {entry.repo} ({entry.branch}) "
                            f"(remote={short_ref(probe.remote_head)} "
                            f"delta={delta_commits if delta_commits is not None else 'unknown'})"
                        )
                    dry_steps.append(
                        DryRunStepReport(
                            path=entry.local_path,
                            repo=entry.repo,
                            branch=entry.branch,
                            operation="push",
                            status=RunResult.OK,
                            commit_before=projected_tree[-1],
                            commit_after=projected_tree[-1],
                            commit_message="no local commit (push updates remote only)",
                            projected_tree=tuple(projected_tree),
                            remote_head=probe.remote_head,
                            delta_commits=delta_commits,
                            note=probe.note,
                        )
                    )
                    summary.record_ok(entry.local_path)
                    continue

                if not full.is_dir():
                    print(f"  push  {entry.local_path} ... {self.console.yellow('skipped')} (not present)")
                    summary.record_skip(entry.local_path)
                    continue

                print(f"  push  {self.console.bold(entry.local_path)} -> {entry.branch} ... ", end="", flush=True)
                ok = self._run_subtree(["push", f"--prefix={entry.local_path}", entry.repo, entry.branch])
                if ok:
                    print(self.console.green("ok"))
                    summary.record_ok(entry.local_path)
                else:
                    print(self.console.red("FAILED"))
                    summary.record_fail(entry.local_path)
        finally:
            if not dry_run:
                restore_ok = self._restore_subtree_clean_tree()

        if dry_run and print_branch:
            print("")
            print(
                render_dry_run_report(
                    action=SUBTREE_PUSH_ACTION,
                    filters=filters,
                    head_start=head_start,
                    steps=dry_steps,
                )
            )
        if not dry_run:
            self.print_summary(summary)

        return 1 if summary.fail_count > 0 or not restore_ok else 0

    def do_remove(self, paths: list[str]) -> int:
        if not paths:
            self.die("remove requires at least one path")

        self._prepare_subtree_clean_tree()
        summary = Summary()

        restore_ok = True
        try:
            for raw_path in paths:
                try:
                    path, full = self._resolve_local_path(raw_path)
                except ValueError as exc:
                    print(f"  remove  {raw_path} ... {self.console.red('FAILED')} ({exc})")
                    summary.record_fail(raw_path)
                    continue

                if full.is_symlink():
                    print(f"  remove  {path} ... {self.console.red('FAILED')} (symlink path not supported)")
                    summary.record_fail(path)
                    continue
                if not full.is_dir():
                    print(f"  remove  {path} ... {self.console.yellow('skipped')} (not found)")
                    summary.record_skip(path)
                    continue

                print(f"  remove  {self.console.bold(path)} ... ", end="", flush=True)
                shutil.rmtree(full)
                self._git(["add", "-A", path])

                diff_code, _, _ = self._git(["diff", "--cached", "--quiet", "--", path])
                if diff_code == 0:
                    print(f"{self.console.yellow('skipped')} (already clean)")
                    summary.record_skip(path)
                    continue

                commit_code, _, _ = self._git(["commit", "-m", f"Remove subtree {path}"])
                if commit_code == 0:
                    print(self.console.green("ok"))
                    summary.record_ok(path)
                else:
                    print(self.console.red("FAILED"))
                    summary.record_fail(path)
        finally:
            restore_ok = self._restore_subtree_clean_tree()

        self.print_summary(summary)
        return 1 if summary.fail_count > 0 or not restore_ok else 0

    def do_list(self) -> int:
        if not self.libs_file.exists():
            print(f"No libs.txt at {self.libs_file}")
            return 0

        lines = self.read_libs_file()
        if not lines:
            print("No entries in libs.txt")
            return 0

        entries = self.validate_entries(lines)
        for entry in entries:
            _, full = self._resolve_local_path(entry.local_path)
            if full.is_dir():
                status = self.console.green("[present]")
            else:
                status = self.console.red("[missing]")
            print(f"  {entry.local_path:<40} {status}  {entry.repo} ({entry.branch})")

        return 0

    def do_status(self, *, filters: list[str], print_branch: bool = DEFAULT_PRINT_BRANCH) -> int:
        lines = self.read_libs_file()
        if not lines:
            print("No entries in libs.txt")
            return 0

        entries = self.validate_entries(lines)
        rows: list[StatusEntryReport] = []
        summary = Summary()
        remote_keys: list[tuple[str, str]] = []
        for entry in entries:
            if not entry.repo or not entry.local_path:
                continue
            if not self.match_filter(entry.local_path, filters):
                continue
            remote_keys.append((entry.repo, entry.branch))
        remote_lookup_map = self._prefetch_remote_heads(remote_keys)

        for entry in entries:
            if not entry.repo or not entry.local_path:
                continue
            if not self.match_filter(entry.local_path, filters):
                continue

            _, full = self._resolve_local_path(entry.local_path)
            present = full.is_dir()
            probe = self._probe_sync(
                path=entry.local_path,
                repo=entry.repo,
                branch=entry.branch,
                present=present,
                remote_lookup=remote_lookup_map.get((entry.repo, entry.branch)),
            )

            status = RunResult.OK
            note = probe.note
            ff_count = probe.pull_commits
            if not present:
                status = RunResult.SKIPPED
                note = (f"{note}; " if note else "") + "local subtree is missing (would add on pull)"
            elif not probe.remote_head:
                status = RunResult.FAILED
                note = (f"{note}; " if note else "") + "unable to resolve remote head"
            elif not probe.local_split:
                status = RunResult.SKIPPED
                note = (f"{note}; " if note else "") + "unable to resolve local split commit"
            elif ff_count is None:
                status = RunResult.FAILED
                note = (f"{note}; " if note else "") + "unable to compute fast-forward commit count"

            row = StatusEntryReport(
                path=entry.local_path,
                repo=entry.repo,
                branch=entry.branch,
                status=status,
                local_split=probe.local_split,
                remote_head=probe.remote_head,
                fast_forward_commits=ff_count,
                note=note,
            )
            rows.append(row)

            ff_text = str(ff_count) if ff_count is not None else "unknown"
            print(
                f"  status {entry.local_path:<34} "
                f"local={short_ref(probe.local_split)} "
                f"-> remote={short_ref(probe.remote_head)} "
                f"ff={ff_text}"
            )

            if status is RunResult.OK:
                summary.record_ok(entry.local_path)
            elif status is RunResult.FAILED:
                summary.record_fail(entry.local_path)
            else:
                summary.record_skip(entry.local_path)

        if print_branch:
            print("")
            print(render_status_report(filters=filters, rows=rows))

        print("")
        print(
            f"{self.console.green(str(summary.ok_count))} ok, "
            f"{self.console.red(str(summary.fail_count))} failed, "
            f"{self.console.yellow(str(summary.skip_count))} skipped"
        )
        return 1 if summary.fail_count > 0 else 0

    def show_help(self) -> int:
        print(self.console.bold("Subtree management"))
        print("")
        print("Usage:")
        print(f"  {CLI_PROG} pull [--dry-run|-n] [-P] [<path>...]   Pull (or add) subtrees")
        print(f"  {CLI_PROG} push [--dry-run|-n] [-P] [<path>...]   Push subtrees upstream")
        print(f"  {CLI_PROG} remove <path>...                  Remove and commit")
        print(f"  {CLI_PROG} status [-P] [<path>...]           Show fast-forward target and commit delta")
        print(f"  {CLI_PROG} list|ls                           Show configured subtrees")
        print(f"  {CLI_PROG} help                              Show this help")
        print("")
        print("libs.txt format:  repo_url, local_path, branch")
        print("Branch defaults to main if omitted.")
        return 0


def parse_cli(argv: list[str]) -> argparse.Namespace:
    args = list(argv)
    if not args:
        action = SUBTREE_PULL_ACTION
        rest: list[str] = []
    elif args[0] in HELP_ALIASES:
        return argparse.Namespace(action=SUBTREE_HELP_ACTION)
    elif args[0] in {
        SUBTREE_PULL_ACTION,
        SUBTREE_PUSH_ACTION,
        SUBTREE_REMOVE_ACTION,
        REMOVE_ALIAS,
        SUBTREE_STATUS_ACTION,
        *LIST_ALIASES,
        *STATUS_ALIASES,
    }:
        action = args[0]
        rest = args[1:]
    elif args[0].startswith("-"):
        # Shorthand: "subtree_update.py -n" means "subtree_update.py pull -n".
        action = SUBTREE_PULL_ACTION
        rest = args
    else:
        raise ValueError(args[0])

    if action in {SUBTREE_PULL_ACTION, SUBTREE_PUSH_ACTION}:
        sub = argparse.ArgumentParser(prog=f"{CLI_PROG} {action}")
        sub.add_argument("-n", "--dry-run", action="store_true")
        sub.add_argument("-P", "--print-branch", action="store_true", default=DEFAULT_PRINT_BRANCH)
        sub.add_argument("paths", nargs="*")
        parsed = sub.parse_args(rest)
        return argparse.Namespace(action=action, dry_run=parsed.dry_run, print_branch=parsed.print_branch, paths=parsed.paths)

    if action in {SUBTREE_REMOVE_ACTION, REMOVE_ALIAS}:
        sub = argparse.ArgumentParser(prog=f"{CLI_PROG} remove")
        sub.add_argument("paths", nargs="*")
        parsed = sub.parse_args(rest)
        if not parsed.paths:
            sub.error("the following arguments are required: paths")
        return argparse.Namespace(action="remove", paths=parsed.paths)

    if action in LIST_ALIASES:
        sub = argparse.ArgumentParser(prog=f"{CLI_PROG} list")
        sub.parse_args(rest)
        return argparse.Namespace(action=SUBTREE_LIST_ACTION)

    if action in {SUBTREE_STATUS_ACTION, *STATUS_ALIASES}:
        sub = argparse.ArgumentParser(prog=f"{CLI_PROG} status")
        sub.add_argument("-P", "--print-branch", action="store_true", default=DEFAULT_PRINT_BRANCH)
        sub.add_argument("paths", nargs="*")
        parsed = sub.parse_args(rest)
        return argparse.Namespace(action=SUBTREE_STATUS_ACTION, print_branch=parsed.print_branch, paths=parsed.paths)

    raise ValueError(action)


def main(argv: list[str] | None = None) -> int:
    root = find_repo_root(Path(__file__))
    manager = SubtreeManager(
        repo_root=root,
        libs_file=root / LIBS_RELATIVE_PATH,
        ignored_paths=[root / LIBS_RELATIVE_PATH, root / UPDATE_PY_RELATIVE_PATH],
    )

    manager.require_git_repo()

    args_raw = argv if argv is not None else sys.argv[1:]
    try:
        args = parse_cli(args_raw)
    except ValueError as exc:
        manager.die(f"Unknown action: {exc.args[0]} (use: pull, push, remove, status, list, help)")
        return 1

    if args.action == SUBTREE_HELP_ACTION:
        return manager.show_help()
    if args.action == SUBTREE_PULL_ACTION:
        return manager.do_pull(
            dry_run=bool(args.dry_run),
            filters=list(args.paths),
            print_branch=bool(getattr(args, "print_branch", DEFAULT_PRINT_BRANCH)),
        )
    if args.action == SUBTREE_PUSH_ACTION:
        return manager.do_push(
            dry_run=bool(args.dry_run),
            filters=list(args.paths),
            print_branch=bool(getattr(args, "print_branch", DEFAULT_PRINT_BRANCH)),
        )
    if args.action == SUBTREE_REMOVE_ACTION:
        return manager.do_remove(list(args.paths))
    if args.action == SUBTREE_LIST_ACTION:
        return manager.do_list()
    if args.action == SUBTREE_STATUS_ACTION:
        return manager.do_status(
            filters=list(getattr(args, "paths", [])),
            print_branch=bool(getattr(args, "print_branch", DEFAULT_PRINT_BRANCH)),
        )

    manager.die(f"Unknown action: {args.action} (use: pull, push, remove, status, list, help)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
