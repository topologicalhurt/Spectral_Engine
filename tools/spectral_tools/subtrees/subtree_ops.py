"""Protected subtree operation base used by SubtreeManager."""

from __future__ import annotations

import sys
import time
import threading
import uuid

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

from ..core.utils import short_ref
from ..core.console import ThreadProgressDisplay
from ..core.git_ops import GitSubtreeOps
from ..core.process import run
from .subtree_models import (
    ALLOW_SUBTREE_SPLIT_FALLBACK,
    DEFAULT_BRANCH,
    CountFetchCache,
    COUNT_FETCH_DEPTH,
    COUNT_FETCH_FILTER,
    DryRunStepReport,
    LibEntry,
    LocalSplitCache,
    LOOKUP_SHARE_DEFAULT,
    OperationState,
    LOOKUP_SHARE_KIB_PER_UNIT,
    LOOKUP_SHARE_TIMEOUT_SEC,
    MAX_PARALLEL_REMOTE_QUERIES,
    RefLookupResult,
    RemoteHeadCache,
    RepoBranchKey,
    RunResult,
    Summary,
    SUBTREE_PULL_ACTION,
    SUBTREE_PUSH_ACTION,
    SUBTREE_DIR_TRAILER,
    SUBTREE_SPLIT_TRAILER,
    SyncProbe,
    TRACK_MASK_BOTH,
    TRACK_MASK_VALUES,
    TRACK_UPDATE_MASK_VALUES,
    apply_track_mask,
    dry_run_ref,
    entry_allows,
    entry_to_line,
    normalize_path,
    normalize_track_mask,
)
from .subtree_reporting import render_dry_run_report
from ..core.utils import read_utf8_lf, write_utf8_lf



class SubtreeOpsBase(GitSubtreeOps):
    """Git/subtree adapter base class for manager classes.

    Exposes protected methods so higher-level managers can call/override them
    without binding directly to raw git client method names.
    """

    def __init__(self, repo_root: Path, *, ignored_paths: Iterable[Path] = ()) -> None:
        super().__init__(repo_root)
        self._stash_ref = ""
        self._cache_lock = threading.Lock()
        self._remote_head_cache: RemoteHeadCache = {}
        self._local_split_cache: LocalSplitCache = {}
        self._count_fetch_cache: CountFetchCache = {}
        self._lookup_share_cache: dict[str, float] = {}
        self._ignored_rel = [
            rel
            for path in (item.resolve() for item in ignored_paths)
            if (rel := self._safe_relative_to_root(path))
        ]

    def _safe_relative_to_root(self, path: Path) -> str | None:
        """Return a repo-relative path string, or None when path is outside repo root."""
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            return None

    def _resolve_local_path(self, raw_path: str) -> tuple[str, Path]:
        norm = normalize_path(raw_path)
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

    def _write_entries(self, entries: list[LibEntry]) -> None:
        self.libs_file.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(entry_to_line(entry) for entry in entries).strip()
        write_utf8_lf(self.libs_file, f"{content}\n" if content else "")

    def parse_lib_entry(self, line: str) -> LibEntry:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            raise ValueError(f"too few fields in libs entry (need ≥ 2, got {len(parts)}): {line!r}")
        if len(parts) > 4:
            raise ValueError("expected format: repo_url, local_path, branch[, track_mask]")
        repo = parts[0] if len(parts) >= 1 else ""
        local_path = parts[1] if len(parts) >= 2 else ""
        branch = parts[2] if len(parts) >= 3 and parts[2] else DEFAULT_BRANCH
        track_mask = normalize_track_mask(parts[3] if len(parts) >= 4 else TRACK_MASK_BOTH)
        entry_kwargs = {
            "raw": line,
            "repo": repo,
            "local_path": normalize_path(local_path),
            "branch": branch,
            "track_mask": track_mask,
        }
        return LibEntry(**entry_kwargs)

    def read_libs_file(self, *, allow_missing: bool = False) -> list[str]:
        if not self.libs_file.exists():
            if allow_missing:
                return []
            self.die(f"libs.txt not found at {self.libs_file}")
        return [
            line
            for raw in read_utf8_lf(self.libs_file).splitlines()
            if (line := raw.strip()) and not line.startswith("#")
        ]

    def validate_entries(self, lines: list[str]) -> list[LibEntry]:
        entries: list[LibEntry] = []
        seen_paths: set[str] = set()
        errors = 0

        for line in lines:
            try:
                entry = self.parse_lib_entry(line)
            except ValueError as exc:
                self.console.print_error(f"{line}: {exc}")
                errors += 1
                continue
            entries.append(entry)

            if not entry.repo:
                self.console.print_error(f"{line}: missing repo URL")
                errors += 1
                continue
            if not entry.local_path:
                self.console.print_error(f"{line}: missing local path")
                errors += 1
                continue
            try:
                self._resolve_local_path(entry.local_path)
            except ValueError as exc:
                self.console.print_error(f"{line}: invalid local path ({exc})")
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
        target_norm = normalize_path(target)
        for flt in filters:
            flt_norm = normalize_path(flt)
            if not flt_norm:
                continue
            if target_norm == flt_norm or target_norm.startswith(f"{flt_norm}/"):
                return True
        return False

    def _iter_selected_entries(self, entries: list[LibEntry], filters: list[str]) -> Iterable[tuple[LibEntry, Path]]:
        for entry in entries:
            if not entry.repo or not entry.local_path:
                continue
            if not self.match_filter(entry.local_path, filters):
                continue
            _, full = self._resolve_local_path(entry.local_path)
            yield entry, full

    @staticmethod
    def _append_dry_step(
        *,
        dry_steps: list[DryRunStepReport],
        entry: LibEntry,
        operation: str,
        status: RunResult,
        projected_tree: list[str],
        note: str,
        local_split: str | None = None,
        remote_head: str | None = None,
        delta_commits: int | None = None,
        commit_before: str | None = None,
        commit_after: str | None = None,
        commit_message: str = "",
        squash_commit: str | None = None,
        commit_count: int | None = None,
        projected_graph: tuple[str, ...] | None = None,
    ) -> None:
        before = projected_tree[-1] if commit_before is None else commit_before
        after = before if commit_after is None else commit_after
        count = max(0, commit_count) if commit_count is not None else (0 if before == after else 1)
        if projected_graph is None:
            graph = (f"local DAG unchanged at {before}",) if count == 0 else (f"first-parent: {before} -> {after}",)
        else:
            graph = projected_graph
        step_kwargs = {
            "path": entry.local_path,
            "repo": entry.repo,
            "branch": entry.branch,
            "operation": operation,
            "status": status,
            "commit_before": before,
            "commit_after": after,
            "commit_message": commit_message,
            "projected_tree": tuple(projected_tree),
            "local_split": local_split,
            "remote_head": remote_head,
            "delta_commits": delta_commits,
            "note": note,
            "squash_commit": squash_commit,
            "commit_count": count,
            "projected_graph": graph,
        }
        dry_steps.append(DryRunStepReport(**step_kwargs))

    def _record_non_directory_failure(self, *, entry: LibEntry, operation: str, state: OperationState) -> None:
        if state.dry_run:
            print(f"  {self.console.yellow('[dry-run]')} {entry.local_path} exists but is not a directory, would fail")
            step_kwargs = {
                "dry_steps": state.dry_steps,
                "entry": entry,
                "operation": operation,
                "status": RunResult.FAILED,
                "projected_tree": state.projected_tree,
                "note": "local path exists but is not a directory",
            }
            self._append_dry_step(**step_kwargs)
        else:
            pull_msg = f"  {self.console.red('[fail]')} {entry.local_path} exists but is not a directory"
            push_msg = f"  push  {entry.local_path} ... {self.console.red('FAILED')} (exists but is not a directory)"
            print(pull_msg if operation == "pull" else push_msg)
        state.summary.record_fail(entry.local_path)

    def _record_skip(
        self,
        *,
        entry: LibEntry,
        operation: str,
        note: str,
        state: OperationState,
    ) -> None:
        if state.dry_run:
            step_kwargs = {
                "dry_steps": state.dry_steps,
                "entry": entry,
                "operation": operation,
                "status": RunResult.SKIPPED,
                "projected_tree": state.projected_tree,
                "note": note,
            }
            self._append_dry_step(**step_kwargs)
        state.summary.record_skip(entry.local_path)

    @staticmethod
    def _append_note(note: str, suffix: str) -> str:
        return f"{note}; {suffix}" if note else suffix

    @staticmethod
    def _print_pull_final_dag(*, head_start: str, dry_steps: list[DryRunStepReport]) -> None:
        current_head = head_start
        lines = ["", "Projected DAG (dry-run, pull)", f"  start: {head_start}"]
        for idx, step in enumerate(dry_steps, start=1):
            if step.status != RunResult.OK or step.commit_count <= 0:
                continue

            if step.squash_commit:
                range_text = (
                    f"{short_ref(step.local_split)} -> {short_ref(step.remote_head)}"
                    if step.local_split and step.remote_head
                    else "unknown"
                )
                lines.append(f"  step {idx} {step.path}:")
                lines.append(
                    f"    squash: {step.squash_commit} parent={step.commit_before} "
                    f"subtree_range={range_text}"
                )
                lines.append(
                    f"    merge:  {step.commit_after} "
                    f"parents=({step.commit_before}, {step.squash_commit})"
                )
            else:
                lines.append(f"  step {idx} {step.path}: {step.commit_before} -> {step.commit_after}")
            current_head = step.commit_after

        lines.append(f"  final: {current_head}")
        print("\n".join(lines))

    @staticmethod
    def _shared_dry_report_kwargs(
        *, dry_run: bool, print_branch: bool, head_start: str, dry_steps: list[DryRunStepReport],
    ) -> dict[str, object]:
        return {
            "dry_run": dry_run,
            "print_branch": print_branch,
            "head_start": head_start,
            "dry_steps": dry_steps,
        }

    @staticmethod
    def _print_dry_report(
        *, dry_run: bool, print_branch: bool, action: str,
        head_start: str, dry_steps: list[DryRunStepReport],
    ) -> None:
        if dry_run and print_branch:
            print("")
            report_kwargs = {"action": action, "head_start": head_start, "steps": dry_steps}
            print(render_dry_run_report(**report_kwargs))
            if action == SUBTREE_PULL_ACTION:
                SubtreeOpsBase._print_pull_final_dag(head_start=head_start, dry_steps=dry_steps)

    @staticmethod
    def _next_projected_squash_merge(
        *, projected_tree: list[str], projected_index: int, operation: str, path: str,
    ) -> tuple[int, str, str, str, tuple[str, ...]]:
        before = projected_tree[-1]
        squash_ref = dry_run_ref(projected_index + 1)
        merge_ref = dry_run_ref(projected_index + 2)
        projected_tree.append(merge_ref)
        commit_message = f"Update subtree {path}" if operation == "pull" else f"Add subtree {path}"
        graph = (
            f"first-parent: {before} -> {merge_ref}",
            f"squash-parent: {before} -> {squash_ref}",
            f"merge-parent: {squash_ref} -> {merge_ref}",
        )
        return projected_index + 2, squash_ref, merge_ref, commit_message, graph

    def _run_subtree(self, args: list[str], *, no_cache: bool = False) -> bool:
        """Invoke `git subtree ...` and print a merged failure payload on error."""
        result = self.run_subtree(args, no_cache=no_cache)
        if result.returncode == 0:
            return True
        self.console.print_error("git subtree failed")
        merged = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        if merged:
            self.console.print_info(merged)
        return False

    def print_summary(self, summary: Summary) -> None:
        print("")
        status_views = {
            RunResult.OK: self.console.green("ok"),
            RunResult.FAILED: self.console.red("FAILED"),
            RunResult.SKIPPED: self.console.yellow("skipped"),
        }
        for path, result in zip(summary.paths, summary.results, strict=True):
            print(f"  {path:<40} {status_views.get(result, self.console.yellow('skipped'))}")
        print("")
        print(
            f"{self.console.green(str(summary.ok_count))} succeeded, "
            f"{self.console.red(str(summary.fail_count))} failed, "
            f"{self.console.yellow(str(summary.skip_count))} skipped"
        )

    def _update_tracking(self, *, paths: list[str], mask: str, track: bool) -> int:
        if not paths:
            self.die(("track" if track else "untrack") + " requires at least one path filter")

        update_mask = normalize_track_mask(mask)
        if update_mask not in TRACK_UPDATE_MASK_VALUES:
            self.die("mask must be one of: pull, push, both")

        lines = self.read_libs_file()
        entries = self.validate_entries(lines)
        matched = 0
        changed = 0
        action = "track" if track else "untrack"

        for idx, entry in enumerate(entries):
            if not self.match_filter(entry.local_path, paths):
                continue
            matched += 1
            updated_mask = apply_track_mask(entry.track_mask, update_mask, track=track)
            if updated_mask == entry.track_mask:
                print(
                    f"  {action:<7} {entry.local_path} ... {self.console.yellow('skipped')} "
                    f"(unchanged: {entry.track_mask})"
                )
                continue

            updated_raw = (
                f"{entry.repo}, {entry.local_path}, {entry.branch}"
                if updated_mask == TRACK_MASK_BOTH
                else f"{entry.repo}, {entry.local_path}, {entry.branch}, {updated_mask}"
            )
            updated_entry_kwargs = {
                "raw": updated_raw,
                "repo": entry.repo,
                "local_path": entry.local_path,
                "branch": entry.branch,
                "track_mask": updated_mask,
            }
            entries[idx] = LibEntry(**updated_entry_kwargs)
            print(
                f"  {action:<7} {entry.local_path} ... {self.console.green('ok')} "
                f"({entry.track_mask} -> {updated_mask})"
            )
            changed += 1

        if not matched:
            self.warn("no entries matched provided path filters")
            return 1

        if changed:
            self._write_entries(entries)
            print(f"\nUpdated {self.libs_file}")
        else:
            print("\nNo changes.")
        return 0

    def die(self, message: str) -> None:
        self.console.print_error(message)
        raise SystemExit(1)

    def warn(self, message: str) -> None:
        self.console.print_warning(message)

    @staticmethod
    def _share_for_key(key: RepoBranchKey, key_shares: dict[RepoBranchKey, float] | None) -> float:
        if not key_shares:
            return LOOKUP_SHARE_DEFAULT
        return max(LOOKUP_SHARE_DEFAULT, float(key_shares.get(key, LOOKUP_SHARE_DEFAULT)))

    @staticmethod
    def _partition_remote_keys(
        keys: list[RepoBranchKey], worker_count: int, key_shares: dict[RepoBranchKey, float] | None = None,
    ) -> list[list[RepoBranchKey]]:
        if key_shares:
            batches: list[list[RepoBranchKey]] = [[] for _ in range(worker_count)]
            loads = [0.0 for _ in range(worker_count)]
            for key in sorted(keys, key=lambda value: SubtreeOpsBase._share_for_key(value, key_shares), reverse=True):
                idx = loads.index(min(loads))
                batches[idx].append(key)
                loads[idx] += SubtreeOpsBase._share_for_key(key, key_shares)
            return batches

        batches: list[list[RepoBranchKey]] = [[] for _ in range(worker_count)]
        for idx, key in enumerate(keys):
            batches[idx % worker_count].append(key)
        return batches

    def _estimate_lookup_share(self, full: Path) -> float:
        cache_key = str(full)
        with self._cache_lock:
            if cache_key in self._lookup_share_cache:
                return self._lookup_share_cache[cache_key]

        if not full.is_dir():
            with self._cache_lock:
                self._lookup_share_cache[cache_key] = LOOKUP_SHARE_DEFAULT
            return LOOKUP_SHARE_DEFAULT

        share_value = LOOKUP_SHARE_DEFAULT
        try:
            # -k: always report 1024-byte blocks (macOS default is 512-byte).
            du_env = {"LC_ALL": "C"}
            du_cmd = ["du", "-sk", str(full)]
            du_result = run(du_cmd, cwd=self.repo_root, env=du_env, timeout_sec=LOOKUP_SHARE_TIMEOUT_SEC, check=False)
            token = du_result.stdout.strip().split(maxsplit=1)[0] if du_result.returncode == 0 and du_result.stdout.strip() else ""
            if token:
                size_kib = max(0, int(token))
                units = (size_kib + LOOKUP_SHARE_KIB_PER_UNIT - 1) // LOOKUP_SHARE_KIB_PER_UNIT
                share_value = max(LOOKUP_SHARE_DEFAULT, float(units))
        except (ValueError, OSError):
            share_value = LOOKUP_SHARE_DEFAULT

        with self._cache_lock:
            self._lookup_share_cache[cache_key] = share_value
        return share_value

    def _remote_key_shares(
        self, *, selected_entries: list[tuple[LibEntry, Path]], include_when: Callable[[Path], bool],
    ) -> dict[RepoBranchKey, float]:
        shares: dict[RepoBranchKey, float] = {}
        for entry, full in selected_entries:
            if not include_when(full):
                continue
            key = (entry.repo, entry.branch)
            share = self._estimate_lookup_share(full)
            shares[key] = max(shares.get(key, LOOKUP_SHARE_DEFAULT), share)
        return shares

    def _prefetch_remote_batch(
        self,
        *,
        worker_idx: int,
        keys: list[RepoBranchKey],
        progress: ThreadProgressDisplay | None,
        key_shares: dict[RepoBranchKey, float] | None = None,
    ) -> dict[RepoBranchKey, RefLookupResult]:
        total = len(keys)
        results: dict[RepoBranchKey, RefLookupResult] = {}
        for pos, (repo, branch) in enumerate(keys, start=1):
            if progress:
                progress.task_start(worker_idx, f"ls-remote {repo} {branch} ({pos}/{total})")
            try:
                lookup = self.ls_remote_head(repo, branch)
            except Exception as exc:
                lookup = self._new_lookup_result(ref=None, note=f"remote head lookup failed: {exc}")
            results[(repo, branch)] = lookup
            task_share = self._share_for_key((repo, branch), key_shares)

            if lookup.ref:
                status = f"{repo} ({branch}) -> {short_ref(lookup.ref)} [{pos}/{total}]"
                if progress:
                    progress.task_done_with_share(worker_idx, status, success=True, share_done=task_share)
            else:
                note = lookup.note or "unknown error"
                status = f"{repo} ({branch}) FAILED [{pos}/{total}] {note}"
                if progress:
                    progress.task_done_with_share(worker_idx, status, success=False, share_done=task_share)

        if progress:
            progress.worker_done(worker_idx, f"completed {total} lookup(s)")
        return results

    def _prefetch_remote_heads(
        self,
        keys: list[RepoBranchKey],
        *,
        key_shares: dict[RepoBranchKey, float] | None = None,
        show_progress: bool = True,
    ) -> dict[RepoBranchKey, RefLookupResult]:
        unique_keys = sorted(set(keys))
        with self._cache_lock:
            missing_keys = [key for key in unique_keys if key not in self._remote_head_cache]

        if missing_keys:
            worker_count = min(MAX_PARALLEL_REMOTE_QUERIES, len(missing_keys))
            worker_batches = self._partition_remote_keys(missing_keys, worker_count, key_shares)
            worker_totals = [len(batch) for batch in worker_batches]
            worker_shares = [sum(self._share_for_key(key, key_shares) for key in batch) for batch in worker_batches]

            console = getattr(self, "console", None)
            has_progress = bool(show_progress and console and hasattr(console, "thread_progress"))
            progress = (
                console.thread_progress(title="Remote head lookup", worker_totals=worker_totals, worker_shares=worker_shares)
                if has_progress
                else None
            )
            progress_ctx = progress if progress is not None else nullcontext()
            with progress_ctx:
                with ThreadPoolExecutor(max_workers=worker_count) as pool:
                    future_map = {
                        pool.submit(self._prefetch_remote_batch, worker_idx=idx, keys=batch, progress=progress, key_shares=key_shares): idx
                        for idx, batch in enumerate(worker_batches)
                        if batch
                    }
                    for future in as_completed(future_map):
                        worker_idx = future_map[future]
                        try:
                            batch_result = future.result()
                            with self._cache_lock:
                                self._remote_head_cache.update(batch_result)
                        except Exception as exc:
                            if progress:
                                progress.worker_done(worker_idx, f"FAILED: {exc}")
                            with self._cache_lock:
                                for key in worker_batches[worker_idx]:
                                    self._remote_head_cache[key] = self._new_lookup_result(
                                        ref=None,
                                        note=f"remote head lookup failed: {exc}",
                                    )

        with self._cache_lock:
            return {
                key: self._remote_head_cache.get(key, self._new_lookup_result(ref=None, note="remote head lookup unavailable"))
                for key in unique_keys
            }

    def _prefetch_remote_lookup(
        self,
        *,
        dry_run: bool,
        selected_entries: list[tuple[LibEntry, Path]],
        include_when: Callable[[Path], bool],
        show_progress: bool,
    ) -> dict[RepoBranchKey, RefLookupResult]:
        if not dry_run:
            return {}
        remote_keys = [(entry.repo, entry.branch) for entry, full in selected_entries if include_when(full)]
        key_shares = self._remote_key_shares(selected_entries=selected_entries, include_when=include_when)
        return self._prefetch_remote_heads(remote_keys, key_shares=key_shares, show_progress=show_progress)

    def _fetch_remote_for_count(self, repo: str, branch: str) -> RefLookupResult:
        key = (repo, branch)

        # Single lock span: cache check → fetch → FETCH_HEAD resolution.
        # Without this, two threads can both miss the cache and race on
        # FETCH_HEAD between separate lock acquisitions.
        with self._mutation_lock:
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

            result = self.run(filtered_args, mutates_repo=False)
            note = ""
            if result.returncode != 0:
                fallback_args = ["fetch", "--quiet", "--no-tags", f"--depth={COUNT_FETCH_DEPTH}", repo, branch]
                fallback_result = self.run(fallback_args, mutates_repo=False)
                if fallback_result.returncode != 0:
                    value = self._new_lookup_result(
                        ref=None,
                        note=f"shallow fetch failed: {fallback_result.stderr.strip() or 'unknown error'}",
                    )
                    self._count_fetch_cache[key] = value
                    return value
                note = "count fetch fallback used (remote does not support filter)"

            fetch_head = self.rev_parse("FETCH_HEAD")

            if not fetch_head.ref:
                value = self._new_lookup_result(
                    ref=None,
                    note=f"unable to resolve FETCH_HEAD after shallow fetch: {fetch_head.note}",
                )
                self._count_fetch_cache[key] = value
                return value

            value = self._new_lookup_result(ref=fetch_head.ref, note=note)
            self._count_fetch_cache[key] = value
            return value

    def _local_subtree_split_from_log(self, path: str) -> str | None:
        result = self.run(["log", "-n", "1", "--format=%B", "--grep", f"{SUBTREE_DIR_TRAILER} {path}", "--fixed-strings"])
        if result.returncode != 0 or not result.stdout.strip():
            return None
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if line.startswith(SUBTREE_SPLIT_TRAILER):
                commit = line.split(":", 1)[1].strip()
                if commit:
                    return commit
        return None

    def _discover_local_split(self, path: str, *, present: bool) -> RefLookupResult:
        with self._cache_lock:
            if path in self._local_split_cache:
                return self._local_split_cache[path]
        if not present:
            value = self._new_lookup_result(ref=None, note="local subtree is missing")
            self._local_split_cache[path] = value
            return value

        split_from_log = self._local_subtree_split_from_log(path)
        if split_from_log:
            value = self._new_lookup_result(ref=split_from_log, note="local split resolved from subtree metadata")
            self._local_split_cache[path] = value
            return value

        if not ALLOW_SUBTREE_SPLIT_FALLBACK:
            value = self._new_lookup_result(
                ref=None,
                note="local split trailer not found (slow subtree split fallback disabled)",
            )
            self._local_split_cache[path] = value
            return value

        result = self.run_mutating(["subtree", "split", f"--prefix={path}", "HEAD"])
        if result.returncode != 0 or not result.stdout.strip():
            value = self._new_lookup_result(
                ref=None,
                note=f"unable to derive local split: {result.stderr.strip() or 'unknown error'}",
            )
            self._local_split_cache[path] = value
            return value

        split = result.stdout.strip().splitlines()[-1].strip()
        value = self._new_lookup_result(
            ref=split if split else None,
            note="local split resolved via git subtree split",
        )
        self._local_split_cache[path] = value
        return value

    def _probe_sync(
        self,
        *,
        path: str,
        repo: str,
        branch: str,
        present: bool,
        remote_lookup: RefLookupResult | None = None,
    ) -> SyncProbe:
        local_lookup = self._discover_local_split(path, present=present)
        local_split = local_lookup.ref
        local_note = local_lookup.note
        if remote_lookup is None:
            remote_lookup = self.ls_remote_head(repo, branch)
        remote_head = remote_lookup.ref
        remote_note = remote_lookup.note

        pull_commits: int | None = None
        push_commits: int | None = None
        diverged = False
        notes: list[str] = []
        if local_note:
            notes.append(local_note)
        if remote_note:
            notes.append(remote_note)

        if local_split and remote_head:
            compare_target = remote_head
            if not self.commit_exists(compare_target):
                fetched_lookup = self._fetch_remote_for_count(repo, branch)
                fetched_head = fetched_lookup.ref
                fetch_note = fetched_lookup.note
                if fetch_note:
                    notes.append(fetch_note)
                if fetched_head:
                    compare_target = fetched_head
                    if fetched_head != remote_head:
                        notes.append(
                            f"remote moved during status check ({short_ref(remote_head)} -> {short_ref(fetched_head)})"
                        )

            if self.commit_exists(compare_target):
                local_is_ancestor = self.is_ancestor(local_split, compare_target)
                remote_is_ancestor = self.is_ancestor(compare_target, local_split)
                if local_is_ancestor is True and remote_is_ancestor is True:
                    pull_commits = 0
                    push_commits = 0
                elif local_is_ancestor is True and remote_is_ancestor is False:
                    pull_commits = self.count_commits(local_split, compare_target)
                    push_commits = 0
                elif local_is_ancestor is False and remote_is_ancestor is True:
                    pull_commits = 0
                    push_commits = self.count_commits(compare_target, local_split)
                elif local_is_ancestor is False and remote_is_ancestor is False:
                    diverged = True
                    notes.append("local and remote histories diverged (non fast-forward)")
                else:
                    pull_commits = self.count_commits(local_split, compare_target)
                    push_commits = self.count_commits(compare_target, local_split)
            else:
                notes.append(
                    f"remote commit {short_ref(remote_head)} not available locally after bounded fetch; "
                    "exact commit deltas unavailable"
                )

        if local_split and remote_head and pull_commits is None and not diverged:
            notes.append("unable to compute pull commit delta")
        if local_split and remote_head and push_commits is None and not diverged:
            notes.append("unable to compute push commit delta")

        return SyncProbe(
            local_split=local_split,
            remote_head=remote_head,
            pull_commits=pull_commits,
            push_commits=push_commits,
            note="; ".join(notes),
        )

    def require_git_repo(self) -> None:
        if not self.is_git_repo():
            self.die(f"Not a git repository: {self.repo_root}")

    def require_clean_tree(self) -> None:
        if self.has_changes(ignored_rel=self._ignored_rel):
            self.die("Working tree is not clean. Commit or stash your changes first.\n  git stash push --include-untracked")

    def _prepare_subtree_clean_tree(self) -> None:
        dirty = self.collect_dirty_paths()
        if not dirty:
            return

        allowed = set(self._ignored_rel)
        if disallowed := next((path for path in dirty if path not in allowed), ""):
            self.die(f"Working tree has modifications in '{disallowed}'. Commit or stash before running subtree.")
        allowed_dirty = sorted(set(dirty) & allowed)
        if not allowed_dirty:
            return

        message = f"subtree-temp-{uuid.uuid4().hex[:8]}"
        args = ["stash", "push", "-u", "-m", message, "--", *allowed_dirty]
        result = self.run_mutating(args)
        if result.returncode != 0:
            self.die("Unable to stash temporary subtree metadata changes.")

        self._stash_ref = self.find_stash_ref_by_marker(message)
        if not self._stash_ref:
            self.die("Unable to identify temporary stash entry. Restore manually via 'git stash list' and retry.")

    def _restore_subtree_clean_tree(self) -> bool:
        if not self._stash_ref:
            return True
        result = self.run_mutating(["stash", "pop", "-q", self._stash_ref])
        self._stash_ref = ""
        if result.returncode != 0:
            self.warn("stash pop had conflicts; resolve manually.")
            return False
        return True
