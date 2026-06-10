"""Subtree management implementation."""

from __future__ import annotations

import argparse
import shutil
import sys

from functools import partial
from pathlib import Path
from typing import Iterable

# When executed directly (python subtree_manager.py), __package__ is unset.
# Patch sys.path once so relative imports work in both modes.
if __package__ in {None, ""}:
    _tools_root = str(Path(__file__).resolve().parents[2])
    if _tools_root not in sys.path:
        sys.path.insert(0, _tools_root)
    __package__ = "spectral_tools.subtrees"  # noqa: A001

from ..core.cli import add_dry_run_flag, add_print_branch_flag, add_progress_bar_flag
from ..core.utils import short_ref, find_repo_root
from ..core.console import Console
from .subtree_ops import SubtreeOpsBase
from .subtree_models import (
    CLI_PROG,
    DEFAULT_PRINT_BRANCH,
    HELP_ALIASES,
    LIBS_RELATIVE_PATH,
    LIST_ALIASES,
    MOVE_ALIAS,
    LibEntry,
    OperationState,
    REMOVE_ALIAS,
    STATUS_ALIASES,
    SUBTREE_ADD_ACTION,
    SUBTREE_HELP_ACTION,
    SUBTREE_LIST_ACTION,
    SUBTREE_MOVE_ACTION,
    SUBTREE_PULL_ACTION,
    SUBTREE_PUSH_ACTION,
    SUBTREE_REMOVE_ACTION,
    SUBTREE_STATUS_ACTION,
    SUBTREE_TRACK_ACTION,
    SUBTREE_UNTRACK_ACTION,
    TRACK_MASK_BOTH,
    TRACK_UPDATE_MASK_VALUES,
    UPDATE_PY_RELATIVE_PATH,
    DryRunStepReport,
    RunResult,
    StatusEntryReport,
    Summary,
    entry_allows,
)
from .subtree_reporting import render_status_report


class SubtreeManager(SubtreeOpsBase):
    """High-level subtree workflow orchestration for pull/push/status operations."""

    def __init__(self, repo_root: Path, libs_file: Path, ignored_paths: Iterable[Path]) -> None:
        super().__init__(repo_root, ignored_paths=ignored_paths)
        self.libs_file = libs_file.resolve()
        self.console = Console()

    def do_pull(
        self,
        *,
        dry_run: bool,
        filters: list[str],
        skip_filters: list[str] | None = None,
        print_branch: bool = DEFAULT_PRINT_BRANCH,
        show_progress: bool = True,
        no_cache: bool = False,
    ) -> int:
        lines = self.read_libs_file()
        if not lines:
            print("No subtree entries in the manifest (third_party/libs.yaml)")
            return 0

        entries = self.validate_entries(lines)
        selected_entries = list(self._iter_selected_entries(entries, filters))
        skip_filters = list(skip_filters or [])
        active_entries = [
            (entry, full)
            for entry, full in selected_entries
            if entry_allows(entry, SUBTREE_PULL_ACTION)
            and not (skip_filters and self.match_filter(entry.local_path, skip_filters))
        ]
        summary = Summary()
        dry_steps: list[DryRunStepReport] = []
        head_start = self.current_head()
        projected_tree: list[str] = [head_start]
        projected_index = 0
        state_kwargs = {
            "dry_run": dry_run,
            "summary": summary,
            "dry_steps": dry_steps,
            "projected_tree": projected_tree,
        }
        state = OperationState(**state_kwargs)
        record_skip = partial(self._record_skip, operation="pull", state=state)
        record_non_directory_failure = partial(self._record_non_directory_failure, operation="pull", state=state)
        prefetch_kwargs = {"dry_run": dry_run, "selected_entries": active_entries, "include_when": lambda full: not (full.exists() and not full.is_dir()), "show_progress": show_progress}
        remote_lookup_map = self._prefetch_remote_lookup(**prefetch_kwargs)

        if not dry_run:
            if no_cache:
                print(f"  {self.console.yellow('[no-cache]')} subtree fetch runs with fetch.negotiationAlgorithm=noop")
            self._prepare_subtree_clean_tree()

        restore_ok = True
        try:
            for entry, full in selected_entries:
                if skip_filters and self.match_filter(entry.local_path, skip_filters):
                    print(f"  pull  {entry.local_path} ... {self.console.yellow('skipped')} (--skip)")
                    record_skip(entry=entry, note="excluded via --skip")
                    continue
                if not entry_allows(entry, SUBTREE_PULL_ACTION):
                    print(f"  pull  {entry.local_path} ... {self.console.yellow('skipped')} (untracked for pull)")
                    record_skip(entry=entry, note=f"tracking mask '{entry.track_mask}' excludes pull")
                    continue
                if full.exists() and not full.is_dir():
                    record_non_directory_failure(entry=entry)
                    continue

                if dry_run:
                    operation = "pull" if full.is_dir() else "add"
                    probe_kwargs = {"path": entry.local_path, "repo": entry.repo, "branch": entry.branch, "present": full.is_dir(), "remote_lookup": remote_lookup_map.get((entry.repo, entry.branch))}
                    probe = self._probe_sync(**probe_kwargs)
                    delta_commits = probe.pull_commits
                    projected_before = projected_tree[-1]
                    local_commit_count = 2
                    if operation == "add":
                        projected_kwargs = {"projected_tree": projected_tree, "projected_index": projected_index, "operation": operation, "path": entry.local_path}
                        projected_index, squash_commit, projected_after, commit_message, projected_graph = self._next_projected_squash_merge(**projected_kwargs)
                        projected_graph = (*projected_graph, f"local-dag-squash-base: {projected_before}")
                    elif delta_commits and delta_commits > 0:
                        projected_kwargs = {"projected_tree": projected_tree, "projected_index": projected_index, "operation": operation, "path": entry.local_path}
                        projected_index, squash_commit, projected_after, commit_message, projected_graph = self._next_projected_squash_merge(**projected_kwargs)
                        subtree_range = (
                            f"{short_ref(probe.local_split)} -> {short_ref(probe.remote_head)}"
                            if probe.local_split and probe.remote_head
                            else "unknown"
                        )
                        projected_graph = (
                            *projected_graph,
                            f"local-dag-squash-base: {projected_before}",
                            f"subtree-range-squashed: {subtree_range}",
                        )
                    else:
                        local_commit_count = 0
                        squash_commit = None
                        projected_after = projected_before
                        commit_message = "No local commit (already up to date or delta unavailable)"
                        reason = "already up to date" if delta_commits == 0 else "commit delta unavailable"
                        projected_graph = (f"local DAG unchanged at {projected_before} ({reason})",)

                    if full.is_dir():
                        projected_suffix = f" -> {projected_after}" if local_commit_count else ""
                        print(
                            f"  {self.console.yellow('[dry-run]')} would pull "
                            f"{self.console.bold(entry.local_path)} from {entry.repo} ({entry.branch}) "
                            f"{projected_suffix}"
                            f"(local_commits={local_commit_count if delta_commits is not None else 'unknown'} "
                            f"remote={short_ref(probe.remote_head)} "
                            f"ff={delta_commits if delta_commits is not None else 'unknown'})"
                        )
                    else:
                        print(
                            f"  {self.console.yellow('[dry-run]')} would add "
                            f"{self.console.bold(entry.local_path)} from {entry.repo} ({entry.branch}) "
                            f"-> {projected_after} "
                            f"(local_commits=2 "
                            f"remote={short_ref(probe.remote_head)})"
                        )
                    step_kwargs = {
                        "dry_steps": dry_steps,
                        "entry": entry,
                        "operation": operation,
                        "status": RunResult.OK,
                        "projected_tree": projected_tree,
                        "note": probe.note,
                        "local_split": probe.local_split,
                        "remote_head": probe.remote_head,
                        "delta_commits": delta_commits,
                        "commit_before": projected_before,
                        "commit_after": projected_after,
                        "commit_message": commit_message,
                        "squash_commit": squash_commit,
                        "commit_count": local_commit_count,
                        "projected_graph": projected_graph,
                    }
                    self._append_dry_step(**step_kwargs)
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
                        ],
                        no_cache=no_cache,
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
                        ],
                        no_cache=no_cache,
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

        dry_report_kwargs = self._shared_dry_report_kwargs(
            dry_run=dry_run, print_branch=print_branch, head_start=head_start, dry_steps=dry_steps,
        )
        dry_report_kwargs["action"] = SUBTREE_PULL_ACTION
        self._print_dry_report(**dry_report_kwargs)
        if not dry_run:
            self.print_summary(summary)

        return 1 if summary.fail_count > 0 or not restore_ok else 0

    def do_push(
        self,
        *,
        dry_run: bool,
        filters: list[str],
        skip_filters: list[str] | None = None,
        print_branch: bool = DEFAULT_PRINT_BRANCH,
        show_progress: bool = True,
    ) -> int:
        lines = self.read_libs_file()
        if not lines:
            print("No subtree entries in the manifest (third_party/libs.yaml)")
            return 0

        entries = self.validate_entries(lines)
        selected_entries = list(self._iter_selected_entries(entries, filters))
        skip_filters = list(skip_filters or [])
        active_entries = [
            (entry, full)
            for entry, full in selected_entries
            if entry_allows(entry, SUBTREE_PUSH_ACTION)
            and not (skip_filters and self.match_filter(entry.local_path, skip_filters))
        ]
        summary = Summary()
        dry_steps: list[DryRunStepReport] = []
        head_start = self.current_head()
        projected_tree: list[str] = [head_start]
        state_kwargs = {
            "dry_run": dry_run,
            "summary": summary,
            "dry_steps": dry_steps,
            "projected_tree": projected_tree,
        }
        state = OperationState(**state_kwargs)
        record_skip = partial(self._record_skip, operation="push", state=state)
        record_non_directory_failure = partial(self._record_non_directory_failure, operation="push", state=state)
        prefetch_kwargs = {"dry_run": dry_run, "selected_entries": active_entries, "include_when": lambda full: full.is_dir(), "show_progress": show_progress}
        remote_lookup_map = self._prefetch_remote_lookup(**prefetch_kwargs)

        if not dry_run:
            self._prepare_subtree_clean_tree(allow_tools_dirty_without_stash=True)

        restore_ok = True
        try:
            for entry, full in selected_entries:
                if skip_filters and self.match_filter(entry.local_path, skip_filters):
                    print(f"  push  {entry.local_path} ... {self.console.yellow('skipped')} (--skip)")
                    record_skip(entry=entry, note="excluded via --skip")
                    continue
                if not entry_allows(entry, SUBTREE_PUSH_ACTION):
                    print(f"  push  {entry.local_path} ... {self.console.yellow('skipped')} (untracked for push)")
                    record_skip(entry=entry, note=f"tracking mask '{entry.track_mask}' excludes push")
                    continue
                if full.exists() and not full.is_dir():
                    record_non_directory_failure(entry=entry)
                    continue

                if dry_run:
                    if not full.is_dir():
                        print(
                            f"  {self.console.yellow('[dry-run]')} {entry.local_path} does not exist, would skip"
                        )
                        record_skip(entry=entry, note="local subtree is not present")
                        continue

                    probe_kwargs = {"path": entry.local_path, "repo": entry.repo, "branch": entry.branch, "present": True, "remote_lookup": remote_lookup_map.get((entry.repo, entry.branch))}
                    probe = self._probe_sync(**probe_kwargs)
                    delta_commits = probe.push_commits
                    print(
                        f"  {self.console.yellow('[dry-run]')} would push "
                        f"{self.console.bold(entry.local_path)} to {entry.repo} ({entry.branch}) "
                        f"(remote={short_ref(probe.remote_head)} "
                        f"delta={delta_commits if delta_commits is not None else 'unknown'})"
                    )
                    step_kwargs = {
                        "dry_steps": dry_steps,
                        "entry": entry,
                        "operation": "push",
                        "status": RunResult.OK,
                        "projected_tree": projected_tree,
                        "note": probe.note,
                        "local_split": probe.local_split,
                        "remote_head": probe.remote_head,
                        "delta_commits": delta_commits,
                        "commit_message": "no local commit (push updates remote only)",
                    }
                    self._append_dry_step(**step_kwargs)
                    summary.record_ok(entry.local_path)
                    continue

                if not full.is_dir():
                    print(f"  push  {entry.local_path} ... {self.console.yellow('skipped')} (not present)")
                    record_skip(entry=entry, note="local subtree is not present")
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

        dry_report_kwargs = self._shared_dry_report_kwargs(
            dry_run=dry_run, print_branch=print_branch, head_start=head_start, dry_steps=dry_steps,
        )
        dry_report_kwargs["action"] = SUBTREE_PUSH_ACTION
        self._print_dry_report(**dry_report_kwargs)
        if not dry_run:
            self.print_summary(summary)

        return 1 if summary.fail_count > 0 or not restore_ok else 0

    def do_add(self, entry_specs: list[str]) -> int:
        if not entry_specs:
            self.die("add requires at least one entry in format: repo_url, local_path[, branch[, track_mask]]")

        existing_lines = self.read_libs_file(allow_missing=True)
        existing_entries = self.validate_entries(existing_lines) if existing_lines else []
        new_entries = self.validate_entries(entry_specs)

        existing_paths = {entry.local_path for entry in existing_entries}
        added_count = 0
        for entry in new_entries:
            if entry.local_path in existing_paths:
                print(f"  add   {entry.local_path} ... {self.console.yellow('skipped')} (already tracked)")
                continue
            existing_entries.append(entry)
            existing_paths.add(entry.local_path)
            print(f"  add   {entry.local_path} ... {self.console.green('ok')} (tracked only, not pulled)")
            added_count += 1

        if added_count:
            self._write_entries(existing_entries)
            print(f"\nUpdated {self.libs_file}")
        else:
            print("\nNo changes.")
        return 0

    def do_track(self, *, paths: list[str], mask: str) -> int:
        return self._update_tracking(paths=paths, mask=mask, track=True)

    def do_untrack(self, *, paths: list[str], mask: str) -> int:
        return self._update_tracking(paths=paths, mask=mask, track=False)

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
                self.run_mutating(["add", "-A", path])

                diff_result = self.run(["diff", "--cached", "--quiet", "--", path])
                if diff_result.returncode == 0:
                    print(f"{self.console.yellow('skipped')} (already clean)")
                    summary.record_skip(path)
                    continue

                commit_result = self.run_mutating(["commit", "-m", f"Remove subtree {path}"])
                if commit_result.returncode == 0:
                    print(self.console.green("ok"))
                    summary.record_ok(path)
                else:
                    print(self.console.red("FAILED"))
                    summary.record_fail(path)
        finally:
            restore_ok = self._restore_subtree_clean_tree()

        self.print_summary(summary)
        return 1 if summary.fail_count > 0 or not restore_ok else 0

    def do_move(self, *, source_path: str, destination_path: str) -> int:
        summary = Summary()
        self._prepare_subtree_clean_tree()
        restore_ok = True

        try:
            try:
                source_norm, source_full = self._resolve_local_path(source_path)
                destination_norm, destination_full = self._resolve_local_path(destination_path)
            except ValueError as exc:
                self.console.print_error(f"invalid path: {exc}")
                return 1

            if source_norm == destination_norm:
                print(f"  move   {source_norm} ... {self.console.yellow('skipped')} (source and destination are identical)")
                summary.record_skip(source_norm)
                self.print_summary(summary)
                return 0

            lines = self.read_libs_file()
            entries = self.validate_entries(lines)
            source_idx = next((idx for idx, entry in enumerate(entries) if entry.local_path == source_norm), -1)

            if source_idx < 0:
                self.console.print_error(f"'{source_norm}' is not configured in {self.libs_file}")
                return 1
            if any(entry.local_path == destination_norm for entry in entries):
                self.console.print_error(f"'{destination_norm}' is already configured in {self.libs_file}")
                return 1
            if destination_full.exists():
                self.console.print_error(f"destination path already exists: {destination_norm}")
                return 1
            if source_full.exists() and source_full.is_symlink():
                self.console.print_error(f"source path is a symlink and not supported: {source_norm}")
                return 1
            if source_full.exists() and not source_full.is_dir():
                self.console.print_error(f"source path exists but is not a directory: {source_norm}")
                return 1

            moved_on_disk = False
            if source_full.is_dir():
                destination_full.parent.mkdir(parents=True, exist_ok=True)
                git_mv = self.run_mutating(["mv", "--", source_norm, destination_norm])
                if git_mv.returncode != 0:
                    shutil.move(str(source_full), str(destination_full))
                    self.run_mutating(["add", "-A", "--", source_norm, destination_norm])
                moved_on_disk = True

            source_entry = entries[source_idx]
            updated_raw = (
                f"{source_entry.repo}, {destination_norm}, {source_entry.branch}"
                if source_entry.track_mask == TRACK_MASK_BOTH
                else f"{source_entry.repo}, {destination_norm}, {source_entry.branch}, {source_entry.track_mask}"
            )
            updated_entry_kwargs = {
                "raw": updated_raw,
                "repo": source_entry.repo,
                "local_path": destination_norm,
                "branch": source_entry.branch,
                "track_mask": source_entry.track_mask,
            }
            entries[source_idx] = LibEntry(**updated_entry_kwargs)
            self._write_entries(entries)

            if libs_rel := self._safe_relative_to_root(self.libs_file):
                self.run_mutating(["add", "--", libs_rel])

            if self.run(["diff", "--cached", "--quiet"]).returncode == 0:
                print(f"  move   {source_norm} -> {destination_norm} ... {self.console.yellow('skipped')} (no changes)")
                summary.record_skip(source_norm)
                self.print_summary(summary)
                return 0

            print(f"  move   {self.console.bold(source_norm)} -> {self.console.bold(destination_norm)} ... ", end="", flush=True)
            commit_result = self.run_mutating(["commit", "-m", f"Move subtree {source_norm} -> {destination_norm}"])
            if commit_result.returncode == 0:
                mode = "moved directory + updated manifest" if moved_on_disk else "updated manifest (directory absent)"
                print(f"{self.console.green('ok')} ({mode})")
                summary.record_ok(f"{source_norm} -> {destination_norm}")
            else:
                print(self.console.red("FAILED"))
                summary.record_fail(source_norm)
        finally:
            restore_ok = self._restore_subtree_clean_tree()

        self.print_summary(summary)
        return 1 if summary.fail_count > 0 or not restore_ok else 0

    def do_list(self) -> int:
        if not self.libs_file.exists():
            print(f"No manifest at {self.libs_file}")
            return 0

        lines = self.read_libs_file()
        if not lines:
            print("No subtree entries in the manifest (third_party/libs.yaml)")
            return 0

        entries = self.validate_entries(lines)
        for entry in entries:
            _, full = self._resolve_local_path(entry.local_path)
            if full.is_dir():
                status = self.console.green("[present]")
            else:
                status = self.console.red("[missing]")
            print(
                f"  {entry.local_path:<40} {status} track={entry.track_mask:<4}  "
                f"{entry.repo} ({entry.branch})"
            )

        return 0

    def do_status(
        self, *, filters: list[str], print_branch: bool = DEFAULT_PRINT_BRANCH, show_progress: bool = True,
    ) -> int:
        lines = self.read_libs_file()
        if not lines:
            print("No subtree entries in the manifest (third_party/libs.yaml)")
            return 0

        entries = self.validate_entries(lines)
        selected_entries = list(self._iter_selected_entries(entries, filters))
        rows: list[StatusEntryReport] = []
        summary = Summary()
        remote_keys = [(entry.repo, entry.branch) for entry, _ in selected_entries]
        key_shares = self._remote_key_shares(selected_entries=selected_entries, include_when=lambda _: True)
        remote_lookup_map = self._prefetch_remote_heads(remote_keys, key_shares=key_shares, show_progress=show_progress)

        for entry, full in selected_entries:
            present = full.is_dir()
            probe_kwargs = {"path": entry.local_path, "repo": entry.repo, "branch": entry.branch, "present": present, "remote_lookup": remote_lookup_map.get((entry.repo, entry.branch))}
            probe = self._probe_sync(**probe_kwargs)

            status = RunResult.OK
            note = probe.note
            ff_count = probe.pull_commits
            if not present:
                status = RunResult.SKIPPED
                note = self._append_note(note, "local subtree is untracked in working tree (configured; would add on pull)")
            elif not probe.remote_head:
                status = RunResult.FAILED
                note = self._append_note(note, "unable to resolve remote head")
            elif not probe.local_split:
                status = RunResult.SKIPPED
                note = self._append_note(note, "unable to resolve local split commit")
            elif ff_count is None:
                status = RunResult.FAILED
                note = self._append_note(note, "unable to compute fast-forward commit count")

            row_kwargs = {
                "path": entry.local_path,
                "repo": entry.repo,
                "branch": entry.branch,
                "status": status,
                "local_split": probe.local_split,
                "remote_head": probe.remote_head,
                "fast_forward_commits": ff_count,
                "note": note,
            }
            row = StatusEntryReport(**row_kwargs)
            rows.append(row)

            ff_text = str(ff_count) if ff_count is not None else "unknown"
            presence_text = "untracked" if not present else "present"
            print(
                f"  status {entry.local_path:<34} "
                f"state={presence_text:<9} "
                f"track={entry.track_mask:<4} "
                f"local={short_ref(probe.local_split)} "
                f"-> remote={short_ref(probe.remote_head)} "
                f"ff={ff_text}"
            )

            summary.record(entry.local_path, status)

        if print_branch:
            print("")
            report_kwargs = {"rows": rows}
            print(render_status_report(**report_kwargs))

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
        print(f"  {CLI_PROG} pull [--dry-run|-n] [-P] [--no-progress-bar] [-f|--flush|--no-cache] [-s|--skip <path> ...] [<path>...]   Pull (or add) subtrees")
        print(f"  {CLI_PROG} push [--dry-run|-n] [-P] [--no-progress-bar] [-s|--skip <path> ...] [<path>...]   Push subtrees upstream")
        print(f"  {CLI_PROG} add <repo,path[,branch[,track_mask]]>...   Register subtree entries in the manifest only")
        print(f"  {CLI_PROG} track --mask {{pull|push|both}} <path>...   Re-enable updates for selected entries")
        print(f"  {CLI_PROG} untrack --mask {{pull|push|both}} <path>... Disable updates for selected entries")
        print(f"  {CLI_PROG} move|mv <source_path> <destination_path>    Move subtree path + update manifest")
        print(f"  {CLI_PROG} remove <path>...                  Remove and commit")
        print(f"  {CLI_PROG} status [-P] [--no-progress-bar] [<path>...]           Show fast-forward target and commit delta")
        print(f"  {CLI_PROG} list|ls                           Show configured subtrees")
        print(f"  {CLI_PROG} help                              Show this help")
        print("")
        print("add-entry format:  repo_url, local_path, branch[, track_mask]  (stored in third_party/libs.yaml)")
        print("Branch defaults to main if omitted. track_mask defaults to both.")
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
        SUBTREE_ADD_ACTION,
        SUBTREE_TRACK_ACTION,
        SUBTREE_UNTRACK_ACTION,
        SUBTREE_MOVE_ACTION,
        MOVE_ALIAS,
        SUBTREE_REMOVE_ACTION,
        REMOVE_ALIAS,
        SUBTREE_STATUS_ACTION,
        *LIST_ALIASES,
        *STATUS_ALIASES,
    }:
        action = args[0]
        rest = args[1:]
    elif args[0].startswith("-"):
        # Shorthand: "subtree_manager ... -n" means "subtree_manager ... pull -n".
        action = SUBTREE_PULL_ACTION
        rest = args
    else:
        raise ValueError(args[0])

    if action in {SUBTREE_PULL_ACTION, SUBTREE_PUSH_ACTION}:
        sub = argparse.ArgumentParser(prog=f"{CLI_PROG} {action}")
        add_dry_run_flag(sub)
        add_print_branch_flag(sub, default=DEFAULT_PRINT_BRANCH)
        add_progress_bar_flag(sub)
        sub.add_argument("-s", "--skip", dest="skip_paths", action="append", default=[])
        if action == SUBTREE_PULL_ACTION:
            sub.add_argument("-f", "--flush", "--no-cache", dest="no_cache", action="store_true")
        sub.add_argument("paths", nargs="*")
        parsed = sub.parse_args(rest)
        namespace_kwargs = {
            "action": action,
            "dry_run": parsed.dry_run,
            "print_branch": parsed.print_branch,
            "no_progress_bar": parsed.no_progress_bar,
            "no_cache": bool(getattr(parsed, "no_cache", False)),
            "skip_paths": list(parsed.skip_paths),
            "paths": parsed.paths,
        }
        return argparse.Namespace(**namespace_kwargs)

    if action == SUBTREE_ADD_ACTION:
        sub = argparse.ArgumentParser(prog=f"{CLI_PROG} add")
        sub.add_argument("entries", nargs="*")
        parsed = sub.parse_args(rest)
        if not parsed.entries:
            sub.error("the following arguments are required: entries")
        namespace_kwargs = {"action": SUBTREE_ADD_ACTION, "entries": parsed.entries}
        return argparse.Namespace(**namespace_kwargs)

    if action in {SUBTREE_TRACK_ACTION, SUBTREE_UNTRACK_ACTION}:
        sub = argparse.ArgumentParser(prog=f"{CLI_PROG} {action}")
        sub.add_argument("--mask", choices=sorted(TRACK_UPDATE_MASK_VALUES), default=TRACK_MASK_BOTH)
        sub.add_argument("paths", nargs="*")
        parsed = sub.parse_args(rest)
        if not parsed.paths:
            sub.error("the following arguments are required: paths")
        namespace_kwargs = {"action": action, "mask": parsed.mask, "paths": parsed.paths}
        return argparse.Namespace(**namespace_kwargs)

    if action in {SUBTREE_MOVE_ACTION, MOVE_ALIAS}:
        sub = argparse.ArgumentParser(prog=f"{CLI_PROG} move")
        sub.add_argument("source_path")
        sub.add_argument("destination_path")
        parsed = sub.parse_args(rest)
        namespace_kwargs = {
            "action": SUBTREE_MOVE_ACTION,
            "source_path": parsed.source_path,
            "destination_path": parsed.destination_path,
        }
        return argparse.Namespace(**namespace_kwargs)

    if action in {SUBTREE_REMOVE_ACTION, REMOVE_ALIAS}:
        sub = argparse.ArgumentParser(prog=f"{CLI_PROG} remove")
        sub.add_argument("paths", nargs="*")
        parsed = sub.parse_args(rest)
        if not parsed.paths:
            sub.error("the following arguments are required: paths")
        namespace_kwargs = {"action": "remove", "paths": parsed.paths}
        return argparse.Namespace(**namespace_kwargs)

    if action in LIST_ALIASES:
        sub = argparse.ArgumentParser(prog=f"{CLI_PROG} list")
        sub.parse_args(rest)
        namespace_kwargs = {"action": SUBTREE_LIST_ACTION}
        return argparse.Namespace(**namespace_kwargs)

    if action in {SUBTREE_STATUS_ACTION, *STATUS_ALIASES}:
        sub = argparse.ArgumentParser(prog=f"{CLI_PROG} status")
        add_print_branch_flag(sub, default=DEFAULT_PRINT_BRANCH)
        add_progress_bar_flag(sub)
        sub.add_argument("paths", nargs="*")
        parsed = sub.parse_args(rest)
        namespace_kwargs = {"action": SUBTREE_STATUS_ACTION, "print_branch": parsed.print_branch, "no_progress_bar": parsed.no_progress_bar, "paths": parsed.paths}
        return argparse.Namespace(**namespace_kwargs)

    raise ValueError(action)


def main(argv: list[str] | None = None) -> int:
    root = find_repo_root(Path(__file__))
    manager_kwargs = {"repo_root": root, "libs_file": root / LIBS_RELATIVE_PATH, "ignored_paths": [root / LIBS_RELATIVE_PATH, root / UPDATE_PY_RELATIVE_PATH]}
    manager = SubtreeManager(**manager_kwargs)

    manager.require_git_repo()

    args_raw = argv if argv is not None else sys.argv[1:]
    try:
        args = parse_cli(args_raw)
    except ValueError as exc:
        manager.die(f"Unknown action: {exc.args[0]} (use: pull, push, add, track, untrack, move, remove, status, list, help)")
        return 1

    if args.action == SUBTREE_HELP_ACTION:
        return manager.show_help()
    if args.action in {SUBTREE_PULL_ACTION, SUBTREE_PUSH_ACTION}:
        call_kwargs = {
            "dry_run": bool(args.dry_run),
            "filters": list(args.paths),
            "skip_filters": list(getattr(args, "skip_paths", [])),
            "print_branch": bool(getattr(args, "print_branch", DEFAULT_PRINT_BRANCH)),
            "show_progress": not bool(getattr(args, "no_progress_bar", False)),
        }
        if args.action == SUBTREE_PULL_ACTION:
            call_kwargs["no_cache"] = bool(getattr(args, "no_cache", False))
        method = manager.do_pull if args.action == SUBTREE_PULL_ACTION else manager.do_push
        return method(**call_kwargs)
    if args.action == SUBTREE_ADD_ACTION:
        return manager.do_add(list(getattr(args, "entries", [])))
    if args.action in {SUBTREE_TRACK_ACTION, SUBTREE_UNTRACK_ACTION}:
        call_kwargs = {"paths": list(getattr(args, "paths", [])), "mask": str(getattr(args, "mask", TRACK_MASK_BOTH))}
        method = manager.do_track if args.action == SUBTREE_TRACK_ACTION else manager.do_untrack
        return method(**call_kwargs)
    if args.action == SUBTREE_MOVE_ACTION:
        call_kwargs = {"source_path": str(getattr(args, "source_path", "")), "destination_path": str(getattr(args, "destination_path", ""))}
        return manager.do_move(**call_kwargs)
    if args.action == SUBTREE_REMOVE_ACTION:
        return manager.do_remove(list(args.paths))
    if args.action == SUBTREE_LIST_ACTION:
        return manager.do_list()
    if args.action == SUBTREE_STATUS_ACTION:
        call_kwargs = {
            "filters": list(getattr(args, "paths", [])),
            "print_branch": bool(getattr(args, "print_branch", DEFAULT_PRINT_BRANCH)),
            "show_progress": not bool(getattr(args, "no_progress_bar", False)),
        }
        return manager.do_status(**call_kwargs)

    manager.die(f"Unknown action: {args.action} (use: pull, push, add, track, untrack, move, remove, status, list, help)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
