"""Git command abstractions for subtree tooling."""

from __future__ import annotations

import threading

from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

from .git_models import GIT_DEFAULT_TIMEOUT_SEC, GitCommandResult, RefLookupResult
from .process import run


class GitOps:
    """Low-level git facade with mutation serialization and typed results."""

    def __init__(self, repo_root: Path, *, default_timeout_sec: int = GIT_DEFAULT_TIMEOUT_SEC) -> None:
        """Initialise with the repository root.

        .. note::

            ``_mutation_lock`` is a **process-local** ``threading.Lock``.
            It serializes repo-mutating git commands within a single Python
            process but does **not** protect against concurrent access from
            multiple processes (e.g. parallel CI jobs sharing a worktree).
        """
        self.repo_root = repo_root.resolve()
        self.default_timeout_sec = default_timeout_sec
        self._mutation_lock = threading.Lock()

    @staticmethod
    def _new_lookup_result(*, ref: str | None, note: str = "") -> RefLookupResult:
        return RefLookupResult(ref=ref, note=note)

    @staticmethod
    def _new_git_result(*, returncode: int, stdout: str, stderr: str) -> GitCommandResult:
        return GitCommandResult(returncode=returncode, stdout=stdout, stderr=stderr)

    def run(
        self,
        args: list[str],
        *,
        timeout_sec: int | None = None,
        mutates_repo: bool = False,
    ) -> GitCommandResult:
        """Run a git command rooted at ``repo_root``."""
        timeout = self.default_timeout_sec if timeout_sec is None else timeout_sec
        lock_ctx = self._mutation_lock if mutates_repo else nullcontext()
        with lock_ctx:
            result = run(
                ["git", "-C", str(self.repo_root), *args],
                cwd=self.repo_root,
                timeout_sec=timeout,
                check=False,
            )
        return self._new_git_result(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)

    def run_mutating(self, args: list[str], *, timeout_sec: int | None = None) -> GitCommandResult:
        """Run a repo-mutating git command under a process-local lock."""
        return self.run(args, timeout_sec=timeout_sec, mutates_repo=True)

    def rev_parse(self, ref: str, *, short: int | None = None) -> RefLookupResult:
        """Resolve a git ref to a commit hash and return note text on failure."""
        args = ["rev-parse", f"--short={short}"] if short is not None else ["rev-parse"]
        args.append(ref)
        result = self.run(args)
        if result.returncode != 0:
            note = f"rev-parse failed for {ref}: {result.stderr.strip() or 'unknown error'}"
            return self._new_lookup_result(ref=None, note=note)
        resolved = result.stdout.strip()
        if not resolved:
            return self._new_lookup_result(ref=None, note=f"rev-parse returned no output for {ref}")
        return self._new_lookup_result(ref=resolved)

    @staticmethod
    def canonical_head_ref(branch: str) -> str:
        """Normalize a branch name to a full ``refs/heads/...`` ref.

        Rejects names invalid per ``git check-ref-format``.
        """
        branch_name = branch.strip()
        if not branch_name:
            raise ValueError("empty branch name")
        # Characters and sequences forbidden by git check-ref-format.
        _deny_chars = (" ", "\n", "\t", "~", "^", ":", "?", "*", "[", "\\")
        if any(ch in branch_name for ch in _deny_chars):
            raise ValueError(f"invalid branch name: {branch_name!r}")
        if ".." in branch_name or "@{" in branch_name:
            raise ValueError(f"invalid branch name: {branch_name!r}")
        if branch_name.startswith("-") or branch_name.endswith(".") or branch_name.endswith(".lock"):
            raise ValueError(f"invalid branch name: {branch_name!r}")
        if branch_name.startswith("refs/heads/"):
            return branch_name
        return f"refs/heads/{branch_name.lstrip('/')}"

    def ls_remote_head(self, repo: str, branch: str) -> RefLookupResult:
        """Resolve exact remote head for ``refs/heads/<branch>`` without history fetch."""
        exact_ref = self.canonical_head_ref(branch)
        result = self.run(["ls-remote", "--heads", "--exit-code", repo, exact_ref])
        if result.returncode != 0:
            note = f"remote head lookup failed: {result.stderr.strip() or 'unknown error'}"
            return self._new_lookup_result(ref=None, note=note)

        match = next(
            (
                fields[0].strip()
                for line in result.stdout.splitlines()
                if (fields := line.split()) and len(fields) >= 2 and fields[1].strip() == exact_ref
            ),
            "",
        )
        if not match:
            note = (
                f"remote head lookup returned no exact match for {exact_ref} "
                f"(ls-remote returned {len(result.stdout.splitlines())} ref(s))"
            )
            return self._new_lookup_result(ref=None, note=note)
        return self._new_lookup_result(ref=match)

    def current_head(self) -> str:
        """Resolve current HEAD in short form, or ``HEAD`` when unavailable."""
        resolved = self.rev_parse("HEAD", short=12)
        return resolved.ref if resolved.ref else "HEAD"

    def commit_exists(self, commit: str) -> bool:
        """Return whether ``commit`` exists locally as a commit object."""
        if not commit:
            return False
        return self.run(["cat-file", "-e", f"{commit}^{{commit}}"]).returncode == 0

    def is_ancestor(self, older: str, newer: str) -> bool | None:
        """Return ancestry relation for ``older`` -> ``newer``."""
        result = self.run(["merge-base", "--is-ancestor", older, newer])
        return (not result.returncode) if result.returncode in {0, 1} else None

    def count_commits(self, older: str, newer: str) -> int | None:
        """Return commit count in ``older..newer`` or ``None`` when unavailable."""
        result = self.run(["rev-list", "--count", f"{older}..{newer}"])
        if result.returncode != 0:
            return None
        try:
            return int(result.stdout.strip())
        except ValueError:
            return None

    def is_git_repo(self) -> bool:
        """Return whether ``repo_root`` points to a git repository."""
        return self.run(["rev-parse", "--git-dir"]).returncode == 0

    def find_stash_ref_by_marker(self, marker: str) -> str:
        """Return stash ref whose subject carries ``marker`` as the message tail.

        ``git stash push -m <marker>`` typically stores a subject like:
        ``On <branch>: <marker>`` (or ``WIP on <branch>: <marker>``),
        so exact equality with ``marker`` is not reliable.
        """
        result = self.run(["stash", "list", "--format=%gd%x00%gs"])
        if result.returncode != 0:
            return ""
        for raw in result.stdout.splitlines():
            ref, _, message = raw.partition("\x00")
            subject = message.strip()
            tail = subject.rsplit(": ", 1)[-1]
            if ref and (subject == marker or tail == marker):
                return ref.strip()
        return ""

    @staticmethod
    def exclude_specs(paths: Iterable[str]) -> list[str]:
        """Build git pathspec exclusions for repo-relative paths."""
        return [f":(exclude){rel}" for rel in paths]

    def has_changes(self, *, ignored_rel: Iterable[str] = ()) -> bool:
        """Return whether tracked or untracked changes exist outside ignored paths."""
        ignored = set(ignored_rel)
        excludes = self.exclude_specs(ignored)
        diff_checks = (
            ["diff", "--cached", "--quiet", "--", ".", *excludes],
            ["diff", "--quiet", "--", ".", *excludes],
        )
        if any(self.run(cmd).returncode != 0 for cmd in diff_checks):
            return True

        untracked = self.run(["ls-files", "--others", "--exclude-standard"])
        if untracked.returncode != 0:
            return True
        return any(
            (line := raw.strip()) and line not in ignored
            for raw in untracked.stdout.splitlines()
        )

    def collect_dirty_paths(self) -> list[str]:
        """Collect modified/untracked paths from porcelain status output."""
        result = self.run(["status", "--porcelain=v1", "-z"])
        if result.returncode != 0 or not result.stdout:
            return []

        dirty: list[str] = []
        records = iter(result.stdout.split("\x00"))
        for rec in records:
            if not rec:
                continue
            xy = rec[:2]
            path = rec[3:]
            if path:
                dirty.append(path)
            if xy[:1] in {"R", "C"} and (extra := next(records, "")):
                dirty.append(extra)
        return dirty


class GitSubtreeOps(GitOps):
    """Subtree-specialized operations layered over :class:`GitOps`."""

    def run_subtree(self, args: list[str], *, no_cache: bool = False) -> GitCommandResult:
        """Run a git subtree command.

        When ``no_cache`` is enabled, fetch negotiation is switched to ``noop`` so
        subtree fetches advertise fewer local commits and tend to re-download data.
        """
        if no_cache:
            return self.run_mutating(["-c", "fetch.negotiationAlgorithm=noop", "subtree", *args])
        return self.run_mutating(["subtree", *args])
