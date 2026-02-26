"""Constants and data models for subtree tooling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

DEFAULT_BRANCH = "main"
SUBTREE_PULL_ACTION = "pull"
SUBTREE_PUSH_ACTION = "push"
SUBTREE_REMOVE_ACTION = "remove"
SUBTREE_LIST_ACTION = "list"
SUBTREE_STATUS_ACTION = "status"
SUBTREE_HELP_ACTION = "help"
REMOVE_ALIAS = "rm"
LIST_ALIASES = {"list", "ls"}
STATUS_ALIASES = {"status"}
HELP_ALIASES = {"help", "-h", "--help"}
LIBS_RELATIVE_PATH = Path("subtrees/libs.txt")
UPDATE_PY_RELATIVE_PATH = Path("tools/subtree_update.py")
CLI_PROG = UPDATE_PY_RELATIVE_PATH.name
DEFAULT_PRINT_BRANCH = False
DRY_RUN_REF_PREFIX = "dryrun"
SUBTREE_SPLIT_TRAILER = "git-subtree-split:"
SUBTREE_DIR_TRAILER = "git-subtree-dir:"
COUNT_FETCH_DEPTH = 512
COUNT_FETCH_FILTER = "tree:0"
ALLOW_SUBTREE_SPLIT_FALLBACK = False
MAX_PARALLEL_REMOTE_QUERIES = 8


class RunResult(str, Enum):
    OK = "ok"
    FAILED = "FAILED"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class LibEntry:
    raw: str
    repo: str
    local_path: str
    branch: str


@dataclass
class Summary:
    ok_count: int = 0
    fail_count: int = 0
    skip_count: int = 0
    paths: list[str] = field(default_factory=list)
    results: list[RunResult] = field(default_factory=list)

    def record(self, path: str, result: RunResult) -> None:
        if result is RunResult.OK:
            self.ok_count += 1
        elif result is RunResult.FAILED:
            self.fail_count += 1
        else:
            self.skip_count += 1
        self.paths.append(path)
        self.results.append(result)

    def record_ok(self, path: str) -> None:
        self.record(path, RunResult.OK)

    def record_fail(self, path: str) -> None:
        self.record(path, RunResult.FAILED)

    def record_skip(self, path: str) -> None:
        self.record(path, RunResult.SKIPPED)


@dataclass(frozen=True)
class SyncProbe:
    local_split: str | None
    remote_head: str | None
    pull_commits: int | None
    push_commits: int | None
    note: str


@dataclass(frozen=True)
class DryRunStepReport:
    path: str
    repo: str
    branch: str
    operation: str
    status: RunResult
    commit_before: str
    commit_after: str
    commit_message: str
    projected_tree: tuple[str, ...]
    remote_head: str | None
    delta_commits: int | None
    note: str


@dataclass(frozen=True)
class StatusEntryReport:
    path: str
    repo: str
    branch: str
    status: RunResult
    local_split: str | None
    remote_head: str | None
    fast_forward_commits: int | None
    note: str
