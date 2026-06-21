"""Constants and data models for subtree tooling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..core.git_models import GIT_DEFAULT_TIMEOUT_SEC, GitCommandResult, RefLookupResult

DEFAULT_BRANCH = "main"
SUBTREE_PULL_ACTION = "pull"
SUBTREE_PUSH_ACTION = "push"
SUBTREE_ADD_ACTION = "add"
SUBTREE_TRACK_ACTION = "track"
SUBTREE_UNTRACK_ACTION = "untrack"
SUBTREE_REMOVE_ACTION = "remove"
SUBTREE_MOVE_ACTION = "move"
SUBTREE_LIST_ACTION = "list"
SUBTREE_STATUS_ACTION = "status"
SUBTREE_HELP_ACTION = "help"
REMOVE_ALIAS = "rm"
MOVE_ALIAS = "mv"
LIST_ALIASES = {"list", "ls"}
STATUS_ALIASES = {"status"}
HELP_ALIASES = {"help", "-h", "--help"}
TRACK_MASK_BOTH = "both"
TRACK_MASK_PULL = "pull"
TRACK_MASK_PUSH = "push"
TRACK_MASK_NONE = "none"
TRACK_MASK_VALUES = {TRACK_MASK_BOTH, TRACK_MASK_PULL, TRACK_MASK_PUSH, TRACK_MASK_NONE}
TRACK_UPDATE_MASK_VALUES = {TRACK_MASK_BOTH, TRACK_MASK_PULL, TRACK_MASK_PUSH}
LIBS_RELATIVE_PATH = Path("third_party/libs.yaml")
UPDATE_PY_RELATIVE_PATH = Path("tools/spectral_tools/subtrees/subtree_manager.py")
CLI_PROG = "python -m spectral_tools.subtrees.subtree_manager"
DEFAULT_PRINT_BRANCH = False
DRY_RUN_REF_PREFIX = "dryrun"
SUBTREE_SPLIT_TRAILER = "git-subtree-split:"
SUBTREE_DIR_TRAILER = "git-subtree-dir:"
COUNT_FETCH_DEPTH = 512
COUNT_FETCH_FILTER = "tree:0"
ALLOW_SUBTREE_SPLIT_FALLBACK = False
MAX_PARALLEL_REMOTE_QUERIES = 8
LOOKUP_SHARE_KIB_PER_UNIT = 32768  # 32 MiB
LOOKUP_SHARE_DEFAULT = 1.0
LOOKUP_SHARE_TIMEOUT_SEC = 10

TRACK_MASK_TO_BITS: dict[str, tuple[bool, bool]] = {
    TRACK_MASK_BOTH: (True, True),
    TRACK_MASK_PULL: (True, False),
    TRACK_MASK_PUSH: (False, True),
    TRACK_MASK_NONE: (False, False),
}
TRACK_BITS_TO_MASK: dict[tuple[bool, bool], str] = {
    bits: mask for mask, bits in TRACK_MASK_TO_BITS.items()
}


def normalize_path(path: str) -> str:
    """Strip leading ``./`` and trailing ``/`` from a subtree path."""
    path = path.strip()
    while path.startswith("./"):
        path = path[2:]
    return path.rstrip("/")


def normalize_track_mask(value: str | None) -> str:
    """Normalize a raw mask string to one of the canonical mask constants."""
    if not value:
        return TRACK_MASK_BOTH
    normalized = value.strip().lower()
    if normalized in {"all", "tracked"}:
        return TRACK_MASK_BOTH
    if normalized in TRACK_MASK_VALUES:
        return normalized
    raise ValueError(f"invalid track mask '{value}' (expected one of: both, pull, push, none)")


def mask_bits(mask: str) -> tuple[bool, bool]:
    """Return ``(pull_enabled, push_enabled)`` for *mask*."""
    return TRACK_MASK_TO_BITS[normalize_track_mask(mask)]


def bits_to_mask(*, pull_enabled: bool, push_enabled: bool) -> str:
    """Return the canonical mask string for a pair of enable bits."""
    return TRACK_BITS_TO_MASK[(pull_enabled, push_enabled)]


def apply_track_mask(current_mask: str, update_mask: str, *, track: bool) -> str:
    """Merge *update_mask* into *current_mask* (``track=True`` adds, ``False`` removes)."""
    pull_enabled, push_enabled = mask_bits(current_mask)
    update_pull, update_push = mask_bits(update_mask)
    if track:
        pull_enabled = pull_enabled or update_pull
        push_enabled = push_enabled or update_push
    else:
        pull_enabled = pull_enabled and not update_pull
        push_enabled = push_enabled and not update_push
    return bits_to_mask(pull_enabled=pull_enabled, push_enabled=push_enabled)


def entry_allows(entry: "LibEntry", action: str) -> bool:
    """Return whether *entry*'s track mask permits *action*."""
    pull_enabled, push_enabled = mask_bits(entry.track_mask)
    if action not in {SUBTREE_PULL_ACTION, SUBTREE_PUSH_ACTION}:
        return True
    return pull_enabled if action == SUBTREE_PULL_ACTION else push_enabled


def entry_to_line(entry: "LibEntry") -> str:
    """Render a :class:`LibEntry` in the engine-internal legacy line format."""
    base = f"{entry.repo}, {entry.local_path}, {entry.branch}"
    return base if entry.track_mask == TRACK_MASK_BOTH else f"{base}, {entry.track_mask}"


def dry_run_ref(step_index: int) -> str:
    """Build a synthetic ref label for a dry-run commit step."""
    return f"{DRY_RUN_REF_PREFIX}-{step_index:03d}"


class RunResult(str, Enum):
    OK = "ok"
    FAILED = "FAILED"
    SKIPPED = "skipped"


RepoBranchKey = tuple[str, str]
RemoteHeadCache = dict[RepoBranchKey, RefLookupResult]
LocalSplitCache = dict[str, RefLookupResult]
CountFetchCache = dict[RepoBranchKey, RefLookupResult]


@dataclass(frozen=True)
class LibEntry:
    raw: str
    repo: str
    local_path: str
    branch: str
    track_mask: str = TRACK_MASK_BOTH


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
    local_split: str | None
    remote_head: str | None
    delta_commits: int | None
    note: str
    squash_commit: str | None = None
    commit_count: int = 0
    projected_graph: tuple[str, ...] = ()


@dataclass(slots=True)
class OperationState:
    dry_run: bool
    summary: Summary
    dry_steps: list[DryRunStepReport]
    projected_tree: list[str]


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
