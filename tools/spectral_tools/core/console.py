"""Console formatting helpers shared across tool scripts."""

from __future__ import annotations

import threading
import sys

from dataclasses import dataclass
from os import environ
from shutil import get_terminal_size

from .constants import NO_COLOR_ENV

PROGRESS_BAR_WIDTH = 28
PROGRESS_FILL_CHAR = "\033[32m█\033[0m"
PROGRESS_EMPTY_CHAR = "."
PROGRESS_WAIT_MESSAGE = "waiting for work"
PROGRESS_HEADER_PREFIX = "Progress"
ERROR_TAG = "ERROR:"
WARNING_TAG = "WARNING:"
INFO_TAG = "INFO:"


@dataclass(slots=True)
class _WorkerProgressState:
    total_tasks: int
    total_share: float
    done_tasks: int = 0
    done_share: float = 0.0
    failed: int = 0
    active: bool = False
    status: str = PROGRESS_WAIT_MESSAGE


class ThreadProgressDisplay:
    """Render stacked per-worker progress bars with one status line each."""

    def __init__(
        self,
        *,
        console: Console,
        title: str,
        worker_totals: list[int],
        worker_shares: list[float] | None = None,
        clear_on_exit: bool = True,
    ) -> None:
        self._console = console
        self._title = title
        if worker_shares is None:
            worker_shares = [float(max(1, total)) for total in worker_totals]
        if len(worker_shares) != len(worker_totals):
            raise ValueError("worker_shares length must match worker_totals")

        self._workers = [
            _WorkerProgressState(total_tasks=max(1, total), total_share=max(1.0, float(share)))
            for total, share in zip(worker_totals, worker_shares, strict=True)
        ]
        self._total_share = sum(worker.total_share for worker in self._workers)
        self._clear_on_exit = clear_on_exit
        self._lock = threading.Lock()
        self._rendered_lines = 0
        self._enabled = bool(sys.stdout.isatty())

    def __enter__(self) -> ThreadProgressDisplay:
        with self._lock:
            self._render_locked()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        with self._lock:
            if self._clear_on_exit:
                self._clear_locked()
            else:
                self._render_locked()

    @staticmethod
    def _bar_text(ratio: float) -> str:
        safe_ratio = min(max(ratio, 0.0), 1.0)
        filled = int(safe_ratio * PROGRESS_BAR_WIDTH)
        empty = PROGRESS_BAR_WIDTH - filled
        return f"[{PROGRESS_FILL_CHAR * filled}{PROGRESS_EMPTY_CHAR * empty}]"

    @staticmethod
    def _ratio_text(done: float, total: float) -> str:
        return f"{done:.2f}/{total:.2f} share"

    def _clip(self, text: str) -> str:
        width = get_terminal_size(fallback=(120, 20)).columns
        max_chars = max(24, width - 6)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def _worker_state_text(self, worker: _WorkerProgressState) -> str:
        if worker.active:
            return self._console.yellow("running")
        if worker.failed > 0:
            return self._console.red("FAILED")
        if worker.done_tasks >= worker.total_tasks:
            return self._console.green("done")
        return self._console.yellow("idle")

    def _render_locked(self) -> None:
        if not self._enabled:
            return

        total_done_share = sum(worker.done_share for worker in self._workers)
        total_ratio = total_done_share / self._total_share if self._total_share else 1.0
        total_percent = int(min(max(total_ratio, 0.0), 1.0) * 100)
        total_ratio_text = self._ratio_text(total_done_share, self._total_share)
        lines = [
            (
                f"{self._console.bold(self._title)}  "
                f"{PROGRESS_HEADER_PREFIX}: {total_ratio_text} "
                f"{self._bar_text(total_ratio)} {total_percent:3d}%"
            )
        ]

        for idx, worker in enumerate(self._workers, start=1):
            worker_ratio = worker.done_share / worker.total_share if worker.total_share else 1.0
            worker_percent = int(min(max(worker_ratio, 0.0), 1.0) * 100)
            worker_ratio_text = self._ratio_text(worker.done_share, worker.total_share)
            lines.append(
                f"  thread-{idx:02d}  {worker_ratio_text} "
                f"{self._bar_text(worker_ratio)} {worker_percent:3d}%  "
                f"{self._worker_state_text(worker)}"
            )
            lines.append(f"    {self._clip(worker.status)}")

        if self._rendered_lines:
            sys.stdout.write(f"\x1b[{self._rendered_lines}F")

        buf = "".join(f"\x1b[2K{line}\n" for line in lines)
        sys.stdout.write(buf)
        sys.stdout.flush()
        self._rendered_lines = len(lines)

    def _clear_locked(self) -> None:
        if not self._enabled or not self._rendered_lines:
            return
        sys.stdout.write(f"\x1b[{self._rendered_lines}F")
        for _ in range(self._rendered_lines):
            # Delete each rendered line so the terminal collapses the region instead of leaving blank rows.
            sys.stdout.write("\x1b[2K\x1b[M")
        sys.stdout.flush()
        self._rendered_lines = 0

    def task_start(self, worker_idx: int, task: str) -> None:
        with self._lock:
            worker = self._workers[worker_idx]
            worker.active = True
            worker.status = task
            self._render_locked()

    def task_done(self, worker_idx: int, status: str, *, success: bool) -> None:
        with self._lock:
            worker = self._workers[worker_idx]
            worker.active = False
            worker.done_tasks = min(worker.total_tasks, worker.done_tasks + 1)
            share_step = worker.total_share / float(worker.total_tasks)
            worker.done_share = min(worker.total_share, worker.done_share + share_step)
            if not success:
                worker.failed += 1
            worker.status = status
            self._render_locked()

    def task_done_with_share(
        self, worker_idx: int, status: str, *, success: bool, share_done: float,
    ) -> None:
        with self._lock:
            worker = self._workers[worker_idx]
            worker.active = False
            worker.done_tasks = min(worker.total_tasks, worker.done_tasks + 1)
            worker.done_share = min(worker.total_share, worker.done_share + max(0.0, share_done))
            if not success:
                worker.failed += 1
            worker.status = status
            self._render_locked()

    def worker_done(self, worker_idx: int, status: str) -> None:
        with self._lock:
            worker = self._workers[worker_idx]
            worker.active = False
            worker.done_tasks = worker.total_tasks
            worker.done_share = worker.total_share
            worker.status = status
            self._render_locked()


@dataclass(slots=True)
class Console:
    use_color: bool = True

    def __post_init__(self) -> None:
        has_tty = bool(sys.stdout.isatty() or sys.stderr.isatty())
        self.use_color = bool(self.use_color and has_tty and environ.get(NO_COLOR_ENV) is None)

    def _style(self, text: str, *codes: str) -> str:
        if not self.use_color or not codes:
            return text
        return f"{''.join(codes)}{text}\033[0m"

    def green(self, text: str) -> str:
        return self._style(text, "\033[32m")

    def red(self, text: str) -> str:
        return self._style(text, "\033[31m")

    def yellow(self, text: str) -> str:
        return self._style(text, "\033[33m")

    def cyan(self, text: str) -> str:
        return self._style(text, "\033[36m")

    def bold(self, text: str) -> str:
        return self._style(text, "\033[1m")

    def italic(self, text: str) -> str:
        return self._style(text, "\033[3m")

    def error_tag(self) -> str:
        return self._style(ERROR_TAG, "\033[1m", "\033[31m")

    def warning_tag(self) -> str:
        return self._style(WARNING_TAG, "\033[1m", "\033[33m")

    def info_tag(self) -> str:
        return self._style(INFO_TAG, "\033[1m", "\033[36m")

    def format_error(self, message: str) -> str:
        return f"{self.error_tag()} {message}"

    def format_warning(self, message: str) -> str:
        return f"{self.warning_tag()} {self.italic(message)}"

    def format_info(self, message: str) -> str:
        return f"{self.info_tag()} {self.italic(message)}"

    def print_error(self, message: str) -> None:
        print(self.format_error(message), file=sys.stderr)

    def print_warning(self, message: str) -> None:
        print(self.format_warning(message), file=sys.stderr)

    def print_info(self, message: str) -> None:
        print(self.format_info(message), file=sys.stderr)

    def thread_progress(
        self, *, title: str, worker_totals: list[int], worker_shares: list[float] | None = None, clear_on_exit: bool = True,
    ) -> ThreadProgressDisplay:
        return ThreadProgressDisplay(
            console=self, title=title, worker_totals=worker_totals, worker_shares=worker_shares, clear_on_exit=clear_on_exit,
        )

    def status(self, value: str) -> str:
        """Map a status string to its canonical colored representation."""
        norm = value.lower()
        if norm == "ok":
            return self.green("ok")
        if norm == "failed":
            return self.red("FAILED")
        if norm in ("skipped", "blocked", "unavailable"):
            return self.yellow(norm)
        return value
