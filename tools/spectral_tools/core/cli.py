"""Shared CLI argument helpers for spectral_tools entrypoints.

Composable functions that register common argument groups onto an existing
``ArgumentParser`` or subparser.  Each function mutates the parser in place
(standard argparse pattern).
"""

from __future__ import annotations

import argparse
from typing import Sequence


def add_dry_run_flag(
    parser: argparse.ArgumentParser,
    *,
    short: str = "-n",
    help_text: str = "Print commands logically without executing",
) -> None:
    """Add a ``--dry-run`` boolean flag with a configurable short form."""
    parser.add_argument(short, "--dry-run", action="store_true", help=help_text)


def add_output_options(
    parser: argparse.ArgumentParser,
    *,
    default_json_output: str | None = None,
) -> None:
    """Add ``--no-color``, ``--raw``, and ``--json-out`` output formatting options."""
    parser.add_argument(
        "-o", "--json-out",
        default=default_json_output,
        help="Write JSON report to this file",
    )
    parser.add_argument(
        "-r", "--raw",
        action="store_true",
        help="Print raw JSON report to stdout",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors for pretty output",
    )


def add_print_branch_flag(
    parser: argparse.ArgumentParser,
    *,
    default: bool = False,
) -> None:
    """Add ``-P / --print-branch`` flag for branch/tree view rendering."""
    parser.add_argument(
        "-P", "--print-branch",
        action="store_true",
        default=default,
        help="Render per-item status in branch/tree view",
    )


def add_progress_bar_flag(parser: argparse.ArgumentParser) -> None:
    """Add ``--no-progress-bar`` flag to disable live progress display."""
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable live progress bar display",
    )
