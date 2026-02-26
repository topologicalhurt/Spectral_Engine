"""Report rendering helpers for subtree tooling."""

from __future__ import annotations

from .constants import short_ref
from .report_output import render_report
from .subtree_models import DryRunStepReport, StatusEntryReport


def render_dry_run_report(
    *,
    action: str,
    filters: list[str],
    head_start: str,
    steps: list[DryRunStepReport],
) -> str:
    tests: list[dict[str, object]] = []
    for step in steps:
        tests.append(
            {
                "name": step.path,
                "status": step.status.value,
                "summary": (
                    f"{step.operation} {step.commit_before} -> {step.commit_after} "
                    f"remote={short_ref(step.remote_head)} "
                    f"delta={step.delta_commits if step.delta_commits is not None else 'unknown'}"
                ),
                "details": {
                    "repo": step.repo,
                    "branch": step.branch,
                    "operation": step.operation,
                    "planned_commit": {
                        "before": step.commit_before,
                        "after": step.commit_after,
                        "message": step.commit_message,
                    },
                    "projected_tree": list(step.projected_tree),
                    "sync": {
                        "remote_head": short_ref(step.remote_head) if step.remote_head else None,
                        "delta_commits": step.delta_commits,
                    },
                    "note": step.note,
                },
            }
        )

    report = {
        "suite": "subtree_update",
        "context": {
            "action": action,
            "mode": "dry-run",
            "head_start": head_start,
            "filters": filters,
            "steps": len(steps),
        },
        "tests": tests,
    }
    return render_report(report, raw=False, use_color=True)


def render_status_report(*, filters: list[str], rows: list[StatusEntryReport]) -> str:
    tests: list[dict[str, object]] = []
    for row in rows:
        tests.append(
            {
                "name": row.path,
                "status": row.status.value,
                "summary": (
                    f"local={short_ref(row.local_split)} "
                    f"remote={short_ref(row.remote_head)} "
                    f"ff={row.fast_forward_commits if row.fast_forward_commits is not None else 'unknown'}"
                ),
                "details": {
                    "repo": row.repo,
                    "branch": row.branch,
                    "local_split": short_ref(row.local_split) if row.local_split else None,
                    "remote_head": short_ref(row.remote_head) if row.remote_head else None,
                    "fast_forward_commits": row.fast_forward_commits,
                    "note": row.note,
                },
            }
        )

    report = {
        "suite": "subtree_status",
        "context": {
            "action": "status",
            "filters": filters,
            "entries": len(rows),
        },
        "tests": tests,
    }
    return render_report(report, raw=False, use_color=True)
