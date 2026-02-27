"""Report rendering helpers for subtree tooling."""

from __future__ import annotations

from ..core.utils import short_ref
from ..core.report_output import render_report
from .subtree_models import DryRunStepReport, StatusEntryReport


def render_dry_run_report(
    *,
    action: str,
    head_start: str,
    steps: list[DryRunStepReport],
) -> str:
    tests: list[dict[str, object]] = []
    for step in steps:
        range_text = (
            f"{short_ref(step.local_split)}->{short_ref(step.remote_head)}"
            if step.local_split and step.remote_head
            else "unknown"
        )
        tests.append(
            {
                "name": step.path,
                "status": step.status.value,
                "summary": (
                    f"{step.operation} {step.commit_before} -> {step.commit_after} "
                    f"local_commits={step.commit_count} "
                    f"subtree_range={range_text} "
                    f"remote={short_ref(step.remote_head)} "
                    f"delta={step.delta_commits if step.delta_commits is not None else 'unknown'}"
                ),
                "details": {
                    "repo": step.repo,
                    "branch": step.branch,
                    "operation": step.operation,
                    "planned_commit": {
                        "before": step.commit_before,
                        "squash": step.squash_commit,
                        "after": step.commit_after,
                        "count": step.commit_count,
                        "message": step.commit_message,
                    },
                    "projected_tree": list(step.projected_tree),
                    "projected_graph": list(step.projected_graph),
                    "sync": {
                        "local_split": short_ref(step.local_split) if step.local_split else None,
                        "remote_head": short_ref(step.remote_head) if step.remote_head else None,
                        "delta_commits": step.delta_commits,
                    },
                    "note": step.note,
                },
            }
        )

    context_kwargs = {
        "action": action,
        "mode": "dry-run",
        "head_start": head_start,
        "steps": len(steps),
    }

    report = {
        "suite": "subtree_update",
        "context": context_kwargs,
        "tests": tests,
    }
    return render_report(report, raw=False, use_color=True)


def render_status_report(*, rows: list[StatusEntryReport]) -> str:
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

    context_kwargs = {"action": "status", "entries": len(rows)}

    report = {
        "suite": "subtree_status",
        "context": context_kwargs,
        "tests": tests,
    }
    return render_report(report, raw=False, use_color=True)
