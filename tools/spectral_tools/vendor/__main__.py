"""CLI front door for dependency management.

    python -m spectral_tools.vendor list
    python -m spectral_tools.vendor submodule status|verify|sync|add|remove ...
    python -m spectral_tools.vendor subtree <subtree-manager args...>

``subtree`` forwards verbatim to the subtree engine
(``spectral_tools.subtrees.subtree_manager``) so there is exactly one entry
point for vendored-dependency work, while the two kinds keep separate
engines underneath (ADR-0003).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    _tools_root = str(Path(__file__).resolve().parents[2])
    if _tools_root not in sys.path:
        sys.path.insert(0, _tools_root)
    __package__ = "spectral_tools.vendor"  # noqa: A001

from ..core.cli import add_dry_run_flag, add_output_options
from ..core.report_output import render_report, write_report_json
from ..core.utils import find_repo_root
from . import manifest as manifest_mod
from . import submodules
from .manifest import ManifestError
from .submodules import SubmoduleError


def _emit(args: argparse.Namespace, report: dict) -> None:
    if getattr(args, "json_out", None):
        write_report_json(Path(args.json_out), report)
    print(render_report(report, raw=bool(getattr(args, "raw", False)),
                        use_color=not bool(getattr(args, "no_color", False))))


def _report(suite: str, tests: list[dict]) -> dict:
    return {"suite": suite, "tests": tests}


def cmd_list(args: argparse.Namespace, root: Path) -> int:
    mani = manifest_mod.load(root)
    tests = [
        {
            "name": entry.name,
            "status": "ok",
            "summary": f"{entry.kind} @ {entry.ref}" + ("" if entry.sync else " [sync: false]"),
            "details": entry.as_dict(),
        }
        for entry in mani.entries
    ]
    _emit(args, _report("vendor-manifest", tests))
    return 0


def cmd_submodule(args: argparse.Namespace, root: Path) -> int:
    if args.sub_action == "status":
        rows = submodules.status(root)
        tests = [
            {
                "name": row["name"],
                "status": "ok" if not row["issues"] else "FAILED",
                "summary": (row["sha"] or "absent")[:12] + ("" if not row["issues"] else f" — {len(row['issues'])} issue(s)"),
                "details": row,
            }
            for row in rows
        ]
        _emit(args, _report("vendor-submodule-status", tests))
        return 1 if any(t["status"] == "FAILED" for t in tests) else 0

    if args.sub_action == "verify":
        problems = submodules.verify(root)
        tests = [{"name": "consistency", "status": "ok" if not problems else "FAILED",
                  "summary": "manifest ⇄ git consistent" if not problems else f"{len(problems)} problem(s)",
                  "details": {"problems": problems}}]
        _emit(args, _report("vendor-submodule-verify", tests))
        return 1 if problems else 0

    if args.sub_action == "sync":
        rows = submodules.sync(root, only_paths=args.paths or None, dry_run=args.dry_run)
        tests = [{"name": row["name"], "status": "ok",
                  "summary": row["result"], "details": row} for row in rows]
        _emit(args, _report("vendor-submodule-sync", tests))
        return 0

    if args.sub_action == "add":
        entry = submodules.add(root, name=args.name, url=args.url, path=args.path,
                               ref=args.ref, sparse=tuple(args.sparse or ()),
                               dry_run=args.dry_run)
        _emit(args, _report("vendor-submodule-add", [
            {"name": entry.name, "status": "ok",
             "summary": f"{'planned' if args.dry_run else 'added'} {entry.path} @ {entry.ref}",
             "details": entry.as_dict()}]))
        return 0

    if args.sub_action == "remove":
        log = submodules.remove(root, args.path, dry_run=args.dry_run)
        _emit(args, _report("vendor-submodule-remove", [
            {"name": args.path, "status": "ok",
             "summary": "planned" if args.dry_run else "removed",
             "details": {"steps": log}}]))
        return 0

    raise AssertionError(f"unhandled sub_action {args.sub_action}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m spectral_tools.vendor",
        description="Vendored-dependency management (manifest: third_party/libs.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="Show every manifest entry (both kinds)")
    add_output_options(list_p)

    submodule_p = sub.add_parser("submodule", help="Git-submodule operations")
    sm_sub = submodule_p.add_subparsers(dest="sub_action", required=True)

    sm_status = sm_sub.add_parser("status", help="Per-entry state + drift vs manifest")
    add_output_options(sm_status)

    sm_verify = sm_sub.add_parser("verify", help="Strict manifest ⇄ git consistency (kind-separation gate)")
    add_output_options(sm_verify)

    sm_sync = sm_sub.add_parser("sync", help="Make git match the manifest (add/init/sparse)")
    add_dry_run_flag(sm_sync)
    sm_sync.add_argument("paths", nargs="*", help="Limit to these paths (also overrides sync:false bypass)")
    add_output_options(sm_sync)

    sm_add = sm_sub.add_parser("add", help="Declare + materialize a new submodule")
    sm_add.add_argument("--name", required=True)
    sm_add.add_argument("--url", required=True)
    sm_add.add_argument("--path", required=True)
    sm_add.add_argument("--ref", default=manifest_mod.DEFAULT_REF)
    sm_add.add_argument("--sparse", action="append", default=[],
                        help="sparse-checkout cone (repeatable)")
    add_dry_run_flag(sm_add)
    add_output_options(sm_add)

    sm_remove = sm_sub.add_parser("remove", help="Deinit + git rm + manifest update")
    sm_remove.add_argument("path")
    add_dry_run_flag(sm_remove)
    add_output_options(sm_remove)

    sub.add_parser("subtree", add_help=False,
                   help="Forward to the subtree engine (subtree_manager args verbatim)")

    return parser


def main(argv: list[str] | None = None) -> int:
    args_raw = list(sys.argv[1:] if argv is None else argv)

    # `vendor subtree ...` forwards verbatim — argparse must not eat its args.
    if args_raw and args_raw[0] == "subtree":
        from ..subtrees.subtree_manager import main as subtree_main

        return subtree_main(args_raw[1:])

    args = build_parser().parse_args(args_raw)
    root = find_repo_root(Path(__file__))
    try:
        if args.command == "list":
            return cmd_list(args, root)
        if args.command == "submodule":
            return cmd_submodule(args, root)
    except (ManifestError, SubmoduleError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
