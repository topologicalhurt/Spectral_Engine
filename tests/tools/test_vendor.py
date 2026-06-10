"""Tests for spectral_tools.vendor — manifest SSOT, submodule engine, and
the submodule/subtree kind separation (ADR-0003).

Unit tests build throwaway git repos under tmp_path; live tests assert the
real repo's manifest ⇄ git consistency (simde/xxHash submodules).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from spectral_tools.vendor import manifest, submodules  # noqa: E402

SPEC = manifest.DependencySpec


def _spec(name="dep", kind="submodule", **kw):
    defaults = {"url": f"https://example.com/{name}.git", "path": f"third_party/{name}"}
    defaults.update(kw)
    return SPEC(name=name, kind=kind, **defaults)


def _git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


# --- manifest schema ----------------------------------------------------------


def test_manifest_round_trip_is_deterministic(tmp_path):
    entries = [_spec("b"), _spec("a", kind="subtree", track="pull", ref="dev")]
    manifest.save(tmp_path, entries)
    first = (tmp_path / manifest.MANIFEST_RELPATH).read_text()
    loaded = manifest.load(tmp_path)
    manifest.save(tmp_path, list(loaded.entries))
    assert (tmp_path / manifest.MANIFEST_RELPATH).read_text() == first
    assert [e.name for e in loaded.entries] == ["a", "b"]   # sorted


def test_manifest_rejects_unknown_and_kind_mismatched_keys(tmp_path):
    base = {"kind": "submodule", "url": "https://x.com/r", "path": "third_party/r"}
    cases = [
        ({**base, "bogus": 1}, "unknown key"),
        ({**base, "track": "both"}, "unknown key"),                 # track is subtree-only
        ({**base, "kind": "subtree", "sparse": ["x"]}, "unknown key"),  # sparse is submodule-only
        ({**base, "kind": "subtree", "sync": False}, "track: none"),    # one bypass per kind
        ({**base, "kind": "nonsense"}, "kind must be"),
        ({"kind": "submodule", "url": "https://x.com/r"}, "missing required"),
        ({**base, "path": "../escape"}, "repo-relative"),
        ({**base, "path": "-rf"}, "begin with '-'"),                  # option-injection via path
        ({**base, "url": "not a url"}, "https/http/ssh"),
        ({**base, "url": "--upload-pack=x://y"}, "begin with '-'"),   # option-injection via url
        ({**base, "url": "ext::sh -c id"}, "https/http/ssh"),         # ext:: transport not allowlisted
        ({**base, "ref": "-rf"}, "begin with '-'"),                   # option-injection via ref
        ({**base, "kind": "submodule", "sparse": ["-x"]}, "begin with '-'"),  # via sparse cone
    ]
    import yaml as yaml_mod

    for raw, match in cases:
        doc = {"version": 1, "dependencies": {"r": raw}}
        (tmp_path / "third_party").mkdir(exist_ok=True)
        (tmp_path / manifest.MANIFEST_RELPATH).write_text(yaml_mod.safe_dump(doc))
        with pytest.raises(manifest.ManifestError, match=match):
            manifest.load(tmp_path)


def test_manifest_rejects_duplicate_paths_and_bad_version(tmp_path):
    import yaml as yaml_mod

    (tmp_path / "third_party").mkdir()
    doc = {
        "version": 1,
        "dependencies": {
            "a": {"kind": "submodule", "url": "https://x.com/a", "path": "third_party/same"},
            "b": {"kind": "subtree", "url": "https://x.com/b", "path": "third_party/same"},
        },
    }
    (tmp_path / manifest.MANIFEST_RELPATH).write_text(yaml_mod.safe_dump(doc))
    with pytest.raises(manifest.ManifestError, match="duplicate path"):
        manifest.load(tmp_path)

    (tmp_path / manifest.MANIFEST_RELPATH).write_text(yaml_mod.safe_dump({"version": 99}))
    with pytest.raises(manifest.ManifestError, match="version"):
        manifest.load(tmp_path)


def test_replace_entries_preserves_other_kind(tmp_path):
    manifest.save(tmp_path, [_spec("mod"), _spec("tree", kind="subtree")])
    manifest.replace_entries(
        tmp_path, kind="subtree",
        new_entries=[_spec("newtree", kind="subtree", path="third_party/newtree")],
    )
    loaded = manifest.load(tmp_path)
    assert {e.name for e in loaded.submodules()} == {"mod"}
    assert {e.name for e in loaded.subtrees()} == {"newtree"}
    with pytest.raises(manifest.ManifestError, match="kind=submodule"):
        manifest.replace_entries(tmp_path, kind="subtree", new_entries=[_spec("x")])


def test_save_rejects_duplicate_names(tmp_path):
    # Two specs with the same name (distinct paths) must raise, not silently
    # collapse via the by-name dict — otherwise a cross-kind name collision
    # erases a declaration.
    dup = [_spec("foo"), _spec("foo", kind="subtree", path="third_party/other")]
    with pytest.raises(manifest.ManifestError, match="duplicate dependency name"):
        manifest.save(tmp_path, dup)


def test_subtree_write_cannot_clobber_submodule_via_name_collision(tmp_path):
    # A subtree whose path basename equals an existing submodule name must NOT
    # erase the submodule (ADR-0003 kind separation). The subtree name is now
    # derived from the full path, so 'lib/foo' -> 'lib_foo', not 'foo'.
    from spectral_tools.subtrees.subtree_manager import SubtreeManager

    repo = _git_repo(tmp_path)
    manifest.save(repo, [_spec("foo")])  # submodule named 'foo'
    mgr = SubtreeManager(repo_root=repo, libs_file=repo / manifest.MANIFEST_RELPATH, ignored_paths=[])
    from spectral_tools.subtrees.subtree_models import LibEntry
    mgr._write_entries([LibEntry(raw="", repo="https://x.com/f.git", local_path="lib/foo", branch="main")])
    loaded = manifest.load(repo)
    assert {e.name for e in loaded.submodules()} == {"foo"}, "submodule must survive"
    assert [e.path for e in loaded.subtrees()] == ["lib/foo"]
    assert loaded.subtrees()[0].name == "lib_foo"


# --- submodule engine (tmp repos) ---------------------------------------------


def test_sync_bypasses_sync_false_unless_named(tmp_path):
    repo = _git_repo(tmp_path)
    manifest.save(repo, [_spec("skipme", sync=False)])
    rows = submodules.sync(repo, dry_run=True)
    assert rows[0]["result"] == "bypassed (sync: false)"

    rows = submodules.sync(repo, only_paths=["third_party/skipme"], dry_run=True)
    assert rows[0]["result"] == "ok"
    assert any("submodule add" in a for a in rows[0]["actions"])


def test_sync_rejects_unknown_paths(tmp_path):
    repo = _git_repo(tmp_path)
    manifest.save(repo, [_spec("dep")])
    with pytest.raises(submodules.SubmoduleError, match="not in manifest"):
        submodules.sync(repo, only_paths=["third_party/nope"], dry_run=True)


def test_verify_flags_kind_mixing(tmp_path):
    repo = _git_repo(tmp_path)
    # Declare a SUBTREE at a path git tracks as a submodule -> must be flagged.
    manifest.save(repo, [_spec("mixed", kind="subtree", path="third_party/mixed")])
    (repo / ".gitmodules").write_text(
        '[submodule "mixed"]\n\tpath = third_party/mixed\n\turl = https://x.com/m\n'
    )
    problems = submodules.verify(repo)
    assert any("kinds must not mix" in p for p in problems)


def test_status_surfaces_undeclared_submodules(tmp_path):
    repo = _git_repo(tmp_path)
    manifest.save(repo, [])
    (repo / ".gitmodules").write_text(
        '[submodule "ghost"]\n\tpath = third_party/ghost\n\turl = https://x.com/g\n'
    )
    rows = submodules.status(repo)
    assert rows and rows[0]["name"] == "<undeclared:third_party/ghost>"
    assert any("no manifest entry" in i for i in rows[0]["issues"])


# --- subtree engine over the shared manifest ----------------------------------


def test_subtree_engine_reads_and_writes_manifest_without_touching_submodules(tmp_path):
    from spectral_tools.subtrees.subtree_manager import SubtreeManager

    repo = _git_repo(tmp_path)
    manifest.save(repo, [
        _spec("mod"),
        _spec("tree", kind="subtree", ref="dev", track="pull"),
    ])
    mgr = SubtreeManager(
        repo_root=repo,
        libs_file=repo / manifest.MANIFEST_RELPATH,
        ignored_paths=[],
    )
    lines = mgr.read_libs_file()
    assert lines == ["https://example.com/tree.git, third_party/tree, dev, pull"]

    entries = mgr.validate_entries(lines)
    assert len(entries) == 1 and entries[0].track_mask == "pull"

    # Round-trip through the engine's write path: submodule entry must survive.
    mgr._write_entries(entries)
    loaded = manifest.load(repo)
    assert {e.name for e in loaded.submodules()} == {"mod"}
    assert [e.path for e in loaded.subtrees()] == ["third_party/tree"]
    assert loaded.subtrees()[0].track == "pull"


# --- live checks against the real repo ----------------------------------------


def test_real_repo_manifest_is_consistent_with_git():
    mani = manifest.load(ROOT)
    assert {e.path for e in mani.submodules()} >= {"third_party/simde", "third_party/xxHash"}
    problems = submodules.verify(ROOT, mani)
    assert problems == [], f"manifest ⇄ git inconsistent: {problems}"


def test_real_repo_gitmodules_branches_match_manifest():
    states = submodules.git_submodule_states(ROOT)
    mani = manifest.load(ROOT)
    for entry in mani.submodules():
        state = states[entry.path]
        assert state.initialized, f"{entry.path} not initialized"
        assert state.branch == entry.ref
