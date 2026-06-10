"""Third-party dependency management (ADR-0003).

One YAML manifest (``third_party/libs.yaml``) declares every vendored
dependency with an explicit ``kind`` — ``submodule`` or ``subtree`` — and the
two kinds are managed by strictly separated engines:

- ``manifest``    — the typed, validated SSOT loader/serializer.
- ``submodules``  — git-submodule operations (state derived from git, never
  duplicated; manifest ⇄ .gitmodules verification).
- the subtree engine stays in ``spectral_tools.subtrees`` and consumes only
  ``kind: subtree`` entries through this manifest.

CLI front door: ``python -m spectral_tools.vendor`` (``submodule ...``,
``subtree ...``, ``list``).
"""

__all__ = [
    "manifest",
    "submodules",
]
