# Patch notes — Pass 138: fix self-referential analysis path-decision struct

## Problem

`SpectralAnalysisPathDecision` (spectral_analysis_internal.h) declared a member
`SpectralAnalysisPathDecision path;` — a by-value instance of the struct inside
its own definition — where the path flag `int use_fused_path;` belongs. The line
was copy-pasted from `SpectralAnalysisShape`, which legitimately contains a
`SpectralAnalysisPathDecision path;` member.

Every translation unit that compiles `spectral_analysis.c` (desktop, simulate,
and the embedded analysis profiles) failed with `no member named 'use_fused_path'`
at the five `decision.use_fused_path` / `shape.path.use_fused_path` sites. The
branch did not build, so there was no oracle for any subsequent work.

## Change

Restore `int use_fused_path;` as the path-decision flag field. No logic change;
`spectral_analysis.c` already reads and writes `use_fused_path` and passes it to
`spectral_analysis_path_name(int)`. This only repairs the struct to match its
established contract.

`make simulate` now builds clean (target `spectral_arm64_native_simulation`).

## Why minimal

Single-field restoration. Found incidentally while standing up the Campaign-2
ARM oracle (ULTRAPLAN Phase A). It is exactly the self-reference / copy-paste
defect class the Phase C adversarial sweep targets; logged here rather than
deferred because it blocked all builds.
