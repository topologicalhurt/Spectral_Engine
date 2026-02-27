# ADR-0001: `spectral_tools` Toolchain Architecture

- Status: Accepted
- Date: 2026-02-27
- Decision makers: maintainer (topologicalhurt,
csin0659@uni.sydney.edu.au)

## Context

`tools/spectral_tools` now owns multiple workflows that affect build correctness:

- code generation (LUTs, resource hashes),
- benchmark/performance orchestration,
- subtree synchronization,
- shared console/process/git helpers.

The package must stay reliable enough for build-system dependency usage, while
remaining maintainable as new workflows (including possible GUI supervision) are
added.

## Decision

Adopt a layered package architecture with explicit runtime boundaries.

1. `spectral_tools.core`
Defines stable shared primitives: process execution, git wrappers, constants,
console/report formatting, and common utilities.

2. Domain packages (`generators`, `performance`, `subtrees`, `testing`)
Own feature-specific orchestration and models, but depend on `core` for
shared behavior instead of duplicating logic.

3. Build-time first for generators
`generators` remains deterministic and reproducible by default. Generator logic
must not require GUI/runtime state and should be callable from CLI/module entry
points with explicit parameters.

4. Runtime dispatch boundary
If a desktop supervisor/GUI needs dynamic generation, it should call tool
entrypoints out-of-process (subprocess/module invocation) or through a narrow,
versioned adapter layer. Avoid ad-hoc in-process coupling between the engine and
tool internals.

5. Shared algorithm rule
For logic that must remain bit-for-bit equivalent with C (for example hashing
or canonicalization), centralize authoritative behavior and verify parity in CI.
Prefer binding/shared-library reuse when duplication risk is high.

## Consequences

Positive:

- lower drift risk across scripts,
- clearer dependency direction,
- safer refactoring path for threading/git logic,
- easier GUI integration later without contaminating build-only code paths.

Tradeoffs:

- stricter boundaries add small upfront design overhead,
- some helper placement decisions require discipline (core vs domain package).

## Alternatives considered

1. Keep scripts mostly standalone
Rejected: too much duplication and consistency risk in a build-critical toolchain.

2. Allow direct runtime imports of generator internals from arbitrary supervisors
Rejected: creates hidden coupling and increases regression surface.

3. Re-implement all shared C logic independently in Python
Rejected: high long-term parity risk without strong verification contracts.

## Follow-up guidance

- New shared helpers go into `core` only when they are domain-agnostic.
- Domain packages should expose small typed interfaces and avoid leaking
  implementation details across package boundaries.
- Any future dynamic-dispatch package should be treated as an adapter layer, not
  a place for duplicated generator logic.
