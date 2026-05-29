# Patch notes — Pass 146: wire load-path segment validation; add rejection test

## Problem

`spectral_arm32_validate_segment_data` (payload overflow, monotonic ordering,
`output_len` bound, simultaneous-active bound, chirp support) was defined but
never called (`-Wunused-function`): `spectral_arm32_load` did only the chirp
check, so the segment ingress boundary skipped its own validation
(KERNEL_PATCHING_GUIDELINES §2: validate at file/cache load).

## Change

Replace `load`'s chirp-only loop with a call to
`spectral_arm32_validate_segment_data` (a superset — it already includes the
chirp check). Invalid segment data is now rejected at load, and the validation
function runs for the first time.

Add `tests/arm_core` cases: a segment starting beyond `output_len`, and an
out-of-order pair, must both be rejected by `load`.

## Verification

`ctest` green: the valid single-tone still loads and renders (so `validate`
accepts well-formed data), and both invalid cases are rejected. The
`-Wunused-function` warning is gone in the `arm_core_test` and `simulate` builds.
Sim oracle unchanged (the sim never calls `load`).
