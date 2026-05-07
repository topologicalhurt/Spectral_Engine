# Core audit pass 41: segment loop derived-scalar contract

## Summary

Pass 41 hardens the per-segment synthesis loop contract.

Pass 39 proved that `SynthParams` derived from `stretch` and `pitch` are finite.
The remaining hot-loop issue is that each segment combines those parameters with
segment-specific fields:

```text
alpha = omega * pitch_factor * inv_stretch
beta  = df    * pitch_factor * inv_stretch_sq
d_amp = da    * inv_stretch
```

Finite inputs do not guarantee finite products.

## Bug

`segment_loop_params_init()` validated raw segment fields and raw time window
shape, but accepted derived loop scalars without checking them.

A segment with finite but very large `omega`, `df`, or `da` could overflow
`alpha`, `beta`, or `d_amp` to Inf. That invalid state then entered CPU/native
hot loops and wavetable/timbre callbacks.

The same problem exists at the endpoint formulas:

```text
phase_at(phase, alpha, beta, last_sample)
amp_at(amp, d_amp, last_sample)
```

The loop should not be marked valid if endpoint phase/amplitude is already
non-finite.

## Fix

`segment_loop_params_init()` now derives `alpha`, `beta`, and `d_amp` into
temporaries, validates each is finite, and checks endpoint phase/amplitude
before marking the segment loop valid.

Zero-length post-stretch segments are rejected instead of producing a valid loop
with no samples.

## Reviewer Walkthrough

1. Raw segment finiteness checks remain.
2. Stretched start/length bounds remain.
3. `alpha`, `beta`, and `d_amp` are computed before assigning into `lp`.
4. Any non-finite derived scalar rejects the segment.
5. The function checks `phase_at(..., 0)`, `phase_at(..., last)`,
   `amp_at(..., 0)`, and `amp_at(..., last)`.
6. Only after those checks does it populate `SegmentLoopParams` and set
   `valid = 1`.

## Why this is critical

The synthesis hot loops consume derived phase and amplitude increments, not just
raw segment fields. A single non-finite derived scalar can poison an output
buffer or make backend behavior diverge. Invalid segments should be skipped
before entering the hot path.
