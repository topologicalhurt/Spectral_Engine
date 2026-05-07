# Core audit pass 47: segment loop sample-offset float representability contract

## Summary

Pass 47 completes the segment-loop contract introduced by Pass 41.

Pass 41 validated derived phase/amplitude scalars and endpoint values, but the
synthesis callbacks still use float sample offsets:

```c
(float)j
```

inside CPU/native segment loops.

## Bug

`SegmentLoopParams.length` is a `size_t`. A huge but otherwise valid loop length
could make later `size_t -> float` conversions unrepresentable in the hot loop.

The endpoint validation added in Pass 41 also used:

```c
const float last = (float)(lp.length - 1u);
```

without proving that final offset was representable as `float`.

## Fix

`segment_loop_params_init()` now computes:

```c
last_offset_d = (double)(lp.length - 1u)
```

and rejects the segment unless:

```text
last_offset_d is finite
last_offset_d <= FLT_MAX
```

Only then does it narrow the endpoint offset to `float`.

## Reviewer Walkthrough

1. Time-window and zero-length checks remain.
2. The final sample offset is derived in `double`.
3. The value is checked against `FLT_MAX`.
4. Endpoint validation narrows only the checked value to `float`.
5. Hot-loop callbacks can then safely cast their sample offsets within the same
   representable domain.

## Why this is critical

The CPU/native synthesis formulas are float-domain formulas. A loop length that
cannot be represented in that domain is not a valid hot-loop request, even if the
host `size_t` length itself is representable.
