# Core audit pass 57: mono-to-stereo index-domain contract

## Summary

Pass 57 hardens the public mono-to-stereo helpers.

The higher-level file writer already checks stereo allocation sizes, but the
helpers themselves are public and write through indices derived from:

```c
i * 2
i * 2 + 1
```

## Bug

Both float and Q15 helpers accepted any `size_t num_frames`. If a caller passed
a value larger than `SIZE_MAX / 2`, the stereo index arithmetic could wrap.

The Q15 ARM unrolled path also uses `i * 2 + 3`, so it depends on the same
domain proof.

## Fix

Both helpers now fail closed when:

```text
num_frames > SIZE_MAX / 2
```

and compute `stereo_i = i * 2u` once per iteration before indexing.

## Reviewer Walkthrough

1. Existing null/empty checks remain.
2. The helpers reject frame counts outside the stereo-index domain.
3. Float path derives `stereo_i` once and writes `[stereo_i]` and
   `[stereo_i + 1]`.
4. Q15 scalar and ARM paths use the same checked index-domain contract.
5. Callers that already allocate with checked sizes remain unaffected.

## Why this is critical

Void helpers cannot report overflow. Their only safe behavior for an impossible
index domain is to do nothing. This prevents public buffer utilities from
wrapping pointer arithmetic even when called outside the high-level writer.
