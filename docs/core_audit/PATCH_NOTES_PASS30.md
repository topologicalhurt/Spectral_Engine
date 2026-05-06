# Core audit pass 30: segment cache key and store scalar metadata contract

## Summary

Pass 30 closes the scalar-metadata side of the segment-cache persistence
contract.

Passes 26-29 hardened index insertion, data append, index load, and lookup.
The cache key/store ingress path still accepted invalid scalar inputs before
conversion or persistence.

The concrete failure modes were:

```text
NaN/Inf/out-of-range float -> int conversion while building cache key
overlong formatted key string -> silently truncated key material
invalid store metadata -> persisted entry that hardened lookup later rejects
```

## Bug

`spectral_seg_cache_key()` built the hash input with casts such as:

```c
(int)(db_thresh * 10.0f)
(int)(start_sec * 1000.0f)
(int)(end_sec * 1000.0f)
(int)(stretch * 1000000.0f)
```

Converting NaN, Inf, or an out-of-range finite floating-point value to integer
is undefined behavior in C. The function also accepted `snprintf()` truncation
by hashing the truncated buffer.

The store path also wrote scalar metadata directly into the persistent index:

```c
.sample_rate = (uint32_t)sample_rate
.stretch = stretch
.pitch = pitch
.output_length = (uint64_t)output_length
```

without first enforcing the same scalar contract that lookup now requires.

## Fix

The key path now uses a guarded helper:

```c
seg_cache_scale_to_i32(value, scale, &out)
```

which rejects non-finite or out-of-range scaled values before converting to
`int`.

`spectral_seg_cache_key()` now also rejects:

```text
invalid FFT/hop/tile size
non-finite or non-positive stretch
stretch above SPECTRAL_MAX_STRETCH
snprintf truncation
```

The store path now validates metadata before opening the append writer:

```text
key != 0
SegmentArray pointer/segment pointer consistency
sample_rate in configured bounds
stretch finite, positive, <= SPECTRAL_MAX_STRETCH
pitch finite and within configured pitch bounds
output_length representable as uint64_t
```

## Reviewer Walkthrough

1. `spectral_seg_cache_key()` now converts every scaled float through
   `seg_cache_scale_to_i32()`.
2. The helper computes the scaled value in `double`, checks finiteness and
   `INT_MIN..INT_MAX`, then performs the integer conversion.
3. The formatted key string must fit in the fixed stack buffer. If `snprintf()`
   reports truncation, the function returns zero instead of hashing a prefix.
4. `spectral_seg_cache_store()` calls `seg_cache_store_metadata_valid()` before
   any append operation.
5. Store validation mirrors the scalar constraints enforced by lookup, so the
   engine no longer produces cache entries that its own hardened lookup rejects.

## Why this is critical

The segment cache key is the identity of persisted analysis/render state. Key
construction must be deterministic and defined for every accepted input. The
cache store path is a persistence boundary: it must not write metadata that is
outside the live kernel's configured domain.
