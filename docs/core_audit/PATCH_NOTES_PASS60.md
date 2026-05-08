# Core audit pass 60: vector primitive null/span contract

## Summary

Pass 60 hardens public vector primitives.

The core SIMD helpers are used by higher-level kernels with valid buffers, but
they are also public core utilities. Several functions dereferenced inputs or
output pointers before checking null or empty spans.

## Bug

Examples:

```c
spectral_vmax(src, out, len)
spectral_vmaxmgv(src, out, len)
spectral_vmul(a, b, dst, len)
```

`vmax`/`vmaxmgv` wrote `*out` even when `out` could be null. Binary/vector
operations assumed all pointers were valid even for zero-length calls.

## Fix

Vector primitives now fail closed on invalid pointer/span inputs:

```text
vmul/vadd/vsq/vsmul/vatan2: return on null or zero length
vmax/vmaxmgv: return on null out; set zero for null/empty source
```

## Reviewer Walkthrough

1. Binary vector ops reject missing inputs/output before SIMD loads/stores.
2. Unary vector ops reject missing source/output before SIMD loads/stores.
3. Max helpers check `out` before writing.
4. Empty max spans still produce zero when `out` is valid.
5. Hot paths with valid arguments pay only one predictable branch at entry.

## Why this is critical

A public vector primitive should not crash on invalid zero-work input. These
helpers are low-level kernel utilities; their boundary behavior should be
deterministic and safe before SIMD memory access.
