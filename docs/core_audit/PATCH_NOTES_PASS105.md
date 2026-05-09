# Core audit pass 105: checked accumulation contract promotion

## Summary

Pass 105 promotes reusable counter/time accumulation helpers into the canonical
contract layer.

Pass 101 introduced tracker-local helpers for:

```text
uint64_t checked addition
non-negative double timing accumulation
```

Those contracts are not tracker-specific. They are reusable for future cache,
analysis and backend diagnostics.

## Fix

Pass 105 adds to `spectral_contracts.h`:

```c
spectral_u64_add_checked()
spectral_double_accumulate_nonnegative_checked()
```

Tracker stats accumulation now uses those canonical helpers.

## Why this is critical

Diagnostic counters appear throughout the kernel. Overflow-safe accumulation
should not be reimplemented per subsystem.
