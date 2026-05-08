# Core audit pass 55: hash file reset lifecycle contract

## Summary

Pass 55 fixes the hash-file lifecycle contract.

The hash API documents an explicit lifecycle:

```text
init/reset -> update -> digest -> destroy
```

`init()` performs the first reset. Public `reset()` is for an already initialized
method object.

## Bug

`Spectral_hash_file_method_reset()` called the internal reset implementation
without validating the method descriptor.

That allowed this invalid sequence:

```c
SpectralHashFileMethod m = SPECTRAL_HASH_FILE_METHOD_ZERO_INIT;
spectral_hash_file_method_reset(&m);
```

to allocate/reset hash state and mark the method initialized while
`method.type` was still `SPECTRAL_HASH_FILE_METHOD_COUNT`.

That violates the lifecycle contract and bypasses descriptor availability. After
Pass 32, descriptor availability is a real capability boundary because
`FULL_MMAP` is registered but unavailable.

## Fix

Public `reset()` now validates:

```c
spectral_hash_file_method_get_descriptor(method->type, &desc)
```

before resetting state.

This preserves `init()` behavior because `init()` sets `method->type` before
calling reset. It rejects reset-before-init and unavailable method types.

## Reviewer Walkthrough

1. Zero-initialized methods still use `SPECTRAL_HASH_FILE_METHOD_ZERO_INIT`.
2. `init()` still checks descriptor availability and sets `method->type`.
3. `init()` then calls public reset, which now validates that type.
4. A direct reset on a zero-initialized object fails instead of implicitly
   initializing an object with invalid type.
5. Descriptor availability remains the canonical capability gate.

## Why this is critical

The hash API is now part of cache identity. Lifecycle operations must not bypass
the capability registry or create initialized objects with invalid method types.
