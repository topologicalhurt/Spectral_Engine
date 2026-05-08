# Core audit pass 63: hash consume-file lifecycle contract

## Summary

Pass 63 hardens the hash file `consume_file()` boundary.

Pass 55 made public `reset()` validate the method descriptor. `consume_file()`
still switched directly on `method->type` and let each backend implementation
reset internally.

## Bug

`consume_file()` did not prove that the method object had been initialized
through the public lifecycle before dispatch. It also did not validate the method
descriptor/capability before choosing the implementation.

After Pass 32, descriptor availability is a real capability boundary because
`FULL_MMAP` is registered but intentionally unavailable.

## Fix

Public `spectral_hash_file_method_consume_file()` now validates:

```text
method pointer
file pointer
method->initialized
method descriptor availability
```

before dispatching. It switches on the descriptor type rather than the raw field.

## Reviewer Walkthrough

1. A zero-initialized method returns `SPECTRAL_ERR_NOTINIT`.
2. A destroyed method returns a descriptor/type error.
3. Unavailable methods still fail through the descriptor registry.
4. Valid initialized stream/full-direct methods behave as before.
5. Backend implementations can still reset their state internally after the
   lifecycle/capability boundary is proven.

## Why this is critical

The hash lifecycle now participates in cache identity. File consumption must not
bypass the same capability and lifecycle gates that protect `init()` and
`reset()`.
