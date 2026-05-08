# Core audit pass 64: hash oneshot null-span contract

## Summary

Pass 64 hardens the public one-shot hash helper.

The streaming hash API already treats `(data == NULL, len > 0)` as a parameter
error. `spectral_hash_oneshot()` had no error channel and passed its pointer
directly to xxHash.

## Bug

A public caller could invoke:

```c
spectral_hash_oneshot(NULL, nonzero_len)
```

and pass an invalid span to the hash backend.

For empty spans, a null pointer is conceptually valid, but the helper still needs
to pass a stable pointer to the backend rather than relying on backend-specific
null handling.

## Fix

`Spectral_hash_oneshot()` now:

```text
returns digest zero for NULL + nonzero length
uses a stable static empty byte for NULL + zero length
passes valid spans directly
```

## Reviewer Walkthrough

1. The function checks the input pointer before calling xxHash.
2. Invalid non-empty null spans fail closed with digest zero.
3. Empty spans are hashed using a valid static address and length zero.
4. Host and embedded/sim hash backends share the same span policy.

## Why this is critical

Hashing is an identity boundary. A public one-shot hash helper must define its
null-span behavior explicitly instead of delegating invalid pointer semantics to
the backend library.
