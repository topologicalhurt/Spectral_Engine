# Core audit pass 32: hash file full-direct size and capability contract

## Summary

Pass 32 fixes two hash-file API contract violations exposed by the cache
identity work.

Pass 31 makes cache identity depend on the existing file-hash lifecycle API.
That makes the hash API itself part of the persistence correctness boundary.

## Bug

The full-direct file hashing method measured the remaining file region using
`uint64_t` positions:

```c
uint64_t start_pos;
uint64_t end_pos;
```

but then narrowed the length directly:

```c
total_len = (size_t)(end_pos - start_pos);
```

On a target where `size_t` is narrower than `uint64_t`, that can wrap or
truncate the requested read length. A hash method must either hash the full
declared region or fail/fallback. It must not hash a truncated prefix.

The descriptor table also advertised `SPECTRAL_HASH_FILE_FULL_MMAP` as
available even though its implementation returns `SPECTRAL_ERR_BACKEND_UNAVAIL`.

## Fix

The full-direct method now computes:

```text
total_len_u64 = end_pos - start_pos
```

then checks:

```text
start_pos <= INT64_MAX
total_len_u64 <= SIZE_MAX
```

If the region is too large for a single `size_t` allocation, the method seeks
back to the original position and falls back to the streaming implementation,
which hashes the file in bounded chunks.

The unimplemented full-mmap method remains registered, but its descriptor now
sets:

```text
available = 0
```

so callers fail at method selection rather than at consume time.

## Reviewer Walkthrough

1. `spectral_hash_method_consume_file_full_direct_impl()` still uses the fast
   single-buffer path for files whose remaining length is representable as
   `size_t`.
2. It computes the length in `uint64_t` first.
3. It rejects unrepresentable `start_pos` before passing it to the `int64_t`
   seek API.
4. If the full region cannot fit in `size_t`, it seeks back to `start_pos` and
   delegates to the streaming method.
5. The actual narrowing to `size_t` happens only after the representability
   check.
6. `FULL_MMAP` is no longer advertised as available while its implementation is
   intentionally unavailable.

## Why this is critical

Hashing is an identity boundary. A file hash method that silently hashes less
than the requested file region can create cache/resource identity collisions.
Capability descriptors are also contracts: advertising an unavailable method as
available lets callers build an invalid backend plan.
