# Core audit pass 31: segment cache input identity contract

## Summary

Pass 31 fixes the segment-cache key identity contract.

Passes 26-30 hardened the cache persistence format and scalar key construction.
The remaining key bug was that the pipeline passed only the input basename into
the cache key:

```text
basename + FFT/hop/threshold/window/stretch/tile-size
```

Two different files named `song.wav` in different directories, or a rewritten
file with the same basename and parameters, could therefore reuse the wrong
cached segment set.

## Bug

The segment cache stores analysis output. The cache identity must include the
audio input identity, not only a display-oriented basename.

The previous path did this:

```c
spectral_basename_no_ext(opts->input_path, stem, sizeof(stem));
spectral_seg_cache_key(stem, ...);
```

That is not a content identity. It is a presentation label.

## Fix

The pipeline now hashes the input file content with the existing streaming hash
API:

```c
SpectralHashFileMethod method = SPECTRAL_HASH_FILE_METHOD_ZERO_INIT;
spectral_hash_file_method_init(&method, SPECTRAL_HASH_FILE_STREAM);
spectral_hash_file_method_consume_file(&method, f);
spectral_hash_file_method_digest(&method, &digest);
```

`build_cache_key()` now constructs an input identity string:

```text
basename|xxh=<content digest>
```

and passes that to `spectral_seg_cache_key()`.

The core cache-key API now names its first argument `input_id` and rejects empty
input identities. The basename is still kept as a readable namespace hint, but
the digest is the actual collision boundary.

## Reviewer Walkthrough

1. `build_cache_key()` still extracts the basename for readability.
2. It now also streams the input file through the existing xxHash-backed file
   hashing API.
3. If the input file cannot be hashed, cache is disabled by returning key zero.
4. The formatted identity string must fit in the fixed stack buffer.
5. `spectral_seg_cache_key()` now receives the identity string and rejects empty
   identities before hashing the final parameter blob.
6. The old basename-only call path is removed.

## Why this is critical

A persistence cache key is a correctness boundary. If two different inputs map
to the same key, the engine can return valid-looking segments for the wrong
audio. That is worse than a cache miss: it is silent semantic corruption.
