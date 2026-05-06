# Core audit pass 28: segment cache index load count/file-size contract

## Summary

Pass 28 fixes the read side of the segment-cache index persistence contract.

Passes 26 and 27 hardened index insertion and data append. The index loader
still trusted the on-disk header count far enough to allocate an entry array
before proving that the index file actually contains that many entries.

## Bug

The index file stores:

```text
SpectralSegCacheHeader
SpectralSegCacheEntry[count]
```

`count` comes from disk. The old loader checked the magic/version and then
allocated:

```c
spectral_malloc_array((size_t)hdr.count, sizeof(SpectralSegCacheEntry))
```

before validating that the index file size equals:

```text
sizeof(SpectralSegCacheHeader) + count * sizeof(SpectralSegCacheEntry)
```

A corrupt cache file could therefore request a huge allocation even when the
file is too small to contain the declared entries.

## Fix

`Spectral_seg_cache_fs_index_load()` now:

- computes `entries_bytes` with `spectral_array_bytes()` before allocation;
- computes the expected full index-file size with `spectral_size_add()`;
- queries the actual file size with `spectral_fs_file_size()`;
- rejects mismatches as `SPECTRAL_ERR_FILE_CORRUPT`;
- applies the same exact-size rule to zero-count index files;
- allocates the entries array only after the on-disk count and file length agree.

## Reviewer Walkthrough

1. The loader reads and byte-swaps the header exactly as before.
2. For `count == 0`, the only valid index length is the header size. Extra
   bytes mean stale/corrupt index data and now return `SPECTRAL_ERR_FILE_CORRUPT`.
3. For `count > 0`, the loader first derives `entries_bytes` from the count
   using checked arithmetic.
4. It then derives `expected_index_bytes = sizeof(header) + entries_bytes` with
   checked addition.
5. Only if the actual file size equals that expected byte count does it allocate
   and read the entries.
6. Truncated files, stale trailing bytes, and impossible counts are rejected
   before heap pressure or partial reads.

## Why this is critical

The segment cache is a persistence boundary. A cache file is not trusted input
once it is on disk: it can be truncated, stale, corrupted, or produced by an
older build. The loader must validate the file's declared shape before using
that shape to allocate memory.
