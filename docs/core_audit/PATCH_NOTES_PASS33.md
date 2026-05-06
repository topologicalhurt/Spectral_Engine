# Core audit pass 33: segment file metadata and shape contract

## Summary

Pass 33 hardens the standalone segment-file parser.

The segment cache has now been hardened as a persistence boundary. The exported
segment file path had the same class of problem: metadata and count fields came
from disk, but load used them before proving the file shape and scalar metadata
were valid.

## Bug

`segments_load()` read the header and then allocated based on:

```c
hdr.count * sizeof(Segment)
```

before proving that the file actually had exactly:

```text
sizeof(SegmentFileHeader) + hdr.count * sizeof(Segment)
```

bytes.

A corrupt segment file could therefore request a large allocation even when the
file was truncated or stale.

The parser also trusted scalar metadata:

```text
sample rate
stretch
pitch
```

before narrowing/publishing those values, including the zero-count file path.

On the write side, `segments_save()` could persist invalid metadata or corrupt
segment values that the loader would later reject.

## Fix

The parser now has a shared metadata validator:

```c
segment_file_metadata_valid_u32(sr, stretch, pitch)
```

and a checked shape helper:

```c
segment_file_expected_bytes(count, &bytes)
```

Load now validates:

```text
sample rate in configured bounds
stretch finite, positive, <= SPECTRAL_MAX_STRETCH
pitch finite and within configured bounds
exact file size == header + count * sizeof(Segment)
```

before allocating segment storage or publishing metadata.

Save now validates the same metadata, checks segment byte counts before opening
the output file, and refuses to write corrupt segment arrays.

## Reviewer Walkthrough

1. Save rejects invalid scalar metadata before opening the output file.
2. Save checks the segment array byte count even on big-endian paths, so the
   write contract is proven independent of branch.
3. Save validates every segment before persisting it.
4. Load validates magic/version as before.
5. Load validates scalar metadata before assigning it to public outputs.
6. Load computes the exact expected file size with checked arithmetic and
   compares it against `spectral_fs_file_size()`.
7. Zero-count files must be exactly header-sized; trailing bytes are treated as
   corruption.
8. Only after metadata and file-shape validation does load allocate the segment
   array.

## Why this is critical

Segment files are persistent input. They can be user-provided, stale, truncated,
or produced by an older/corrupt build. The parser must validate the declared
shape before using it to allocate memory, and it must not persist data that its
own loader considers invalid.
