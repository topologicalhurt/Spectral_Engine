# Core audit pass 58: WAV RIFF declared-extent contract

## Summary

Pass 58 completes the WAV PEAK scrubber parser contract started in Pass 38.

Pass 38 validated per-chunk extents against the physical file size. The scrubber
still ignored the RIFF container's declared payload size and scanned to physical
EOF.

## Bug

RIFF declares its container payload in bytes 4..7 of the header:

```text
RIFF chunk size = file bytes after the size field
```

The scrubber validated chunks against `file_size`, not the declared RIFF extent.
A file with trailing bytes after the RIFF payload could therefore have those
bytes parsed as chunks.

The scrubber is best-effort, but it is still a parser. It should stay inside the
declared RIFF container.

## Fix

The scrubber now parses the RIFF payload size, validates:

```text
riff_payload_size >= 4
riff_payload_size <= file_size - 8
riff_end = 8 + riff_payload_size
```

and uses `riff_end` for all subsequent chunk-header, chunk-data, padded-next and
timestamp-field bounds checks.

## Reviewer Walkthrough

1. File size and 12-byte RIFF/WAVE header checks remain.
2. The declared RIFF payload size is parsed little-endian.
3. Invalid RIFF sizes stop scrubbing.
4. Chunk scanning is bounded by `riff_end`, not physical EOF.
5. PEAK timestamp writes are also checked against `riff_end`.

## Why this is critical

A file parser must respect both physical file size and container-declared size.
Ignoring the RIFF extent lets trailing bytes outside the logical WAV container
participate in parser control flow.
