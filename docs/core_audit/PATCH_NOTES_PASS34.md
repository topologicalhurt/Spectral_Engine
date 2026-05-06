# Core audit pass 34: wavetable .spwt file shape and sample contract

## Summary

Pass 34 hardens the native `.spwt` wavetable persistence boundary.

Pass 33 hardened exported segment files. The wavetable parser had the same
class of issue: it read a header from disk, trusted `size`/`format`, and then
read payload samples without proving the exact file shape or rejecting
non-finite float samples.

## Bug

The `.spwt` format declares:

```text
SpectralWavetableHeader
samples[size] in header-selected format
```

The old loader validated `size == SPECTRAL_WAVETABLE_SIZE`, then read the
payload with branch-local `spectral_fs_read()` calls. It did not first prove
that the file was exactly:

```text
sizeof(SpectralWavetableHeader) + size * sample_size(format)
```

bytes.

A truncated or stale file could therefore be detected late, and an overlong file
with trailing stale bytes could still be accepted. The loader also accepted
non-finite float samples, which can propagate NaN/Inf through interpolation and
synthesis.

The save path wrote whatever sample values were in the table, even if the loader
would later consider them corrupt.

## Fix

The wavetable parser now has checked helpers:

```c
wavetable_file_sample_size(format)
wavetable_file_payload_bytes(format, size, &bytes)
wavetable_file_expected_bytes(format, size, &bytes)
wavetable_float_samples_finite(samples, count)
wavetable_runtime_samples_valid(samples, count)
```

Load now validates:

```text
magic/version/format/size
timbre_id < SPECTRAL_MAX_WAVETABLES
exact file size before payload read
finite float payload samples
```

Save now validates runtime samples before opening the output file and writes
header/payload through `spectral_fs_write_exact()` using a checked byte count.

## Reviewer Walkthrough

1. The loader reads the 32-byte header exactly.
2. It validates the declared format and size.
3. It computes the payload and full expected file byte counts with checked
   arithmetic.
4. It compares the actual file size against that expected byte count before any
   payload allocation or payload read.
5. Float payloads are rejected if any sample is non-finite.
6. The save path checks sample finiteness and sample byte count before opening
   the output file.
7. Header and payload writes use exact-write helpers, so partial writes do not
   masquerade as successful saves.

## Why this is critical

Wavetables are synthesis input. A corrupted `.spwt` file can inject non-finite
samples into the oscillator lookup path, and a stale/trailing payload can hide
format drift. The parser must validate file shape and payload sanity before
publishing a table as valid.
