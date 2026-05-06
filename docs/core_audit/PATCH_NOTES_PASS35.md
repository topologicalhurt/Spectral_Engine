# Core audit pass 35: wavetable raw/hex/buffer ingestion contract

## Summary

Pass 35 completes the wavetable persistence/input boundary started in Pass 34.

Pass 34 hardened `.spwt`. The remaining wavetable ingress paths still published
sample tables without the same shape and sample-validity contract:

```text
spectral_wavetable_load_raw()
spectral_wavetable_load_hex()
spectral_wavetable_load_buffer()
```

## Bug

The raw loader read exactly `SPECTRAL_WAVETABLE_SIZE` samples, but did not prove
the file was exactly that length. A stale or concatenated raw file with trailing
bytes could still be accepted.

The HEX loader required byte coverage, but did not require an Intel HEX EOF
record. A truncated file that happened to cover all bytes before losing its
terminator could be accepted.

All three remaining ingress paths could publish non-finite float runtime
samples, which can propagate NaN/Inf through wavetable interpolation and
synthesis.

The HEX loader also used:

```c
(size_t)address + data_len > expected_bytes
```

as an overflow-prone bounds expression.

## Fix

`Spectral_wavetable_load_raw()` now:

- checks exact raw file size with `spectral_fs_file_size()`;
- reads payload with `spectral_fs_read_exact()`;
- validates runtime samples before publishing.

`Spectral_wavetable_load_hex()` now:

- checks the wrap-sample allocation count with `spectral_size_add()`;
- uses subtractive bounds checking for record writes;
- requires the EOF record;
- requires full byte coverage;
- validates runtime samples before publishing.

`Spectral_wavetable_load_buffer()` now validates runtime samples before copying
into the bank.

## Reviewer Walkthrough

1. Raw loader computes the canonical sample byte count.
2. It compares actual file size against that byte count before reading.
3. It uses exact-read semantics and validates the temporary sample array before
   marking the table loaded.
4. HEX loader allocates its wrap-sample table with a checked `+1`.
5. Each HEX data record is bounds-checked with `data_len > expected_bytes - offset`.
6. The loader requires both full coverage and the EOF record.
7. Buffer loading now shares the same runtime sample validator as file loading.

## Why this is critical

Wavetable files and buffers are synthesis input. A loader should never publish a
table containing NaN/Inf or silently accept trailing/stale/truncated data. Once a
table is marked valid, downstream oscillator code treats it as trusted.
