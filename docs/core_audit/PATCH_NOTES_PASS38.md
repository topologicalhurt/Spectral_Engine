# Core audit pass 38: WAV PEAK scrubber chunk extent contract

## Summary

Pass 38 hardens the deterministic WAV post-processing path.

`Spectral_audio_write()` disables libsndfile's non-deterministic PEAK timestamp
where possible, then calls a best-effort RIFF/PEAK scrubber. That scrubber is a
file parser: it walks RIFF chunks and seeks to chunk-derived offsets.

## Bug

The scrubber parsed RIFF chunk sizes from disk and then used unchecked offset
arithmetic:

```c
data_pos + 4
data_pos + size + (size & 1u)
```

before casting the result into the `int64_t` seek API.

Even though the scrubber is best-effort and usually runs on files the engine
just wrote, it still reads a file boundary and must not wrap offsets or seek to
unrepresentable positions.

It also used generic `spectral_fs_read()` / `spectral_fs_write()` calls for
fixed-size header and timestamp operations, so partial I/O could look like an
ordinary end-of-scan instead of an exact operation failure.

## Fix

The scrubber now:

- obtains the file size before walking RIFF chunks;
- checks the RIFF header is fully present;
- reads RIFF and chunk headers with `spectral_fs_read_exact()`;
- validates every chunk extent before adding offsets;
- checks padded chunk positions remain inside the file;
- checks `uint64_t` offsets are representable before seeking through the
  `int64_t` seek API;
- writes the PEAK timestamp field with `spectral_fs_write_exact()`.

The function remains best-effort: malformed files still just stop scrubbing.
The difference is that stopping is now fail-closed and arithmetic-safe.

## Reviewer Walkthrough

1. The function opens the file in `r+b` as before.
2. It reads the actual file size and rejects files shorter than the RIFF header.
3. Before each chunk read, it proves the complete 8-byte chunk header is present.
4. It parses the little-endian RIFF chunk size.
5. It proves `data_pos + size` and the padded next-chunk position stay within
   the file.
6. For a PEAK chunk, it proves the timestamp field lies within the file before
   seeking and writing four zero bytes.
7. Every seek goes through `spectral_wav_seek_u64_checked()`, which rejects
   offsets that cannot be represented by the shared `int64_t` file-seek API.

## Why this is critical

The output path is a persistence boundary too. A deterministic post-processor
must not turn a malformed or externally modified output file into unchecked
offset arithmetic. The scrubber may be best-effort, but its file traversal still
has to be memory- and offset-safe.
