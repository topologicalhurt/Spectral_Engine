# Core audit pass 36: audio file frame-count representability contract

## Summary

Pass 36 hardens the audio file I/O boundary.

Recent passes hardened persistent segment, cache and wavetable inputs. The
desktop audio path still crossed the libsndfile API boundary without proving
that libsndfile frame counts and engine allocation counts were representable in
both domains.

## Bug

libsndfile uses `sf_count_t` for frame counts. The engine allocates and indexes
with `size_t`.

The old input path did this:

```c
info->frames = (size_t)sfinfo.frames;
total_samples = (size_t)sfinfo.frames * sfinfo.channels;
sf_readf_float(file, audio, sfinfo.frames);
```

If `sfinfo.frames` was not representable as `size_t`, the allocation could be
based on a truncated count while `sf_readf_float()` still filled using the
original `sf_count_t` frame count.

The mono copy path also recomputed:

```c
(size_t)sfinfo.frames * sizeof(float)
```

instead of reusing the checked byte count.

The output path had the inverse issue:

```c
info.frames = (sf_count_t)num_frames;
sf_writef_float(file, buffer, (sf_count_t)num_frames);
```

without proving `num_frames` was representable as `sf_count_t`.

## Fix

Input now converts through:

```c
spectral_sf_count_to_size(sfinfo.frames, &frames)
```

before allocation or read. It then derives all allocation/copy sizes from the
checked `frames` value.

Output now converts through:

```c
spectral_size_to_sf_count(num_frames, &frames_sf)
```

before filling `SF_INFO` or calling `sf_writef_float()`.

Both read and write paths now validate sample rate against the canonical
configured bounds.

## Reviewer Walkthrough

1. `spectral_audio_read()` zeroes output state before opening the file.
2. It validates libsndfile metadata: positive frame/channel counts and configured
   sample-rate bounds.
3. It proves `sfinfo.frames` is representable as `size_t`.
4. It computes `frames * channels` and mono byte count with checked helpers.
5. It uses the checked mono byte count for the mono copy.
6. Multi-channel downmix loops use `size_t` indices derived from the checked
   shape.
7. `spectral_audio_write()` proves `num_frames` is representable as `sf_count_t`
   before writing.
8. Write close failure is now folded into `SPECTRAL_ERR_FILE_WRITE`.

## Why this is critical

Audio files are external input/output. The kernel must not allocate in one
integer domain and ask libsndfile to read/write in another unvalidated domain.
A frame-count narrowing bug is a direct memory-size contract violation at the
main audio ingress/egress boundary.
