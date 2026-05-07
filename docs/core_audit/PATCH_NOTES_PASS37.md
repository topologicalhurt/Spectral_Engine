# Core audit pass 37: audio file finite-sample contract

## Summary

Pass 37 completes the audio I/O boundary hardened in Pass 36.

Pass 36 proved frame-count representability across the libsndfile/engine API
boundary. The remaining issue was sample-value validity: libsndfile can read or
write float samples, and the engine was not rejecting NaN/Inf at the audio
ingress/egress boundary.

## Bug

`Spectral_audio_read()` read float samples and immediately converted them to
mono. A malformed or unusual float audio file could inject non-finite samples
into analysis:

```text
audio file -> libsndfile float buffer -> mono buffer -> FFT/tracker
```

Multi-channel downmix also accumulated into `float`, then wrote to `float`
without proving the averaged value remained finite and representable.

`Spectral_audio_write()` accepted whatever float buffer it was handed. If a
synthesis bug or upstream malformed input produced NaN/Inf, output persistence
could write it instead of failing at the boundary.

## Fix

Input now uses:

```c
spectral_audio_samples_finite(audio, total_samples)
```

after the libsndfile read and before mono allocation/publication.

Multi-channel downmix now accumulates in `double` and checks the narrowed mono
sample is finite and within `FLT_MAX` before assigning to `float`.

Output now validates the full interleaved output span before opening the output
file:

```c
spectral_audio_samples_finite(buffer, sample_count)
```

`Spectral_audio_write_stereo()` also validates the mono source before allocating
and duplicating it.

## Reviewer Walkthrough

1. `spectral_audio_read()` reads through libsndfile exactly as before.
2. Before allocating/publishing the mono buffer, it rejects any non-finite input
   sample as `SPECTRAL_ERR_FILE_FORMAT`.
3. Mono files reuse the checked `mono_bytes` copy path.
4. Multi-channel files use double accumulation and prove the averaged sample can
   be represented as finite `float`.
5. `spectral_audio_write()` computes the checked sample count, then validates
   every output sample before filling `SF_INFO` or opening the file.
6. `spectral_audio_write_stereo()` rejects non-finite mono input before
   allocating the stereo buffer.

## Why this is critical

Audio files are external input and output. The analysis and synthesis kernels
assume finite float samples. Letting NaN/Inf enter or leave through file I/O
turns a boundary validation failure into unpredictable DSP behavior or corrupt
rendered artifacts.
