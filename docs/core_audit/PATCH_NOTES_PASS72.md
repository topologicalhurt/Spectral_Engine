# Core audit pass 72: tracker process matrix and frame-time contract

## Summary

Pass 72 hardens `spectral_tracker_process()`.

The incremental tracker API receives raw STFT matrix pointers plus frame counts.
It then performs matrix row pointer arithmetic and derives per-frame sample time.

## Bug

The function did not prove:

```text
chunk_n_frames * n_freqs is representable
global_frame_offset + t is representable
(global_frame_offset + t) * hop_float is finite and representable as float
tracker scalar state is finite
```

before indexing matrix rows or emitting `t_hop` into segment state.

## Fix

`Spectral_tracker_process()` now validates the chunk matrix shape up front and
checks frame-time derivation per pair in `double` before narrowing to `float`.

Overflow or invalid scalar state is recorded through the tracker first-error
setter.

## Reviewer Walkthrough

1. The function validates tracker scalar state before entering OpenMP.
2. `chunk_n_frames * n_freqs` is checked with `spectral_size_mul()`.
3. The maximum global frame index in this call is checked before the loop.
4. Each pair computes `t_hop_d` in double.
5. Only finite, non-negative, `<= FLT_MAX` frame times are narrowed to float.
6. Invalid frame-time state stops the local worker and preserves root-cause
   error.

## Why this is critical

Tracker matrix rows and emitted segment times are core analysis state. An
incremental API must prove its matrix and time-coordinate domains before hot-path
pointer arithmetic and segment emission.
