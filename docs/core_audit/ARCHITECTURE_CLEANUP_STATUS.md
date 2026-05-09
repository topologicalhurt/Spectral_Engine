# Architecture cleanup status

## Completed reusable owners

```text
spectral_contracts.h
  finite spans
  Segment payload/synth contracts
  SegmentGpu payload/match contracts
  GPU tile layout contract
  checked counter/time accumulation

spectral_analysis_internal.h / spectral_analysis.c
  analysis shape derivation
  analysis window context
  full-matrix STFT matrix owner

spectral_analysis_fft.c
  single shared FFT frame dispatch
  backend-specific single-frame kernels only

spectral_peak_estimator.c
  double-domain offset clamp helper
  magnitude triplet loader
  shared estimator neighborhood validation

spectral_peak_track_internal.h
  frame-index -> t_hop conversion helper

spectral_peak_track.c
  peak-model mutation helper
  stats accumulation helpers

spectral_omp.h
  effective OpenMP thread-count clamp
```

## Current principles

```text
validate at boundaries
derive shape once
use canonical contracts
avoid alias wrappers
delete deprecated internal APIs
prefer owner structs for paired resources
keep backend-specific code backend-specific only
```

## Remaining high-value dedup targets

```text
1. GPU tile preprocessing:
   The count/fill algorithm is still dense. Consider extracting a reusable
   tile-layout builder with an explicit scratch object.

2. FFT resource allocation:
   vDSP and FFTW allocation/free are necessarily backend-specific, but resource
   shape/byte derivation can still be documented more cleanly.

3. Peak estimator algebra:
   More helpers could be extracted for double dot products and checked phase
   products, but do this only if it reduces code size without hiding formulas.

4. Tracker candidate flow:
   Candidate queue/flush/emit still has long parameter lists. A candidate frame
   context object would improve readability, but must not obscure hot-path data.

5. GPU prepared dispatch:
   Long-term replacement for process-local one-shot caches should be an explicit
   prepared-GPU-dispatch object passed to Metal/CUDA.
```

## Forbidden anti-patterns

```text
pure alias wrapper functions
unchecked fallback wrappers that discard errors
duplicated finite/range loops when spectral_contracts.h has a helper
count-only identity for cached payloads
backend-specific copies of shared dispatch policy
tests that require stale wrapper names
```

## Next recommended phase

Move from cleanup into architecture-bearing refactors:

```text
Phase D: explicit prepared GPU dispatch object
Phase E: tracker candidate context object
Phase F: full/fused parity harness and behavioral tests
```
