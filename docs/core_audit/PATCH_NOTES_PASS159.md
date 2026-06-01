# Patch notes — Pass 159: COLA/WOLA reconstruction invariant + test (Phase B0)

## Problem

Phase B (the contract/guarantee registry) cannot record "we relaxed COLA" until
COLA is first *defined and tested*. Today there is no overlap-add reconstruction
invariant in the engine: windows are generated unnormalized (`spectral_windows.h`:
"conventional, unnormalized sample-domain window shapes") and nothing asserts the
Constant/Weighted OverLap-Add identity. B0 is the prerequisite for the B1 manifest.

## Change

Defined the COLA/WOLA invariant as header-only predicates, documented it as the
first entry in the guarantee registry, and pinned it with a compiled CTest.

```text
core/spectral_contracts.h        ADD two static-inline predicates:
  spectral_overlap_add_envelope_stats(a, s, length, hop, &min,&max,&mean)
      env[n] = sum_k a[n+k*hop]*s[n+k*hop] over n in [0,hop); s==NULL => COLA.
      requires 0<hop<=length and hop|length. O(length). reports min/max/mean.
  spectral_overlap_add_is_constant(a, s, length, hop, rel_tol, &gain)
      1 iff (max-min) <= rel_tol*mean, mean>0; writes constant overlap gain=mean.

tests/core_contracts/test_cola.c NEW compiled test (mirrors tests/arm_core style).
  Asserts: periodic Hann COLA at hop N/2 (gain~1) and N/4 (gain~2); rectangular
  COLA (gain=N/hop); periodic-Hann WOLA self-window (sum w^2 constant, Griffin-Lim);
  the engine's SYMMETRIC spectral_window_hann FAILS strict COLA (tol 1e-6) but
  passes loose (1e-2) with rel deviation ~1.5e-3 (O(1/N)); and rejects bad inputs
  (hop not dividing N, hop==N no-overlap, NULL). Forces SPECTRAL_USE_VDSP=0 so it
  exercises the source-visible window formulas deterministically.

spectral_engine/cmake/targets/core-contracts-test.cmake  NEW target
  core_contracts_test (test_cola.c + spectral_windows.c + spectral_fast_math.c,
  dead-strip), registered as CTest `core_contracts`. Wired into
  spectral_engine/CMakeLists.txt next to arm-core-test.cmake.

core/spectral_config.h           guard SPECTRAL_USE_VDSP in #ifndef so a build can
  force the portable window/FFT path (-DSPECTRAL_USE_VDSP=0). Default-preserving:
  with no -D it expands exactly as before (1 on __APPLE__, else 0).

docs/core_audit/CORE_CONTRACTS.md  ADD the guarantee-registry framing + the
  COLA/WOLA invariant entry (definition, predicates, test matrix, sources, and the
  symmetric-window finding). B1/B2 sections stubbed as pending.
```

## Finding

The engine ships **symmetric** windows (`N-1` denominator). These are *analysis*
windows and do **not** strictly satisfy COLA — only *periodic* (`N` denominator)
windows at `hop = N/2, N/4, …` and the rectangular window do. The symmetric Hann's
overlap deviation is O(1/N) (~1.5×10⁻³ at N=1024). This is acceptable **because the
engine has no overlap-add resynthesis path** (synthesis is segment/sinusoidal, not
inverse-STFT), so COLA is a *latent* property of the window generators, not an
active runtime guarantee. The contract makes that explicit; any future
inverse-STFT/WOLA path now has a ready gate (use a periodic window or normalize by
`env`, then assert `spectral_overlap_add_is_constant`).

## Verification

```text
- five production targets build clean: desktop, simulate, embedded_arm,
  embedded_arm_float, simulate_daisy (only pre-existing benign -mavx2/-mno-avx512f
  unused-arg notes). The spectral_config.h guard is default-preserving — confirmed
  by clean rebuilds (no -D passed to any production target).
- ctest (both suites): 2/2 PASSED — arm32_process_correctness (regression check
  after the config-header touch) and the new core_contracts.
- core_contracts_test direct run prints "symmetric Hann hop N/2: relative envelope
  deviation = 1.535e-03", matching the expected ~1.5/N.
```

## Scope (Phase B increment)

B0 only: the reconstruction invariant exists, is documented in the registry, and is
tested. Next: B1 (enumerate every correctness-relaxing flag in the manifest with an
error budget + drift test) and B2 (compile-time `SPECTRAL_ACTIVE_GUARANTEES` bitset
+ runtime self-report query API).
