# Patch notes — Pass 205: U2 — adversarial audit of the optimized band-limited file

## Scope

Oscillator-unification step **U2** (`docs/core_audit/OSCILLATOR_UNIFICATION_PLAN.md`):
adversarial-correctness pass over the *optimized* `core/spectral_osc_bandlimited.c`
(post-U3/PASS204). Axes: NaN/Inf propagation, decimation index/alignment at the new
interior/edge split, the new thread-local scratch alloc-failure path, the
timbre×quality fallback matrix, integer overflow, and division safety. This is the
last U-track step; the file is an opt-in audition path, not a golden contract.

## Findings

### 1. NaN frequency slipped the clamp in `osc_bl_norm_freq` — FIXED

`osc_bl_norm_freq` clamps the instantaneous normalized frequency to `[1e-6, 0.5]`:

```c
if (dt < 1e-6f) dt = 1e-6f;
else if (dt > 0.5f) dt = 0.5f;
```

For a NaN `dt` (a NaN chirp slope `alpha`/`c2`/`c3`) **both comparisons are false**,
so NaN passed through unclamped. In `additive` it then poisons
`n_harm = (int)floorf(0.5f / dt)` (NaN→int is undefined). `oversample` is unaffected
(it never calls this helper); `polyblep` degrades gracefully (NaN `dt` makes every
`poly_blep`/`poly_blamp` comparison false → zero residual).

Fix — replace the lower test with its negation so NaN folds to the floor:

```c
if (!(dt > 1e-6f)) dt = 1e-6f;   /* also catches NaN */
else if (dt > 0.5f) dt = 0.5f;
```

`!(dt > 1e-6f)` is identical to `dt < 1e-6f` for **every finite** `dt` (they differ
only at `dt == 1e-6f`, where both branches already produce `1e-6f`), so the finite
path is byte-for-byte unchanged — verified: additive/polyblep/oversample/naive output
hashes are all identical after the fix. Only a NaN slope changes (NaN → `1e-6` instead
of propagating).

### 2. Thread-local scratch is never freed — DOCUMENTED, no change

`osc_bl_os_scratch` grows a per-thread buffer monotonically to the largest segment a
thread renders and never frees it (released by the OS at process exit). In a
long-running host that does one big oversample render then switches to naive, up to
`OSC_BL_OS_MAX_LEN * OS * sizeof(float)` = 64 MB per worker thread stays resident.
This is the deliberate U3 tradeoff (a `malloc`/`free` per segment was the hotspot) and
matches the existing `gpu_tile_cache` thread-local pattern; it is documented at the
declaration. Not a correctness bug.

## Verified sound (no change needed)

- **Decimation indices/alignment.** Interior branch requires `center-H >= 0 &&
  center+H < os_len`, so the symmetric reads `os_buf[center ± m]` (m ≤ H=32) stay in
  `[0, os_len-1]`. The edge branch clamps every `lo`/`hi`/center index. `center = 4j`
  for `j ∈ [0,len)` is always in `[0, os_len-4]`. Every output sample is covered by
  exactly one branch; no out-of-bounds access. `long` index math has no signed/unsigned
  mix and `os_len ≤ 16M` is far below `LONG_MAX`.
- **Alloc-failure path.** `realloc` failure returns NULL and leaves `g_osc_bl_os_buf` /
  `g_osc_bl_os_cap` pointing at the still-valid old buffer; `osc_bl_oversample` bails to
  naive (returns 0). No leak, no torn state, no double-free.
- **Integer overflow.** The `len > OSC_BL_OS_MAX_LEN` guard precedes `len * OS`, so the
  oversample-length multiply and `need * sizeof(float)` (≤ 256 MB) cannot overflow
  `size_t`.
- **Fallback matrix (timbre × quality).** polyblep early-returns 0 for asin/quantized
  and `default:0` for any other; additive returns 0 for pwm/asin/quantized and covers
  sine/saw/square/triangle/parabola; oversample is universal (`osc_bl_naive`
  `default→sine`). Every one of the 8 timbres resolves in every mode, and an
  unsupported pair returns 0 so the caller renders naive.
- **Division safety.** On the finite path `dt ≥ 1e-6` (so `0.5/dt` and the BLEP `t/dt`
  are bounded), `sinf(px)/px` is guarded by the `x==0` special case, and the FIR
  DC-normalizer `1/sum` divides by the strictly-positive window sum.
- **Bad phase (NaN/Inf phase0).** Produces a NaN sample exactly as the naive scalar
  golden path does — shared `spectral_normalize_phase` behavior, not a band-limited
  regression.

## Verification

```text
- 5 production targets build clean (desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float).
- ctest 5/5 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift, osc_parity).
- Finite-path byte-identity after the NaN fix: naive e3ecf301, polyblep 584b9f5a,
  additive e3176ade, oversample 369afd5f — all unchanged.
```

## Status

U2 closes the oscillator-unification U-track (U1a→U1b→U1c→U1d→U3→U2). The
band-limited file is audited: one finite-preserving NaN hardening applied, the
thread-local-retention tradeoff documented, and the decimation/alloc/fallback/overflow
paths confirmed sound. Next effort (per the standing directive) is the **Q-type domain
phase** in `docs/core_audit/QTYPE_DOMAIN_PLAN.md` (Q0→Q4).
