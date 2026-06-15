# Core contracts

Validation belongs at boundaries. Hot loops should consume already validated
state and only check derived hot-loop quantities that are created inside the hot
loop.

## Canonical reusable contracts

```text
spectral_f32_span_finite
spectral_segment_payload_valid
spectral_segment_valid_for_synth
spectral_segment_array_payload_valid
spectral_segment_array_valid_for_synth
spectral_segment_gpu_matches_segment
spectral_segment_gpu_array_matches_segments
spectral_gpu_tile_layout_words_valid
spectral_overlap_add_envelope_stats
spectral_overlap_add_is_constant
```

## Segment payload contract

```text
start  finite, >= 0
length finite, >= 0
phase  finite
omega  finite, >= 0
df     finite
amp    finite
da     finite
width  finite
```

## Synthesis segment contract

```text
payload segment contract
length > 0
amp >= 0
```

## Tile layout contract

```text
range[i].start == running_refs
running_refs + range[i].count does not overflow
final running_refs == total_refs
every tile_segment_ids[j] < segment_count
```

---

# Guarantee registry

The single place that records which kernel correctness/quality invariants hold,
which are relaxed by a flag or runtime path, and how a caller/test discovers the
active set. Generalizes the maintainer's WOLA/COLA concern to all invariants
"hidden behind obfuscations or branches".

It has three parts: the reconstruction invariant (defined + tested); the guarantee
manifest below — every relaxing gate enumerated with an error budget + drift test;
and the compile-time `SPECTRAL_ACTIVE_GUARANTEES` bitset + runtime query API.

## Reconstruction invariant — COLA / WOLA overlap-add

**Invariant.** For an STFT analyzing with window `a[0..N-1]`, resynthesizing with
window `s[0..N-1]`, advancing by `hop` samples per frame, the reconstruction
envelope is

```text
env[n] = sum_k a[n + k*hop] * s[n + k*hop],   n in [0, hop)
```

Overlap-add resynthesis is gain-flat (no amplitude modulation at the hop rate)
exactly when `env[n]` is the same constant for every phase position `n`: the
Constant-OverLap-Add (COLA) condition for one window, or its Weighted form (WOLA)
when an analysis AND synthesis window are both present. The Griffin-Lim
modify-resynthesize identity is the sum-of-squares form `sum_k a^2[n+k*hop] = const`.

Sources: Allen (1977); Allen & Rabiner (Proc. IEEE, 1977); Griffin & Lim (IEEE
TASSP, 1984) [ACADEMIC_SOURCES.md #5]; Harris (1978) [#4]; Smith, SASP [#7].

**Predicates** (`core/spectral_contracts.h`):

```text
spectral_overlap_add_envelope_stats(analysis, synthesis, length, hop, &min,&max,&mean)
    envelope min/max/mean over the hop phase positions (synthesis==NULL => COLA).
    requires 0 < hop <= length and hop divides length. O(length).
spectral_overlap_add_is_constant(analysis, synthesis, length, hop, rel_tol, &gain)
    1 iff (max-min) <= rel_tol*mean with mean>0; writes constant overlap gain=mean.
    rel_tol<0 => exact.
```

**Test** (`tests/core_contracts/test_cola.c`, CTest `core_contracts`):

```text
periodic Hann 0.5(1-cos 2pi n/N), hop N/2  -> COLA, gain ~1
periodic Hann,                    hop N/4  -> COLA, gain ~2
rectangular,                      hop N/4  -> COLA, gain = N/hop = 4
periodic Hann, analysis==synthesis, hop N/4 -> WOLA (sum w^2) constant
engine spectral_window_hann (SYMMETRIC), hop N/2
        -> FAILS strict COLA (tol 1e-6); passes loose (1e-2); rel dev ~1.5e-3
hop does not divide N / hop==N (Hann) / NULL analysis -> rejected
```

The test forces `SPECTRAL_USE_VDSP=0`, a now-redundant belt-and-suspenders: after
the window unification `spectral_windows.c` has **no** vDSP window
path — `vDSP_hann_window`/`hamm`/`blkman` used Apple's *periodic* (`2pi n/N`)
convention and silently diverged from the symmetric form on desktop. All
backends now generate the single documented symmetric formula regardless of
`SPECTRAL_USE_VDSP`. The cross-backend guarantee is enforced by
`tests/core_contracts/test_window_backend_parity.c` (CTest `window_backend_parity`),
compiled with `SPECTRAL_USE_VDSP=1` so any reintroduced periodic window fails.

**Finding the registry now records.** The shipping windows are *symmetric* (`N-1`
denominator — `spectral_windows.h`: "conventional, unnormalized window shapes")
on **every** backend (desktop included).
They are analysis windows and do NOT strictly satisfy COLA; only periodic (`N`
denominator) windows at `hop = N/2, N/4, ...` and the rectangular window do. The
symmetric Hann's overlap deviation is O(1/N) (~1.5e-3 at N=1024). This is
acceptable because the engine has NO overlap-add resynthesis path (synthesis is
segment/sinusoidal, not inverse-STFT), so COLA is a latent property of the window
generators rather than an active runtime guarantee. The contract makes this
explicit and gives any future inverse-STFT/WOLA path a ready gate: use a periodic
window (or normalize by `env`) and assert `spectral_overlap_add_is_constant`.

## Guarantee manifest

Every correctness/quality-relaxing gate the **C sources actually branch on** (verified
by grep, not the plan's aspirational list), the invariant it relaxes, its default,
the cost of relaxing, the error budget, and the drift test that fails if the
documented effect changes. Budgets are the worst case measured with the gate forced
on, rounded up ~2-3x for platform FP variance.

```text
guarantee bit            gate (default)                              relaxes / cost                                   budget (measured)        drift test
ieee_strict_fp           SPECTRAL_CUSTOM_FAST_MATH_MODE (CMake)      project-wide -ffast-math (reassoc, signed-zero,  whole-program; gated     bit-state (reflects
                           =1 in dev, =0 when SPECTRAL_REPRO_BUILD     reciprocal, fp-contract=fast) / vectorizable FP  out of repro/production  the build profile)
exact_trig               SPECTRAL_ENABLE_APPROX_TRIG (0)             sinf -> odd-Taylor poly / speed in osc+peak      2e-6 abs (1.72e-6)       core_guarantees_drift
exact_atan2              SPECTRAL_ENABLE_APPROX_ATAN2 (0)            atan2f -> rational poly / speed in phase         5e-4 rad (2.03e-4)       core_guarantees_drift
exact_inv_sqrt           SPECTRAL_ENABLE_APPROX_INV_SQRT (0)         1/sqrtf -> Quake rsqrt (2 Newton) / speed        1e-5 rel (4.74e-6)       core_guarantees_drift
exact_peak_log           SPECTRAL_ENABLE_APPROX_PEAK_LOG (0)         logf -> atanh series (z^11) / speed in peak dB   2e-6 abs (1.03e-6)       core_guarantees_drift
exact_gpu_fp             SPECTRAL_METAL_FAST_MATH (0)                Metal shader fastMathEnabled / GPU FP            GPU-side; not CPU-       bit-state
                                                                                                                      measurable here
deterministic_reduction  SPECTRAL_SYNTH_DETERMINISTIC_PARTITIONS (0) INVERSE: held when >0. Bit-reproducible CPU      n/a (ordering, not       bit-state
                                                                      reduction order / caps parallel partitions       magnitude)
```

`SPECTRAL_REPRO_BUILD` is a CMake variable (not a C macro): ON (default for
`SPECTRAL_PRODUCTION_BUILD`) defines `SPECTRAL_CUSTOM_FAST_MATH_MODE=0` and omits the
`-ffast-math` family; OFF defines it `=1` and adds them (`host-config.cmake`). It is
therefore represented in the manifest *through* `ieee_strict_fp`, not as its own bit.

`SPECTRAL_OPT_LEVEL` is **intentionally absent**: it is defined (default 1) but no C
source reads it, so it gates no guarantee. The ULTRAPLAN claim that `>= 2` drops LUT
interpolation to nearest is aspirational; do not add a bit until code honors the level.

## Self-report

`core/spectral_guarantees.h` derives a compile-time bitset from the gates above. Each
bit is SET when its invariant holds and CLEARED when a flag relaxed it (the
`deterministic_reduction` bit has inverse polarity — set when the determinism gate is
on). The set is preprocessor-evaluable, so it is usable in `#if` and `_Static_assert`
as well as at run time.

```text
SPECTRAL_ACTIVE_GUARANTEES          compile-time bitset (also via spectral_active_guarantees())
spectral_active_guarantees()        runtime read of the active set
spectral_guarantee_holds(bit)       1 iff that single guarantee is active
spectral_guarantees_satisfy(mask)   FAIL-CLOSED: 1 iff every required bit is active
spectral_guarantee_table(&n)        host/sim only: rows of {bit, name, gate, relaxes}
```

A host/test reads the active set (or names via the table) and refuses a descriptor
that assumes a guarantee this build relaxed, rather than silently emitting
out-of-contract output. Wiring, fail-closed behaviour, and the per-gate drift budgets
are pinned by CTests `core_guarantees` (default: exact bits active) and
`core_guarantees_drift` (`SPECTRAL_ENABLE_APPROX_*` forced on: those bits cleared and
each approximation asserted within budget).
