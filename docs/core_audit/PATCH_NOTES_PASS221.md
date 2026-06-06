# Patch notes — Pass 221: full/fused analysis parity harness (ULTRAPLAN Phase D — D1)

## Scope

First execution slice of master-campaign **Phase D** (compiled harness + tooling
feedback). Phase D's D0 (harness infrastructure: `enable_testing()` + per-component
C runners under CTest) already exists — it was built up organically across the
oscillator/Q-type sub-campaign (passes 194–220), which left **11** compiled CTest
targets (`arm32_process`, `core_contracts`, `core_guarantees`, `osc_parity`,
`osc_width_parity`, `q_domain_contract`, `q15_compute_precision`,
`q15_production_parity`, `q15_simd_parity`, `phase_nco_precision`, plus the two
benches). So the next undone item in plan order is **D1 — the full/fused parity
harness**, which until now existed only as a *string-grep spec test*
(`tests/core_math/test_core_pass119_full_fused_parity_spec.py`). This pass makes it
a real compiled CTest. Purely additive — no production source, no behavior change.

## What D1 asked for

`docs/core_audit/FULL_FUSED_PARITY_HARNESS.md`: drive the production analysis entry
point under both path modes and assert the two segment arrays agree.

- Entry point: `analyze_audio_with_path_mode(... SPECTRAL_ANALYSIS_PATH_FULL ...)`
  vs `... SPECTRAL_ANALYSIS_PATH_FUSED ...` (`analysis/spectral_analysis.c:210`,
  enum `analysis/spectral_analysis_internal.h:67-68`).
- Deterministic fixtures, segment arrays sorted by `(start, omega, amp)`, per-field
  tolerances, per-field max-error print, **non-zero exit on failure**.
- Forbidden: AUTO-threshold-only tests, string-only parity tests, letting one path
  change estimator/window policy.

## What landed

- **`tests/core_contracts/test_full_fused_parity.c`** — the compiled harness. Six
  deterministic fixtures (silence, bin-centered sine, off-bin sine, two separated
  tones, linear chirp, amplitude ramp) at `sr=48000, n_fft=1024, hop=256,
  db=-80, n=16384`. For each: run FULL, run FUSED, `qsort` both by
  `(start, omega, amp)`, compare `start/length/omega/amp/df/da` against the spec's
  documented tolerances, print per-field `max|d|`, return non-zero on any miss.
- **`spectral_engine/cmake/targets/full-fused-parity-test.cmake`** — registers
  `full_fused_parity_test` + `add_test(NAME full_fused_parity)`. Links the **real
  desktop engine** (`SPECTRAL_SOURCES_TARGET_DESKTOP` minus the CLI `main.c`, plus
  the harness `main()`), mirroring `targets/desktop.cmake` (Metal on Apple,
  `SPECTRAL_GPU_LINK_LIBS`) so the analysis path — vDSP FFT, peak tracking, segment
  pooling — resolves exactly as in the shipped binary (the test never calls synth;
  GPU is dead-but-linked, which keeps `HAS_METAL=1` desktop semantics honest).
- **`spectral_engine/CMakeLists.txt`** — one `include(...)` line for the new target.

## Result — the two paths are bit-identical on every fixture

```
full/fused analysis parity (sr=48000 n_fft=1024 hop=256 db=-80 n=16384)
  silence       count=0    max|d|: start/len/omega/amp/df/da all 0.000e+00  ok
  sine_bin      count=60   ...all 0.000e+00  ok
  sine_offbin   count=60   ...all 0.000e+00  ok
  two_tone      count=120  ...all 0.000e+00  ok
  chirp         count=60   ...all 0.000e+00  ok
  amp_ramp      count=60   ...all 0.000e+00  ok
full_fused_parity: all 6 fixtures within tolerance
```

Stronger than the spec's tolerance ask: full and fused agree to **0 ULP** on these
fixtures (identical FFT results feed the same tracker). The harness is non-vacuous —
silence is the only zero-count case (correct), the rest emit 60–120 real segments.
The documented tolerances remain as headroom for configs/fixtures where FP
reordering could legitimately diverge; a non-zero `max|d|` or a count mismatch is
surfaced as a FAIL, never silently absorbed.

## Verification

- `ctest --test-dir build`: **12/12 passed** (was 11 — `full_fused_parity` added,
  every prior test still green).
- `cmake --build build --target desktop`: builds clean (the CMake change is one
  additive `include`; the shipped desktop binary is untouched).
- Ran the binary directly to confirm real segment counts + exact-zero deltas (above),
  i.e. it is not vacuously green on empty arrays.

## Status

**D1 — LANDED.** Phase D now has its full/fused parity gate as a compiled CTest,
retiring the brittle `test_core_pass119_*_spec.py` string-matcher's role (its actual
deletion is folded into **D4**, the string-grep retirement + CI fix). Remaining
Phase D, in order:

- **D2** — golden-vector oracle (commit canonical fixtures + per-field tolerances).
  *Design decision:* the fixture set and frozen tolerances define the numerical
  contract, so this wants maintainer sign-off on what to freeze.
- **D3** — LUT generator feedback loop (validate `core/spectral_lut.c`
  `spectral_lut_init_sine` AND `tools/.../lut_generator.py` against the goldens via
  `generators/native_bridge.py`).
- **D4** — retire the ~131 `tests/core_math/*.py` string-matchers + fix
  `.github/workflows/c-cpp.yml` (it targets non-existent `debian-latest` and calls
  `make check`/`distcheck` the CMake Makefile never defines). *High blast radius:*
  deletes tests + edits CI — wants explicit go-ahead before executing.
- **D5** — backfill every Phase A/B/C finding into a compiled regression test.

See [[campaign2-ultraplan]], [[ultraplan-before-execution]], [[avoid-assumptions]],
[[minimal-decline-on-data]].
