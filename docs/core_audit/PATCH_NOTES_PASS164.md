# Patch notes — Pass 164: CTF sweep increment 4 — hashing/parsing/path cluster (Phase C)

## Problem

Phase C is the CTF/KISS adversarial defect sweep: capture every latent defect in
`core/`, `analysis/`, `synth/` and fix it in place. This pass sweeps the
**hashing/parsing/path cluster** — resource identity (`core/spectral_resource_fs.c`
path canonicalization + FNV-1a id), the xxhash file-method adapter
(`core/spectral_hash_xx32_xx3.c`), the binary segment file format
(`core/spectral_segment_parser.c`), the desktop→Q15 converter
(`cmd/convert_segments.c`), the segment pool / thread-safe array
(`core/spectral_segment_{pool,mt}.c`), and the canonical segment math header.

The cluster is almost entirely Campaign-1 hardened (overflow-checked sizes,
finite/range validation, endian handling, version gating). The one real defect is
a dead, **divergent duplicate** of the canonical segment-payload validator that
also trips `-Wunused-function`.

## Change

```text
1. Dead divergent duplicate validator  (drift hazard + compiler warning)
   core/spectral_segment_parser.c  (static segment_validate)
   segment_validate() validated the same eight Segment fields (start/length/phase/
   omega/df/amp/da/width finite + start/length/omega nonnegative) that the canonical
   contract spectral_segment_payload_valid() (core/spectral_contracts.h) already
   defines. It had NO caller — segments_validate_all() (the only validation entry in
   this file) delegates to spectral_segment_array_payload_valid(), the contract
   wrapper. So segment_validate() was:
     (a) dead code — it emits -Wunused-function under the project's -Wall -Wextra
         (confirmed: spectral_segment_parser.c:39: warning: unused function
         'segment_validate'); the dead-strip is why no binary carried it; and
     (b) a latent drift hazard — a byte-for-byte copy of the canonical contract that
         a future edit could silently diverge from (exactly the duplication the KISS
         sweep targets).
   Fix: delete segment_validate(). The canonical contract is the single source of
   truth; the file already uses it.
```

## Finding

Audited and left unchanged (no defect) — the rest of the cluster is solid:
- `core/spectral_resource_fs.c` — the five-transform path canonicalization
  (lowercase / strip-controls / separator-normalize / component-resolve / RLE) is
  byte-for-byte matched to the Python generator and already hardened across prior
  passes: the `comp_starts[256]` bound (`comp_count < 256u`), the trailing-space-
  before-`..` NTFS guard, the RLE never-fall-back-to-literals rule, and the
  truncate-don't-misencode boundary checks are all correct. FNV-1a is canonical.
- `core/spectral_hash_xx32_xx3.c` — XXH3 (host) / XXH32 (embedded) lifecycle
  adapter; the full-direct path guards `end_pos < start_pos`, `start_pos >
  INT64_MAX` (before the `(int64_t)` seek cast) and `total_len_u64 > SIZE_MAX`
  (falls back to streaming rather than narrowing); `spectral_hash_oneshot` handles
  NULL/zero-length without UB.
- `core/spectral_segment_parser.c` (rest) — save/load are overflow-checked
  (`spectral_array_bytes` + `spectral_size_add`), finite-validated, version-gated,
  and endian-correct; `file_size != expected_file_bytes` rejects truncated/corrupt
  files before allocation.
- `cmd/convert_segments.c` — every float→fixed conversion routes through the
  pass-161-hardened saturating helpers (`OMEGA_TO_Q88` clamps to [0,255]·256 with
  the >255 /4 path; `FLOAT_TO_Q15`/`PHASE_RAD_TO_Q15` saturate and reject
  non-finite); start/length clamps and the `seg_end` overflow saturation are
  correct; `compute_output_length` does the stretch multiply in double with range
  checks.
- `core/spectral_segment_pool.c` — block-chain growth is overflow-guarded
  (`new_max < max_blocks` rejects the doubling wrap); `to_array` clamps the last
  partial block and overflow-checks the copy.
- `core/spectral_segment_mt.c` — mutex-guarded pending/active swap; `_copy` does an
  overflow-checked deep copy; `_get` is a documented shallow borrow.
- `core/spectral_segment_math.h` — version-locked canonical quadratic-phase model
  (phase0 + αt + βt², mirrored in the Metal shader with a compile-time version
  check); internally consistent and pinned by arm32_process_correctness.

Also math-reviewed this pass (adjacent analysis math, no defect): the peak
interpolators in `analysis/spectral_peak_estimator.c` were re-derived from their
papers — Jacobsen sign `Re{(X[k-1]-X[k+1])/(2X[k]-X[k-1]-X[k+1])}`, Candan
`tan(π/N)/(π/N)` correction, magnitude-parabolic `0.5(α-γ)/(α-2β+γ)`, Quinn-second's
asymmetric `dp=-ap/(1-ap)` / `dm=am/(1-am)` with `+τ(dp²)-τ(dm²)` and the τ kernel
`0.25·log(3x²+6x+1) - (√6/24)·log((x+1-√⅔)/(x+1+√⅔))`, and the phase-vocoder
instantaneous-frequency relation — all correct.

## Verification

```text
- five production targets build clean: desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float. The segment_parser -Wunused-function warning is
  GONE; only the pre-existing benign -mavx2 / -mno-avx512f unused-command-line-arg
  notes remain on host.
- ctest: 4/4 PASSED — arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift.
- desktop, simulate AND simulate_daisy binaries are BYTE-IDENTICAL to the
  pre-change build (cmp clean): segment_validate had no caller and was already
  dead-stripped, so deleting the source changes no emitted code — it only removes
  the warning and the source-level drift hazard.
```

## Scope (Phase C increment)

Hashing/parsing/path cluster only — one dead-code removal (warning + divergent-
duplicate hazard), zero behaviour change (byte-identical). Remaining Phase-C
surface per ULTRAPLAN: allocation/pool/cache (seg_cache / allocators / perf model)
and a deeper DSP-math pass over windows + FFT scaling. Phase D follows the sweep.
