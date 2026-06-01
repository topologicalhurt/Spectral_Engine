# Patch notes — Pass 170: CTF sweep increment 10 — binary deserialization / converter cluster (Phase C)

## Problem

Phase C is the CTF/KISS adversarial defect sweep: capture every latent defect in
`core/`, `analysis/`, `synth/` (and the CLI/converter layer) and fix it in place.
This pass sweeps the **binary deserialization / converter cluster** — the on-disk
segment-cache index/data reader (`core/spectral_seg_cache_fs.c`,
`core/spectral_seg_cache.c`), the SPEC `.bin` segment parser
(`core/spectral_segment_parser.c`), the SPEC→SPQ Q15 converter
(`cmd/convert_segments.c`), and the shared contract validators
(`core/spectral_contracts.h`) and primitives (`core/spectral_endian.h`,
`spectral_size_*` / `spectral_array_bytes`).

The one defect is a **missing little-endian byte-swap on the `.bin` load path in
the converter** — a divergence from the documented on-disk contract that every
sibling loader honors.

The SPEC `.bin` format is contractually little-endian:

```text
spectral_segment_parser.h:3   "File format: SegmentFileHeader followed by Segment array in little-endian."
spectral_segment_parser.c:6-7 "Endianness: Files are always stored in little-endian format.
                               On big-endian hosts, byte-swapping is performed during load/save."
```

`segments_load` honors this (`header_from_le(&hdr)` + per-segment
`spectral_segment_swap_endian`), and the segment cache honors it
(`seg_cache_header_swap` / `seg_cache_entry_swap` / `spectral_segment_swap_endian`).
But `convert_segments.c` reads the same `.bin` with **raw `fread` and never swaps**:

```c
/* cmd/convert_segments.c — load_and_validate_header (pre-fix) */
fread(header, sizeof(*header), 1, fin);          /* no swap */
... validate header->version / sr / stretch / pitch ...   /* on raw LE bytes */
...
fread(float_segs, sizeof(Segment), requested_count, fin); /* no swap */
```

On a big-endian host the un-swapped `header->version` byte-reverses, so the
`version != SEGMENT_FILE_VERSION` check rejects every valid file: `convert_segments`
is **unusable on BE** even though the format is portable and the rest of the toolchain
reads/writes it correctly. (It fails safe — it rejects rather than emitting garbage —
but it is still a real, reachable contract violation by the project's own standard:
the codebase invests in BE correctness everywhere else.) This is the same
divergent/missing-handling defect class as the campaign's prior fixes.

## Change

```text
1. Missing LE byte-swap on the converter's .bin load path  (divergent handling)
   cmd/convert_segments.c
   - load_and_validate_header: after the (endian-independent) magic check and
     BEFORE the version/sr/stretch/pitch validation, convert the header to native
     order with spectral_segment_file_header_swap_le(header).
   - main: after the segment fread, swap each read segment to native order on BE:
       if (spectral_is_big_endian())
           for (i in [0, read_count)) spectral_segment_swap_endian(&float_segs[i]);
   Both swaps are guarded by spectral_is_big_endian(), so on a little-endian host
   they are no-ops and the converted .spq is byte-identical to pre-fix output.

2. Single-source-of-truth header swap (avoid a new divergent duplicate)
   core/spectral_segment_parser.h
   Added a header-only `static inline void
   spectral_segment_file_header_swap_le(SegmentFileHeader*)` (symmetric, no-op on
   LE, char[4] magic intentionally untouched). It is header-only deliberately:
   convert_segments links only convert_segments.c + spectral_utils.c +
   spectral_log.c (it does NOT link spectral_segment_parser.c), so a non-inline
   export would not be reachable. The parser now includes spectral_endian.h.

3. Dedupe the parser's private swap to the shared helper
   core/spectral_segment_parser.c
   Deleted the static header_to_le / header_from_le and routed both call sites
   (segments_save, segments_load) through spectral_segment_file_header_swap_le.
   Behaviour-identical (same field swaps, same LE no-op); removes the duplicate so
   the swap semantics cannot drift between the parser and the converter.
```

## Why this is correct and behaviourally inert on the verification host

Every swap added or routed through `spectral_segment_file_header_swap_le` /
`spectral_segment_swap_endian` begins with `if (!spectral_is_big_endian()) return;`.
On the little-endian build/test host `spectral_is_big_endian()` returns 0, so:

- `convert_segments` executes the identical instructions it did before (the swap
  calls return immediately), and the parser's save/load paths are unchanged.
- The behaviour change is confined to big-endian hosts, where the converter now
  reads the documented LE on-disk layout correctly instead of rejecting it.

The `.spq` *output* path (`write_spq_file`) is intentionally left as native-order:
the `.spq` format has no documented endianness contract and is produced and consumed
exclusively by little-endian targets (desktop writer, Cortex-M reader). Normalizing
it would be a speculative change against an undocumented contract, out of scope for
this documented-`.bin`-load fix.

## Finding

Audited and left unchanged (no defect) — the rest of the cluster is solid:
- `spectral_seg_cache_fs_index_load` — validates magic, version, and the EXACT
  count-derived file size (`sizeof(header) + count*sizeof(entry)`, overflow-guarded)
  before `spectral_malloc_array`, so a corrupt count cannot drive a huge allocation;
  swaps every entry on BE; the count==0 case requires a header-only file.
- `seg_cache_entry_metadata_valid` + `seg_cache_validate_data_extent` — range-check
  sample_rate/stretch/pitch, the uint64→size_t output_length round-trip, and the
  seg/tile consistency, then prove `data_offset + total_data_bytes <= data_file_size`
  (overflow-guarded) before any mmap/read.
- `seg_cache_validate_tile_blob` + `spectral_gpu_tile_layout_words_valid` — recompute
  the expected blob size from the on-disk tile header, require an exact match and
  metadata agreement, then validate the tile ranges are contiguous (`start ==
  running_refs`), sum to exactly `total_refs`, and that every segment id `< seg_count`.
- `spectral_segment_array_payload_valid` / `spectral_segment_gpu_array_matches_segments`
  — full finite gauntlet on all 8 Segment fields (start/length/omega non-negative)
  and exact field equality between each stored SegmentGpu and its base Segment, so a
  tampered cache where the two disagree is rejected.
- the mmap vs heap lookup paths and `spectral_seg_cache_result_free` — mmap-backed
  results unmap and return early (tile pointers point INTO the mapping, never freed);
  heap results NULL the tile pointers (free(NULL) safe); no double-free either way.
- `spectral_omega_to_q88` (>255 divided by 4, then clamped to 255 -> max
  255*256 = 65280 < 65536, no uint16 overflow) and `spectral_phase_rad_to_q15`
  (normalized to [0,1) with the documented n>=1.0 round-up guard so the int16 cast
  cannot go out of range).
- the foundational primitives `spectral_size_add` / `spectral_size_mul` /
  `spectral_array_bytes` (builtin-overflow path with a correct manual fallback) and
  the `spectral_swap_u32/u64/float` / `spectral_segment_swap_endian` helpers
  (byte-exact, union-based float pun).

## Verification

```text
- five production targets build clean: desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float — plus the EXCLUDE_FROM_ALL convert_segments
  utility. Only the pre-existing benign -mavx2 / -mno-avx512f unused-command-line-arg
  notes on host; no new warnings (the parser dedupe removed two functions cleanly).
- ctest: 4/4 PASSED — arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift.
- converter byte-parity (the fix is a no-op on the LE host, proven empirically):
  produced a 340-segment output/segments.bin (desktop backend=export on
  resources/testing/sin_440hz.wav, n_fft=1024 hop=256 thresh=-70), then converted it
  with a PRISTINE (git-stashed, HEAD) convert_segments and with the FIXED build:
    pristine .spq sha == fixed .spq sha  (072914858ee0...)  -> BYTE-IDENTICAL.
- functional parity: desktop analysis unchanged (340 segments) — the parser change
  is a behaviour-preserving dedupe (LE no-op).
```

## Scope (Phase C increment)

Binary deserialization / converter cluster, one defect fixed: `convert_segments`
now honors the documented little-endian `.bin` on-disk contract (header + segments
swapped on load) via a single-source-of-truth inline shared with the parser, killing
the divergence from `segments_load` / the seg cache with no behavioural change on the
little-endian host (byte-identical `.spq`). The seg-cache deserializer, tile/segment
validators, Q15 encoders, and the size/endian primitives were audited and are clean.
With this increment the Phase C sweep has cleared fixed-point (161), analysis/peak-
track (162), port/SIMD/out (163), hashing/parsing/path (164), DSP-math/FFT-scaling +
alloc/cache (165), synth-backends + analysis-orchestration (166), CLI/orchestration
(167), embedded fade envelope (168), core synth dispatch/internal helpers (169), and
the binary-deserialization/converter surface (170). Phase D (compiled harness + LUT
golden-vector loop) follows.
