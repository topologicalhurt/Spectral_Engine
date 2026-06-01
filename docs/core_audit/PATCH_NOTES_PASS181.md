# Patch notes — Pass 181: CTF sweep increment 21 — Q15 ARM32 fixed-point synth cluster (defect fixed: expired-segment prune ordering drops boundary-aligned partials) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. This pass audits the **embedded Q15
fixed-point synthesis engine** — the live audio path on Cortex-M (Daisy / STM32H7) and
the path the compiled `arm32_process_correctness` CTest exercises:

```text
- synth/math/spectral_q15.h               float<->Q15/Q31 converters, phase_rad_to_q15,
                                          omega_to_q88, saturating Q15/Q31 primitives,
                                          smlad/smulbb (ARM-DSP + portable fallback)
- synth/backends/arm/spectral_synth_arm32.c
                                          load-time segment validation, active-segment
                                          lifecycle (activate / prune / synthesize),
                                          phase accumulator + linear-fade synthesis,
                                          Q30->Q15 master-gain reduction (DEFECT HERE)
```

**Outcome: one real defect found and fixed** — the active-segment list pruned expired
segments *after* the activation scan, so a slot held by a segment that ended exactly at
the block boundary could block a new segment the loader had validated as fitting,
dropping it. The Q15 math layer (`spectral_q15.h`) is clean.

## The defect — expired actives are pruned after activation, violating the validated overlap model

### The two halves disagree on what "simultaneously active" means

`spectral_arm32_validate_segment_data` (the load boundary) bounds the number of
simultaneously-active segments against `SPECTRAL_ARM32_MAX_ACTIVE` (512) using a
**half-open** overlap model:

```c
while (first_live < i) {
    ... first_end = end(segments[first_live]) ...
    if (first_end > start) break;   /* strictly-past: a seg ending AT `start` is NOT live */
    first_live++;
}
if ((i - first_live + 1u) > SPECTRAL_ARM32_MAX_ACTIVE) return SPECTRAL_ERR_OVERFLOW;
```

So an input with 512 segments ending exactly at sample `X` plus a new segment starting
at `X` is **valid** — at `X` only one segment (the new one) is live.

But `spectral_arm32_process` ran, in order:

```text
1. activation loop   while (next < num_segments && num_active < MAX_ACTIVE) { ... activate ... }
2. processing loop   for each active i: if (out_pos >= seg_end) remove; else synthesize;
```

Expired segments (`seg_end <= out_pos`) from prior blocks were removed only in step 2 —
*after* step 1 had already counted them against the `num_active < MAX_ACTIVE` cap.

### The drop

At the block whose `out_pos == X`:

```text
- The 512 segments ending at X are still in the active list (seg_end == X <= out_pos,
  expired but not yet pruned). num_active == 512 == SPECTRAL_ARM32_MAX_ACTIVE.
- The activation gate `num_active < SPECTRAL_ARM32_MAX_ACTIVE` is false, so the new
  segment starting at X is NOT activated and next_seg_idx is not advanced.
- The processing loop then prunes the 512 expired entries — too late.
- By the next block (out_pos == X + num_samples) the new segment's own window may already
  be in the past, so it renders a truncated tail or nothing: a dropped partial / lost
  onset — for a configuration the loader explicitly accepted.
```

This is reachable for dense, hop-aligned content (≈512 simultaneous sinusoids is
ordinary for noisy/polyphonic spectra, and STFT segments are hop-aligned so segment
ends and block boundaries coincide). It is a correctness divergence between the
validated contract and the runtime, not a memory-safety bug (output stays bounded).

### The fix

Prune expired actives **before** the activation scan, so runtime occupancy matches the
loader's half-open model. A new `spectral_arm32_prune_expired_active(ctx, out_pos)`
drops every active with `seg_end <= out_pos` (swap-with-last, SoA and AoS), and is
called immediately before the activation loop. The now-redundant expiry removal inside
the processing loop is deleted — after prune + activation every active provably
satisfies `seg_end > out_pos` (activation only admits segments with `seg_end > out_pos`
at line 808), so the processing loop is now a pure synthesis loop with a non-empty block
range for every entry.

```text
out_pos = output_position
prune_expired_active(out_pos)          // NEW: free slots of segments ending <= out_pos
activation loop (num_active < MAX)     // now sees true live occupancy -> admits the new seg
processing loop                        // all actives have seg_end > out_pos
```

## What else was checked and is correct (no change)

```text
- spectral_q15.h: float_to_q15/q31 clamp at +/-1 before the cast (no overflow at the
  boundary); phase_rad_to_q15 keeps n in [0,1) (documented n+=1 rounding guard) so
  (n-0.5)*65536 stays in int16; omega_to_q88's /4 prescale for omega>255 is unreachable
  for audio (omega = 2*pi*f/fs < pi) and saturates cleanly; portable smlad wraps in
  uint32 to match ARM's non-saturating accumulator and avoid signed-overflow UB; mul_q15
  saturates the lone -1*-1 overflow; all saturating add/sub clamp to [Q15_MIN,Q15_MAX].
- arm32 phase accumulator: phase_acc = (phase_q15+32768)<<16 stays in uint32; per-sample
  and fast-forward (sample_offset*freq_inc) phase math is unsigned and intentionally
  wraps mod 2^32 (correct for a periodic phase); amp fast-forward uses int64 then clamps.
- block-range math: after prune+activation, blk_start in [0,num_samples), blk_end in
  (blk_start, num_samples], len = seg_length > 0; accum[256] writes stay in [0,num_samples).
- fade partition (synth_segment_m7 / generic): fade_len = min(FADE_SAMPLES_EMBEDDED,
  seg_length/2) >= 1 so Q15_MAX/fade_len is divide-safe; fade_val ramps stay within q15
  (max seg_offset/samples_into_fade < fade_len <= 32, product < Q15_MAX); fi_end/fo_start
  clamped to [blk_start, blk_end]; seg_fade_out_start = seg_length - fade_len >= 0.
- freq_inc = freq_q88 * freq_inc_scale_q24 is a defined unsigned multiply; for valid audio
  freq_q88 < ~804 so no wrap, and a corrupt out-of-range freq_q88 only aliases (bounded),
  never UB. DMA prefetch path (get_segment / dma_rx_sync) bounds the cache-invalidate
  range and falls back to SDRAM until the transfer is coherent.
```

## Verification

```text
- Embedded-compiled code changed (spectral_synth_arm32.c, inside #if SPECTRAL_EMBEDDED).
  The desktop target compiles this file to nothing, so the desktop host binary is
  byte-identical to Pass 180 by construction.
- Functional parity for the embedded/sim path: for any input below the 512 active ceiling
  the prune removes exactly the entries the old processing-loop check removed (same
  seg_end <= out_pos condition), and synthesis is unchanged -> rendered samples identical.
  Only the at-ceiling, boundary-aligned case changes: a previously-dropped partial is now
  rendered. The arm32_process_correctness CTest (real spectral_arm32_process, AoS path)
  stays green, confirming no regression below the ceiling.
- Full triad re-run:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
```

## Phase C status

With this increment the sweep has cleared 161-180 (see prior notes) and now the Q15
ARM32 fixed-point synthesis cluster (181 — the active-segment list pruned expired
segments AFTER the activation scan, so a slot held by a segment ending exactly at the
block boundary could block a new, loader-validated segment and drop it; FIXED by pruning
expired actives before activation so runtime occupancy matches the validator's half-open
overlap model, with the now-redundant processing-loop removal deleted; the spectral_q15.h
math layer is clean). Phase D (compiled harness + LUT golden-vector loop) follows.
