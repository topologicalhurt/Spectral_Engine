# Patch notes — Pass 196: Optimisation track O1-B — CPU output tiling replacing the full private-buffer reduce (bit-identical)

## Scope

Third implementation pass of the **optimisation track** (`docs/core_audit/OPTIMISATION_PLAN.md`).
Implements **O1-B** (Tier 1/2): "Replace O(threads×len) private buffers + reduce pass
(`spectral_synth_cpu.c`) with disjoint output-range tiles, one write/sample — port the
engine's own GPU tiler. Flag: none (strict win). Risk: low-med."

The plan's literal suggestion was to port `gpu_tile_preprocess`. That kernel assigns
segments to tiles with an `omp atomic capture` counter, so the **per-tile segment order is
non-deterministic** — porting it directly to the CPU would make output depend on thread
scheduling and force a ≤1 ULP signed-off golden. Instead this pass keeps the existing
**index-range partitioning** (already deterministic) and tiles the *output*: each partition's
segments occupy a contiguous output window, so per-thread buffers shrink from `out_len` to
the widest window. This delivers the same O(threads×len)→O(threads×max_span) memory and
combine-bandwidth win **with no reproducibility hazard, no flag, and bit-identical output** —
better than the plan's ≤1 ULP estimate (PASS195's forward note anticipated a golden change;
the realised design needs none).

## Design — provably bit-identical to the legacy full reduce

```text
Partitioning is unchanged: partition p owns the contiguous segment-index range
    [p*count/n_parts, (p+1)*count/n_parts).
New span pre-pass (parallel over p, same segment_loop_params_init as the synth loop):
    span_lo[p] = min(start_idx)         over valid segments in p
    span_len[p] = max(start_idx+length) - span_lo[p]
    max_span   = max(span_len[p])       over all p
Per-thread buffers are sized to max_span (was out_len).  A segment writes at the
RELATIVE offset (start_idx - span_lo[p]); the combine pass adds each buffer back at
its absolute span_lo[p] offset, in ascending partition order.
```

**Bit-identity proof (vs. the pre-O1-B full reduce, which zeroed each out_len buffer,
accumulated `out[j] += …` at absolute `j`, then summed all buffers `bufs[0]+bufs[1]+…`):**

```text
1. Same value per element.  A segment computes the identical samples regardless of the
   buffer base; bufs[t] at relative k == legacy bufs[t] at absolute span_lo[t]+k.
2. Outside its window a partition contributed exactly +0.0 in the legacy reduce (the
   buffer was +0.0-initialised and nothing was added there).  The combine skips those
   indices; x + 0.0 == x is an exact IEEE identity, and the running accumulator is never
   -0.0 (it starts +0.0 and -0.0 is unreachable from a +0.0-init sum — same argument as
   the F1/F3 phase-helper proof, PASS194).
3. Order preserved.  Both old and new combine the partitions in ascending index order, so
   for any output sample touched by multiple (possibly overlapping) windows the addition
   order is unchanged — this also holds for the saturating-Q15 native path, where
   SPECTRAL_SAMPLE_ADD is non-associative: SAT(x,0)=x and the sequence of non-zero adds is
   identical, so the native combine is integer-exact too.
```

Overlapping windows are handled correctly (the combine adds at absolute offsets, exactly as
the legacy reduce did); disjointness is the common case but not required for correctness.

## Files changed

```text
- synth/backends/cpu/spectral_synth_cpu.c
    ThreadBuffers                         + const size_t* span_lo, span_len (driver-owned)
    thread_buffers_reduce_float   -> thread_buffers_combine_float   (add span at offset)
    thread_buffers_reduce_native  -> thread_buffers_combine_native  (add span at offset)
    reduce_{float,native}_wrapper         retargeted to the combine functions
    synth_cpu_driver                      + span pre-pass; alloc max_span (was out_len);
                                          write at (start_idx - span_lo[p]); max_span==0
                                          short-circuit to silence; free span arrays.
    thread_buffers_alloc/free             unchanged (Pass-22 overflow guards intact; the
                                          combine bounds-check len against buf_size).
```

The audited `thread_buffers_alloc` overflow arithmetic is untouched — it is simply called
with `max_span` instead of `out_len`. The combine functions retain the
`len > buf_size/sizeof(elem)` and `lo+len <= out_len` guards, so a malformed span can only
return `SPECTRAL_ERR_OVERFLOW`, never write out of bounds.

## Verification

```text
- Five production targets build clean (desktop, simulate, simulate_daisy, embedded_arm,
  embedded_arm_float) — only the pre-existing benign -mavx2 / -mno-avx512f notes.
- ctest: 4/4 PASSED (arm32_process_correctness — the integer-exact Q15 native-path gate,
  core_contracts, core_guarantees, core_guarantees_drift).
- End-to-end bit-identity: built pristine HEAD (c7eff0f0c8, pre-O4-B/pre-O1-B) in a detached
  git worktree, ran the CPU backend (sin_440hz.wav, fft=4096 hop=128), and compared:
    cmp output/out_c.wav  <pristine-HEAD output>  ->  byte-identical.
  This isolates the entire uncommitted default-path stack (O4-B restrict + O1-B tiling +
  PASS194's default-off cubic evaluators) and shows zero output change.
- Golden build/golden/cpu_sine_ref.wav restored from the pristine-HEAD reference (the prior
  copy was lost to build-dir churn); it is the canonical pre-optimisation CPU-sine reference.
```

## Status

O1-B implemented and verified **bit-identical** (no golden change, no flag — a strict win).
Memory/bandwidth on the CPU synth combine drops from O(threads×out_len) to
O(threads×max_span). Per exec order, next is **O4-A** (consume the chirp slope on the ARM
hot path, `spectral_synth_arm32.c`), the embedded enabler for F1.
