# Patch notes — Pass 186: CTF sweep increment 26 — synthesis backend dispatch + wavetable bank (load/HEX-parse/lookup interpolation) (clean audit) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. Two genuinely under-swept logic files
with real branching / parsing / interpolation arithmetic:

```text
- core/spectral_backend.c      vtable-driven backend capability queries + unified
                               synthesis dispatch with the CPU fallback chain
- core/spectral_wavetable.c    wavetable bank: builtin gen, .spwt load/save, raw +
                               Intel-HEX load, buffer load, float + Q-phase lookup
                               (linear interpolation)
```

**Outcome: clean audit. No defect found; no code changed.** The interpolation index
math (the classic wavetable bug surface — index wrap, fractional scaling, OOB on the
guard sample) and the Intel-HEX parser (the classic buffer-overrun surface) were
re-derived against the real configured constants (SIZE = 2048, BITS = 11) and against
the Intel-HEX record spec. Both correct. Per campaign protocol a clean audit is a
legitimate result and a defect must not be fabricated.

## What was checked and is correct

### Wavetable lookup — interpolation index/fraction/guard math

```text
- lookup_f(table, phase_norm): phase_norm -= floorf(phase_norm) then a defensive
  `+1 if <0` puts it in [0,1). idx_f = phase_norm * SIZE in [0,SIZE]; idx = (uint32)idx_f;
  frac = idx_f - idx is computed BEFORE the `if (idx >= SIZE) idx = 0` clamp, so the
  float-rounding case phase_norm -> exactly 1.0f (idx_f == SIZE) yields frac == 0 and a
  clamped idx 0, returning samples[0] exactly. For all other inputs idx in [0,SIZE-1] so
  samples[idx+1] tops out at index SIZE = the wrap-guard slot (samples[SIZE] = samples[0]).
  No OOB; correct convex interpolation.
- lookup_q(table, phase_u16): idx = phase_u16 >> (16-BITS) = phase_u16 >> 5 in [0,2047] =
  [0,SIZE-1]. frac = (phase_u16 & 31) then, since frac_bits(5) < 8, `frac <<= 3` -> max 248.
  248/256 == 31/32 is the correct exclusive-upper weight for a 5-bit fraction (the 32nd
  sub-step would increment idx, so frac never reaches 256). Verified the math is correct for
  ALL frac_bits regimes (>8: right-shift to 8-bit; <8: left-shift up; ==8: identity).
- lerp_q8 (fixed): s0 + (((s1-s0)*frac_q8)>>8). (s1-s0) in [-65535,65535] * frac_q8(<=255)
  ~ 1.67e7 fits int32; result is a convex combination of s0,s1 so stays within Q15 range —
  no overflow/saturation needed. lerp_f mirrors it via fq8 = (int)(frac*256).
```

### Wavetable load paths — bounds + overflow + leak-free error paths

```text
- spectral_wavetable_load (.spwt): validates magic/version/size(==SIZE)/format/timbre_id,
  computes payload + expected file bytes via spectral_array_bytes / spectral_size_add
  (overflow-checked), requires file_size == expected EXACTLY, then one of three checked
  temp-buffer conversion paths (same-format / float->Q15 / Q15->float), each finite-validated
  and freeing temp on every error path. memcpy is bounded by payload_bytes = SIZE*elt into the
  SIZE+1 table; wrap-guard samples[SIZE]=samples[0] set after.
- load_raw / load_buffer: size must equal/meet SIZE*elt; finite-validated; bounded memcpy.
- load_hex: parse_hex_line enforces line[0]==':' , line_len>=11, byte_count <= data_capacity
  (data[32]), required_len == 11 + byte_count*2 EXACTLY (so the data + checksum reads stay in
  the line), and the two's-complement checksum == 0. Data records bound every write by
  `offset > expected_bytes || data_len > expected_bytes - offset` (no integer-underflow in the
  RHS because the LHS already rejected offset > expected_bytes), track per-byte coverage in a
  calloc'd written[] sized expected_bytes, and require covered_bytes == expected_bytes + a
  type-01 EOF record before committing. All error paths free both temp_table and written[].
```

### Backend dispatch — fallback chain + bounds

```text
- spectral_backend_name: names[5] indexed by (unsigned)backend with `idx <= BACKEND_EXPORT`
  (==4) guard; negative backend wraps huge -> "Unknown". No OOB.
- supports_timbre: rejects timbre_id < TIMBRE_SINE, requires vt->id == backend for concrete
  backends (AUTO/EXPORT use the virtual fallback vtable by design), then timbre_id <= max_timbre.
- spectral_synth_dispatch_ex: clamps n_threads>=1; AUTO/EXPORT -> select_for_timbre; then a
  single-direction fallback CASCADE to CPU on every failure (backend-not-compiled vt->id!=backend,
  !available, unsupported timbre, GPU+wavetable-bank, or synth returning non-OK), each logging a
  resolution event. out_effective_backend/timbre are always written to the actually-used path.
  bounded snprintf into reason[192]. No leak, no dispatch to an unsupported/unavailable backend.
```

## Verification

```text
- No source changed this pass (read-only audit). The host triad was re-run on the current
  tree this session and is green:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
```

## Phase C status

With this increment the sweep has cleared 161-185 (see prior notes) and now the synthesis
backend dispatch + wavetable bank cluster (186, clean — the float and Q-phase lookups compute
idx/frac/guard correctly with no OOB and exclusive-upper interpolation; the .spwt/raw/HEX/buffer
loaders are overflow-checked, exact-size-validated, and leak-free on every error path with a
bounds-checked Intel-HEX parser; the backend dispatch is a correct single-direction CPU-fallback
cascade with no OOB name lookup). All major compute, support, and now the dispatch/wavetable
surfaces are audited. Phase D (compiled harness + LUT golden-vector loop) follows.
