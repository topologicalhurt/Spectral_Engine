# Patch notes — Pass 188: CTF sweep increment 28 — Daisy Seed firmware glue + UART command protocol (clean audit; one deferred SD-load validation observation) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. This pass audits the last genuinely-
unswept file with real untrusted-input logic (0 prior PATCH_NOTES mentions):

```text
- api/daisy_seed/daisy_seed_spectral.c   Daisy Seed bare-metal wrapper:
                                         init/SD-load/buffer-load, ADC->param mapping,
                                         playback control, q15 process passthrough, and
                                         a UART byte-stream command state machine
```

**Outcome: clean audit. No defect fixed; no code changed.** The UART protocol
parser — the one true adversarial surface here (bytes arrive from an external host)
— was re-derived state-by-state and is memory-safe. One robustness observation (the
SD `.spq` loader commits segments without re-validation) is recorded as a deferred,
maintainer-directed item: it is bounded (no memory unsafety — see below), and the
file is in **no host-buildable target** (arm-none-eabi + FATFS, absent on this host),
so a blind edit would be unverifiable — the same disposition this campaign has
applied to every other unverifiable-on-host surface (A2/A3, the GPU fade-tail, the
daisy section binding).

## What was checked and is correct

### UART command state machine — bounded, NUL-safe, traversal-guarded

```text
- DATA-state buffer fill (daisy_uart_process_byte): `if (data_len >= DAISY_UART_MAX_MSG_LEN)`
  is checked BEFORE every `data[data_len++] = byte`, and data_len is uint8_t indexing a
  uint8_t[DAISY_UART_MAX_MSG_LEN] buffer — no overrun, no wrap past the guard.
- Fixed-length commands (SET_STRETCH/SET_AMP=4, SEEK=4) transition to CHECKSUM only once
  `data_len >= expected_len`, so uart_execute's `memcpy(&x, data, 4)` always reads 4 valid
  bytes. PING/RESET/GET_STATUS have expected_len 0 -> straight to CHECKSUM (no data read).
- LOAD_FILE (variable length): the only path to uart_execute is the stored '\0' byte
  transitioning DATA->CHECKSUM, so fname = (char*)data is ALWAYS NUL-terminated within the
  buffer (the terminator sits at data[data_len-1] <= MAX_MSG_LEN-1). A filename that fills
  the buffer without a NUL hits the `data_len >= MAX_MSG_LEN` overflow reset first.
- Path-traversal reject loop: `p[0]=='.' && p[1]=='.' && (p[2]=='/'||'\\'||'\0')`. p[2] is
  read ONLY when p[0],p[1] are both '.', i.e. both non-NUL, so the NUL is at p[2] or later;
  the maximum p[2] index is the NUL itself (in bounds). No off-by-one OOB read. Catches
  "..", "../", "..\", and embedded "/../".
- CHECKSUM compare uses the running XOR (seeded with cmd, XORed with each data byte); a
  mismatch emits RESP_ERR and returns to IDLE. IDLE accepts either SYNC (->CMD) or a direct
  command byte (lenient framing, by design) — both branches are side-effect-bounded.
```

### Glue / arithmetic — no overflow, no underflow

```text
- daisy_spectral_get_memory: total_used is clamped to <= DAISY_SDRAM_SIZE before
  `available = DAISY_SDRAM_SIZE - total_used`, so the unsigned subtraction never underflows;
  the NULL-ctx branch reports full SDRAM available.
- daisy_spectral_load_sd / _load_buffer: seg_bytes = num_segments * sizeof(SpectralSegmentQ15)
  with num_segments first rejected if > segments_capacity (== DAISY_MAX_SEGMENTS_SAFE, the
  pool size), so the read/copy fits the pool and the uint32 product cannot realistically
  overflow (capacity is SDRAM-bounded).
- ADC->param maps (set_params_adc) and set_stretch/set_amplitude CLAMP to
  [STRETCH_MIN,STRETCH_MAX] / [0,1] before handing to the embedded synth; init coerces an
  out-of-set sample_rate to the 48 k default; single-active-ctx guard prevents pool aliasing.
- daisy_spectral_process_q15/_interleaved/_reset/_seek/etc. are NULL-guarded passthroughs to
  the (already-audited, pass 181) spectral_arm32_* embedded core.
```

## Deferred observation — SD `.spq` load skips segment re-validation

`daisy_spectral_load_sd` validates the file header (magic, version, num_segments <=
capacity) and then `f_read`s the segment records **directly into the synth pool**,
committing `num_segments`/`output_length` and resetting playback **without** calling
the segment validator. The sibling `daisy_spectral_load_buffer` path delegates to
`spectral_arm32_load`, which DOES run `spectral_arm32_validate_segment_data`
(half-open active-overlap bound vs MAX_ACTIVE, field-range checks — see pass 181).

```text
- Severity: bounded to AUDIO QUALITY, not memory safety. The embedded runtime is already
  defensive against malformed segments: spectral_arm32_process gates activation by
  `num_active < MAX_ACTIVE` and prunes expired segments first (pass 181), so segments that
  violate the simultaneity bound are DROPPED, not written out of bounds; field reads stay
  within the fixed pool. A corrupt-but-header-valid file therefore yields wrong audio, not
  a crash/OOB.
- Why deferred (not fixed now): (1) the file compiles in NO host target — it needs
  arm-none-eabi + the Daisy/FATFS headers (fatfs.h, ff.h, daisy_seed_spectral.h) that are
  not present on this host, and the SD code is itself #ifdef DAISY_HAS_FATFS, so it cannot
  even be syntax-checked here; (2) the validator is `static` to spectral_synth_arm32.c, so a
  real fix means either exporting it (an API change to TRIAD-compiled code purely to serve an
  unbuildable consumer) or staging through a temp segment buffer (the SD path reads in-place
  precisely because embedded RAM may not hold a second copy). Both are unverifiable on this
  host. Per campaign discipline (measure, don't assert; no unverifiable changes) this is left
  for a maintainer-directed change once the Daisy toolchain/hardware is in the loop — the
  same disposition as the deferred GPU fade-tail-under-time-stretch item.
```

## Verification

```text
- No source changed this pass (read-only audit + one deferred note), so the Pass 187 green
  state is preserved by construction. The edited-in-187 file is the only delta this session;
  the triad was last run green this session after that fix:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
  (The Daisy firmware file is not a member of any of these targets, so it could not affect
  the triad regardless.)
```

## Phase C status

With this increment the sweep has cleared 161-187 (see prior notes) and now the Daisy
Seed firmware-glue + UART command-protocol surface (188, clean — the UART byte-stream
state machine is bounds-checked, NUL-safe and path-traversal-guarded; the SDRAM accounting
and ADC/param maps cannot under/overflow; the playback API is NULL-guarded passthrough to the
pass-181 embedded core). One bounded, unverifiable-on-host robustness observation is recorded
(SD `.spq` load skips re-validation; runtime activation gates keep it memory-safe), deferred
maintainer-directed alongside the GPU fade-tail item. All compute, support, dispatch, I/O,
debug-instrumentation, optional-processing, AND firmware-glue surfaces are now audited; the
host-verifiable kernel has no open defect leads. Phase D (compiled harness + LUT golden-vector
loop) remains the natural home for both deferred observations.
