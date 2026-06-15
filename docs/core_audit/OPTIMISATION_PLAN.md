# Optimisation Plan — minimal high-performance spectral-resynthesis kernel

Status: **PARTLY LANDED.** F1/F3 infra, O1-B output tiling, O2-A SIMD sine, and the ARM
coupled-form oscillator are in; the big OPEN decision is **F2 — synthesis method (oscillator-bank
vs inverse-FFT)** — engaged measure-first in [`REVIEWER_HANDOFF.md`](REVIEWER_HANDOFF.md) (the "F"
fork). Original plan text follows.
Owner track: optimisation. Companion to `docs/core_audit/ULTRAPLAN.md` (master plan).
Audit date: 2026-06-01. Evidence cited as `file:line` against the current tree.
Revision: **v2** — adds the SOTA survey (Part A) and an architectural-minimality stage
inventory (Part B), and re-orders the work so the two *foundational algorithm* decisions
(track formation; synthesis method) precede every micro-optimisation. Rationale: a
micro-optimisation applied on top of a non-minimal algorithm is wasted — the algorithm
choice cascades into every tier below it. v1's tiered micro-opts are retained but
re-scoped under the foundation decisions and tagged with their contingencies.

---

## 0. Relationship to the master plan (non-conflict statement)

This is a **separate, additive track**; it does not re-order, re-scope, or contradict any
phase of `ULTRAPLAN.md`. Single point of contact:

```text
ULTRAPLAN Phase D (compiled harness + golden-vector oracle, lines 997–1068)
        IS THE PREREQUISITE AND VERIFICATION GATE for this plan.
```

Goldens are captured from the **current, unoptimised** kernel FIRST (ULTRAPLAN line 79),
freezing today's numerical behaviour as the contract. Every change here is verified by
diffing against those goldens. Deliberate behaviour changes (the foundation work changes
the segment representation and may change reconstruction within tolerance) get a **new,
signed-off golden** — never a silently-moved tolerance. The remaining deferred ULTRAPLAN
observation (GPU fade-tail) stays Phase D's, not this plan's; its companion (Daisy .spq
re-validation) landed in review-3.

---

# Part A — SOTA survey & research plan

The engine is a **classical sinusoidal-modeling resynthesiser**: STFT → spectral-peak
detection → per-partial parameters → additive synthesis. The right SOTA frame of
reference is therefore the McAulay-Quatieri / Serra-Smith / PARSHL lineage, **not** neural
methods (DDSP et al. are a different engine class and out of scope). I surveyed each
pipeline stage against that lineage and the current literature. Verdict legend:
**[KEEP]** SOTA-correct; **[GAP]** naive or missing vs SOTA; **[MIN]** minimality issue
(redundant/dead stage).

### A1 — Windowing & STFT analysis  **[KEEP]**

Current: Hann window, real STFT, magsq+phase per frame (`spectral_analysis_full.c:21-45`;
fused variant in `spectral_analysis_fused.c`). This is textbook and correct; FFT is
vDSP/FFTW/CMSIS per platform. No algorithmic change warranted. (Minimality note on the
*two* STFT implementations → Part B.)

### A2 — Spectral-peak parameter estimation  **[KEEP]**

Current: `spectral_peak_estimator.c` implements and AUTO-selects among Quinn (1994),
Candan (2011), Jacobsen-Kootsookos (2007), Rife-Boorstyn (1974), and JOS quadratic
interpolation — all cited in its header (`spectral_peak_estimator.c:1-22`) and wired
(`:129` resolves AUTO to a concrete estimator). This is **already at SOTA** for
three-bin frequency estimation and is *not* the naive part of the pipeline.

```text
- Optional future refinement: XQIFFT (exponential-magnitude-weighted QIFFT; Werner &
  Germain, 2016) reduces stationary-sinusoid bias up to ~50× vs plain QIFFT. LOW
  priority — the existing estimators already beat naive parabolic, and estimator error
  is dwarfed by the track/synthesis gaps below. Park behind the existing AUTO selector
  if ever wanted.
```

### A3 — Track formation (peak continuation)  **[GAP — #1 algorithmic gap]**

Current: the tracker scans peaks in frame *t* (SIMD, `spectral_peak_track.c:954-976`) and
**matches** each against frame *t+1* (±1-bin nearest peak, `spectral_peak_interp.c:17`),
then emits a segment spanning exactly **one hop** (`spectral_peak_interp.c:200-201`,
`length = hop_float`). Crucially, **no track state is carried across frame-pairs** — the
OMP loop is `schedule(static)` over independent pairs (`spectral_peak_track.c:909`).

```text
SOTA (McAulay & Quatieri 1986; PARSHL/Smith-Serra 1987): the matched peaks must be LINKED
into continuous tracks across many frames (birth → continuation → death). One sustained
partial = ONE track, synthesised as a single time-varying sinusoid.

CURRENT = "MQ matching WITHOUT MQ linkage." A partial lasting F frames becomes F-1
independent one-hop grains instead of 1 track. This inflates BOTH the analysis output
(SegmentArray length, streamed by every backend) AND the synth voice-count by a factor of
the mean track length. It is the textbook naive form, and it is the foundation every
downstream micro-opt would otherwise sit on.
```

Borrow: **McAulay-Quatieri 1986** peak continuation. The hard part (the per-hop match,
incl. the chirp slope `df`) already exists; the missing part is *linking* matches into
tracks and emitting one multi-hop segment per track. This is exactly what the stubbed
`adaptive_track_density` stage (`spectral_proc_adaptive_track_density.c:4-13`, currently
`return SPECTRAL_OK`) was scaffolded for. **→ Foundation F1.**

### A4 — Parameter interpolation along a partial  **[GAP — coupled to A3]**

Current: per-grain quadratic phase `phase0 + α·j + β·j²` (`spectral_segment_math.h`),
linear amplitude, and a 3-region cross-fade overlap-add between adjacent grains
(`spectral_synth_arm32.c` fade regions; CPU/host equivalent). The cross-fade is the device
that hides phase discontinuity *because* there are no tracks — it is a symptom of A3.

```text
SOTA (MQ 1986): once peaks are linked into a track, phase is interpolated by a CUBIC
polynomial that simultaneously matches endpoint phase AND frequency (= phase derivative),
giving "maximally smooth" phase with no discontinuity and no cross-fade needed. Amplitude
is linearly interpolated along the track.
```

With A3 in place, switch per-track interpolation to **MQ cubic phase + linear amp**,
removing the per-grain cross-fade machinery (fewer ops *and* fewer artifacts). A cubic
phase polynomial is still evaluable by forward differences (3 adds/sample) — see the
recurrence-oscillator micro-opt, which composes with this. **→ Foundation F3.**

### A5 — Synthesis method  **[GAP — the dominant Big-O decision]**

Current: **oscillator/grain bank only.** Three backends — CPU (per-thread private buffers
+ reduce, `spectral_synth_cpu.c:188-254`), GPU (output tiling), ARM (LUT phase-accumulator
+ active-voice list). All are `O(#partials × output_len)`. **A tree-wide search confirms
no inverse-FFT synthesis exists anywhere** (`grep ifft|inverse.*fft` → empty).

```text
SOTA offers TWO synthesis families, with a sharp complexity crossover:

  (a) Oscillator / grain bank (MQ, PARSHL): cost ∝ (#partials × samples). Best for SPARSE
      spectra; lowest latency and memory; no per-frame buffering. The engine's current
      design (esp. the ARM LUT bank) is a good instance of this family.

  (b) Inverse-FFT "FFT⁻¹" overlap-add (Rodet & Depalle 1992; Freed-Rodet-Depalle 1993):
      place each partial's window main-lobe into a short-term spectrum, ONE inverse FFT per
      frame, overlap-add. Cost ∝ (#frames × N log N), INDEPENDENT of #partials. Reported
      ~order-of-magnitude (~15×) faster than the oscillator bank for DENSE spectra, and it
      makes filtered-noise / residual essentially free (added in the frequency domain — see
      A6). This entire family is ABSENT from the engine.
```

The minimal-yet-powerful foundation is **density-adaptive**: oscillator bank where
partials are sparse (and on embedded, where it is real-time and low-memory), inverse-FFT
where partials are dense. This crossover is precisely what the engine's **dead
`hybrid_render` parse token** (`spectral_processing_chain.c:31`) was scaffolded for — strong
evidence the maintainer already intended it. **→ Foundation F2.** This decision is what
makes the v1 micro-opts non-wasteful: they optimise the *oscillator-bank regime*, which
remains the correct tool for sparse/embedded — they are no longer "micro-opts on a naive
universal synth," but the right tuning for the regime where the oscillator bank is optimal.

### A6 — Residual / stochastic model  **[optional; out of core scope]**

Current: `spectral_proc_serra_smith_1990.c` is a no-op stub. SOTA is **Serra-Smith SMS
(1990)**: model the signal as sinusoids + a stochastic residual (the part the peaks don't
explain). Real feature, real scope — but it is *additive on top of* the core and is nearly
free under an inverse-FFT synth (the residual is shaped noise added in the frequency
domain, A5b). Recommendation: **do not build now**; record as named future work, and let
F2's synthesis-method choice keep the door open. Delete the no-op stub (Part B).

### A7 — Psychoacoustic partial pruning  **[optional; compounds A3/A5]**

Current: `spectral_proc_johnston_1988.c` is a no-op stub. SOTA is **Johnston (1988)**
masking: prune partials masked by louder neighbours, cutting track/voice count with no
audible change. It is a *track-count reducer* and therefore compounds the F1/F2 wins.
Recommendation: optional, after F1; record as named future work; delete the no-op stub now.

### A-research — open questions to settle empirically (not by assertion)

Per the project's measure-don't-assert principle, F1/F2 carry concrete measurements:

```text
Q1  Crossover P*: at what #partials-per-frame does inverse-FFT beat the oscillator bank
    on (i) desktop, (ii) ARM-M7? → benchmark both synths over a partial-density sweep;
    P* sets the hybrid switch threshold (F2).
Q2  ARM IFFT feasibility: is one CMSIS rfft/frame within the embedded block budget at the
    target N/hop? → measure cycles on the sim perf-model; if not, ARM stays oscillator-bank
    only and the hybrid is desktop/GPU-only.
Q3  Track-merge fidelity: does linking F grains into one cubic-phase track reconstruct the
    pre-linkage audio within the golden tolerance? → A3/A4 golden diff.
Q4  Phase-recurrence drift: over the longest expected track, does forward-difference phase
    stay within tolerance vs closed-form? → sets the re-anchor interval / precise flag.
```

---

# Part B — Architectural minimality: stage inventory

Full pipeline, each stage marked **necessary / fusible / redundant / dead**. Goal: the
*minimal* stage set that still supports the SOTA features.

```text
INPUT → [1] window → [2] STFT(FFT) → [3] peak scan → [4] peak estimate → [5] peak match
      → [6] TRACK FORMATION → [7] param interpolation → [8] (processing chain)
      → [9] synthesis backend → OUTPUT

[1] window                necessary    Hann; trivial. KEEP.
[2] STFT                  necessary    but TWO implementations: full_matrix (small) and
                                       fused spsc_pipeline (large), switched at
                                       SPECTRAL_STFT_CHUNK_THRESHOLD
                                       (spectral_analysis.c:152-160). The fused path is
                                       strictly more memory-minimal (no F×bins matrix).
                                       MIN: evaluate unifying on the fused path and
                                       retiring full_matrix (one stage, less code) unless
                                       full_matrix's random-access is needed by a
                                       multi-pass estimator. Investigate, don't assume.
[3]+[4] scan+estimate     necessary    SIMD scan + SOTA estimators. KEEP (A2).
[5] peak match            necessary    MQ matching; already present. KEEP.
[6] TRACK FORMATION       MISSING      the #1 gap (A3). Must become first-class — tracks
                                       are the natural analysis output; one-hop grains are
                                       a non-minimal fallback. → F1.
[7] param interpolation   present      per-grain quadratic+crossfade; upgrade to per-track
                                       cubic phase once [6] exists (A4). → F3.
[8] processing chain      DEAD/redundant 3 no-op stages + 5 stage-less parse tokens
                                       (spectral_processing_chain.c:26-41). Pure indirection
                                       today. adaptive_track_density ← becomes F1;
                                       hybrid_render ← becomes F2; the rest delete or record
                                       as named future work (A6/A7). → Part C / O5.
[9] synthesis             present      oscillator-bank only; add density-adaptive IFFT
                                       (A5). The CPU private-buffer reduce
                                       (spectral_synth_cpu.c:232) is a redundant
                                       O(threads×len) memory+bandwidth stage — replace with
                                       output tiling (engine already owns the pattern on
                                       GPU). The GPU 64B→32B repack
                                       (spectral_common.h:56) is a stage that VANISHES if
                                       Segment is natively packed (O2-B).
```

**Minimal stage set:** `window → fused-STFT → scan+estimate → match → track-form →
cubic-interp → density-adaptive-synth`. Everything in `[8]` collapses into `[6]`/`[9]` or
is deleted; one STFT implementation; no private-buffer reduce; no GPU repack stage.

---

# Part C — The cascade-ordered plan

Sequencing principle: **fix the algorithm, then tune it.** Foundation (F-) phases change
*what* is computed; micro-opt (O-) phases change *how fast* the now-minimal computation
runs. No O- item lands before the F- decision it depends on.

## Phase 0 — Verification gate (with ULTRAPLAN D0–D2)

Stand up the compiled harness; commit goldens from the **current** kernel. Extend
`arm32_process_correctness` fixtures with a **chirp** case (needed by F1/F3/O4-A). Blocks
all code changes. Risk: none (test-only).

## Phase 1 — FOUNDATION (must precede every micro-opt)

### F1 — Make tracks first-class (MQ peak continuation)  *(Tier 1; A3)*

Two-pass, preserving today's parallel peak-scan:
```text
- Pass 1 (parallel, unchanged): per frame-pair, SIMD scan + estimate + match. Store, per
  frame, the peak list and each peak's forward-match link (peak→peak in next frame). The
  matching already guarantees ≤1 successor/predecessor, so the links form simple chains.
- Pass 2 (cheap, O(total_peaks)): follow links to extract each chain = one track; emit ONE
  segment per track with length = k·hop, df = measured chirp slope, da = measured amp
  slope. (Parallelisable by list-ranking if it ever shows up in a profile; unlikely.)
- Implement as spectral_proc_adaptive_track_density_apply (the existing stub IS this).
```
Flag `SPECTRAL_TRACK_LINKAGE` (default **on** post-validation; off = legacy one-grain-per-
hop, kept as the pre-linkage golden reference). New golden for segment structure; existing
**audio** goldens must still pass within tolerance (linkage must not audibly change
reconstruction — Q3). Risk: medium, fully reversible via flag.

### F2 — Synthesis-method decision: density-adaptive oscillator/IFFT  *(Tier 1; A5)*

```text
- Benchmark oscillator-bank vs a new inverse-FFT (Rodet-Depalle FFT⁻¹) synth over a
  partial-density sweep on desktop and on the ARM sim perf-model (answers Q1/Q2).
- Implement the IFFT synth path for the DENSE regime (desktop/GPU first). Keep the
  oscillator bank for the SPARSE regime and for embedded (real-time, low-mem) — unless Q2
  shows ARM can afford one rfft/frame, in which case ARM gets the hybrid too.
- Wire the density switch as the engine's hybrid_render method (the dead token is the hook)
  with threshold P* from the benchmark.
```
Flag `SPECTRAL_SYNTH_METHOD` (auto|osc|ifft; default auto=hybrid). This is the decision
that makes the O- micro-opts well-scoped. Risk: medium-high (new synth path); each path
golden-verified independently; IFFT introduces its own per-field tolerance (window
main-lobe placement). **Behaviour change → signed-off golden.**

### F3 — Per-track cubic-phase interpolation  *(Tier 1; A4; depends on F1)*

Replace per-grain quadratic-phase + cross-fade with MQ cubic phase + linear amp per track
segment. Removes the cross-fade compute and its artifacts. Composes with O3-A (cubic phase
by 3-add forward differences). Flag rides `SPECTRAL_TRACK_LINKAGE`. Risk: medium; long-
track drift bounded by re-anchoring (Q4). Behaviour change → signed-off golden.

## Phase 2 — Micro-optimisations (each tagged with its contingency)

Tier order = the directive's priority hierarchy. "Regime" = which F2 synthesis path it
optimises.

```text
O1-B  CPU output tiling instead of private-buffer reduce       Tier1  regime: osc-bank
      Replace O(threads×len) private buffers + reduce pass (spectral_synth_cpu.c:188-254)
      with disjoint output-range tiles, one write/sample — port the engine's own GPU
      tiler. Independent of F1/F2 (improves the osc-bank path either way). ≤1 ULP FP
      reorder (CHANGELOG Pass 192). Flag: none (strict win). Risk: low-med.

O2-A  Default vectorised sine + SPECTRAL_PRECISE_TRIG flag      Tier2  regime: osc-bank
      Wire dispatch (osc_set_dispatch is NEVER called, oscillator.c:13) and make the 4-wide
      poly sine (oscillator_simd.c:46) the default; libm sinf behind a precise flag; keep
      SPECTRAL_GUARANTEE_EXACT_TRIG in sync. CONTINGENT on F2 keeping a meaningful osc-bank
      regime (it does — sparse/embedded). Behaviour change → signed-off golden; precise
      build must reproduce the old golden bit-for-bit. Resolves O5-A. Risk: medium.

O2-B  Native 32B Segment packing (finish SPECTRAL_COMPACT_SEG)  Tier2  regime: both
      8 used floats in a 64B aligned slot (spectral_common.h:19-29) → 50% of SegmentArray
      bandwidth is padding; SegmentArray is hard-typed Segment* (:89). Measure-gated; if
      synth is segment-stream bandwidth-bound, pack to 32B (the SegmentGpu shape) and the
      GPU repack stage vanishes. Bit-identical. Risk: low-med (alignment).

O3-A  Recurrence oscillator (+ cubic forward-diff for F3)       Tier3  regime: osc-bank
      Per-sample phase via incremental accumulator (2 adds const-freq; 3 adds chirp; 3 adds
      cubic for F3) instead of Horner re-eval (spectral_segment_math.h). Borrow: coupled-
      form / forward-difference oscillator (Gordon-Smith 1985; JOS). Re-anchor at segment
      boundaries to bound drift (Q4). Flag SPECTRAL_PRECISE_PHASE → closed form. CONTINGENT
      on osc-bank regime; composes with F3. Risk: medium.

O3-B  Hoist per-segment endpoint validation to emit-time        Tier3  REJECTED
      [impl 2026-06-01] NOT VIABLE AS SPECIFIED. The endpoint values validated in
      segment_loop_params_init (spectral_synth_internal.c:325-339) depend on alpha/beta/
      d_amp, which derive from SYNTH-TIME params (pitch_factor, inv_stretch, out_len) not
      known at analysis emit time; and loaded .spq segments never pass through emit. The
      check is inherently synth-time. Residual micro-opt (phase0/amp0 at j=0 provably equal
      s->phase/s->amp, so 2 of the 4 evals are redundant) is negligible — O(segments) not
      O(samples) — and would touch CTF-audited code for no measurable gain. Skipped.

O4-A  Consume the chirp slope on ARM                            Tier4  REJECTED-as-specified
      [impl 2026-06-01, PASS197] NOT VIABLE AS SPECIFIED. The proposed source df_q15 carries
      NO chirp for any realisable input: (1) segment_to_q15() — the runtime in-memory path the
      sim/emulator/oracle exercise — forces df_q15=0 unconditionally (spectral_synth_simulation
      .c:149), config-independent; (2) the compact 14B segment omits df_q15 entirely
      (spectral_q15.h:170-179); (3) the analysis slope df = bin_delta·π/(n_fft·hop)
      (spectral_peak_estimator.c:829, spectral_peak_track.c:706) is ≤~6e-6 for the ±1-bin links
      the tracker targets, and FLOAT_TO_Q15(df) truncates (plain Q15 LSB 3.05e-5) to 0 below
      bin_delta≈5.09 bins/hop @4096/128 — the persisted path's extra /1000
      (convert_segments.c:346) needs ~5090 bins, more than exist. So freq_inc += freq_delta is a
      guaranteed no-op; consuming it would only pessimise the CTF-audited M7 loop (kills the
      const-freq batch) and loosen the Pass-22-lineage validator for always-zero data. Also
      breaks the "F1 enabler" claim: F1's cubic coeffs live in the float Segment pad, not df_q15,
      and its quadratic df vanishes under Q15 too. REAL prerequisite = finer-scale chirp storage
      (reinterpret df_q15 @~2^24, or populate the dead q31 freq_delta = df·2^32/π) + drop the
      spurious /1000 + populate/consume freq_delta + drop the rejection — a .spq WIRE-FORMAT +
      behaviour change needing the "arm32 exact+chirp" golden (Part D), all maintainer-gated.
      Skipped; the q31 freq_delta SOA field stays allocated-but-dead until then.

O4-B  restrict / #pragma omp simd on host CPU loops             Tier4  regime: osc-bank
      Add restrict + simd hints to the sustain/reduce loops to unblock auto-vectorisation.
      Quick, low-risk. ≤1 ULP. Risk: low.

O4-C  Hand NEON                                                 Tier4  DEFERRED
      Only after O3-A/O4-A land and the harness identifies a specific under-vectorised ARM
      loop. No code until a measured target exists.
```

## Phase 3 — Dead-code removal (resolved by the foundation)

```text
O5-A  Host SIMD oscillator dispatch infra: kept & wired by O2-A, or deleted if O2-A is
      declined. No compiled-but-undispatched path survives.
O5-B  Processing-chain: adaptive_track_density → implemented by F1; hybrid_render →
      implemented by F2; serra_smith_1990 (A6) and johnston_1988 (A7) → DELETE the no-op
      stubs, record as named future work in Part A; the 5 stage-less parse tokens
      (reassigned, event_bucket, higher_order_interp, qnoise_shaping, …) → delete unless a
      maintainer wants them as a documented roadmap. Removing no-ops keeps goldens
      unchanged; update the parse-rejection test for dropped tokens.
O5-C  STFT path: investigate unifying on the fused path and retiring full_matrix
      (Part B [2]); only if full_matrix's random access is not required. Behaviour-neutral.
```

---

# Part D — Deliverables (contingency-tagged)

```text
ID    Tier  Depends-on        Flag                      Behaviour-change  Golden
F1    1     Phase0            SPECTRAL_TRACK_LINKAGE    yes (seg struct)  new seg + audio≈
F2    1     Phase0,(Q1,Q2)    SPECTRAL_SYNTH_METHOD     yes (ifft path)   new per-method
F3    1     F1                (rides TRACK_LINKAGE)      yes (cubic phase) new audio≈
O1-B  1     —                 none                      ≤1 ULP            audio≈
O2-A  2     F2(osc regime)    SPECTRAL_PRECISE_TRIG     yes (poly sine)   new + precise≡old
O2-B  2     —                 SPECTRAL_COMPACT_SEG      no                bit-identical
O3-A  3     F2(osc),F3        SPECTRAL_PRECISE_PHASE    yes (recurrence)  audio≈ + precise≡
O3-B  3     —                 none                      no                bit-identical
O4-A  4     Phase0,F1         SPECTRAL_HAS_CHIRP        yes (ARM chirp)   arm32 exact+chirp
O4-B  4     —                 none                      ≤1 ULP            audio≈
O4-C  4     O3-A,O4-A,measure none                      —                 deferred
O5-*  —     F1,F2,O2-A        —                         no (dead code)    unchanged
```

Suggested execution order (value × safety): **Phase0 → O3-B → O4-B → O1-B**
(bit-stable/independent quick wins) **→ F1 → F3 → O4-A → F2 → O2-A → O3-A → O2-B → O5**
(foundation, then its regime-scoped micro-opts, then sign-off-gated defaults, then cleanup).

---

# Part E — Patterns & conventions (for maintainers)

1. **Algorithm before micro-opt.** Settle the foundational algorithm (track formation,
   synthesis method) before tuning it. A micro-opt on a non-minimal algorithm is wasted —
   the algorithm choice cascades into every tier.
2. **Tracks are first-class; grains are a fallback.** Analysis output is one cubic-phase
   track per partial (MQ), not one grain per (frame,bin) peak. The one-grain path survives
   only behind `SPECTRAL_TRACK_LINKAGE=off` as the reference golden.
3. **Scope each synthesis method to its density regime.** Oscillator bank for sparse /
   embedded (low latency+memory); inverse-FFT for dense (partials-independent). The hybrid
   switch (threshold P*, measured) is the minimal way to be optimal at both ends — and it
   keeps the oscillator-bank micro-opts meaningful (they tune the regime where that family
   is correct).
4. **Golden-gated, speed-default + precise-flag.** Capture goldens from the current kernel;
   optimise against them; a deliberate behaviour change means a *new, signed-off* golden,
   never a moved tolerance. Every fast default ships a `SPECTRAL_PRECISE_*` escape hatch
   that reproduces the pre-optimisation golden bit-for-bit, with `SPECTRAL_GUARANTEE_*`
   bits kept in sync.
5. **Scatter→gather over private-buffer reduction.** Partition the *output*, write each
   sample once (the GPU tiler is the reference; the CPU path follows it).
6. **Recurrence over re-evaluation in inner loops.** Per-sample phase/amp via forward-
   difference accumulators (adds only; const-freq=2, chirp=3, cubic=3), re-anchored at
   segment/track boundaries to bound float drift.
7. **No compiled-but-undispatched paths; no no-op stages.** A SIMD/GPU/IFFT path is wired
   as a real default or deleted. Genuine future work (SMS residual A6, Johnston masking A7,
   XQIFFT A2, reassignment) is recorded as a *named paper here*, not as dead C.
8. **Measure, don't assert.** The crossover P*, ARM IFFT feasibility, and drift bounds are
   settled by benchmark (Part A-research Q1–Q4), not by assumption.

---

# Out of scope

```text
- Neural / differentiable synthesis (DDSP etc.) — different engine class.
- The deferred GPU fade-tail-under-time-stretch observation. (Its companion, the Daisy .spq
  re-validation, LANDED in review-3 / rdc-daisy-01: load_sd routes through the new
  spectral_arm32_load_in_place, which validates + SDRAM-fences before synthesis.)
- Non-ARM hand assembly (directive: ARM only for now).
- Building the SMS residual (A6) or Johnston masking (A7) now — recorded as named future
  work; F2 keeps the door open (residual is near-free under inverse-FFT synthesis).
- Any change to ULTRAPLAN's phase ordering or closure criteria; no new test philosophy.
```

---

## References (borrowed work)

```text
- R. McAulay & T. Quatieri, "Speech Analysis/Synthesis Based on a Sinusoidal
  Representation," IEEE ASSP-34(4), 1986.                         → A3 tracks, A4 cubic phase
- J. O. Smith & X. Serra, "PARSHL: …Non-Harmonic Sounds…," ICMC 1987. → A2/A3 peak model
- X. Serra & J. O. Smith, "Spectral Modeling Synthesis," CMJ 14(4), 1990. → A6 residual
- X. Rodet & P. Depalle, "Spectral Envelopes and Inverse FFT Synthesis," AES 1992;
  Freed, Rodet, Depalle, "Synthesis…by Inverse FFT," 1993.        → A5 inverse-FFT synth
- J. D. Johnston, "Transform Coding of Audio Signals Using Perceptual Noise Criteria,"
  IEEE JSAC 6(2), 1988.                                           → A7 masking prune
- F. J. Harris (1978); B. Quinn (1994); Jacobsen-Kootsookos (2007); Candan (2011);
  Rife-Boorstyn (1974); J.O. Smith QIFFT.                         → A2 (already in tree)
- Werner & Germain, "XQIFFT," 2016.                               → A2 optional refinement
- Gordon & Smith, recurrence/coupled-form oscillator, 1985; J.O. Smith DASP. → O3-A
```
