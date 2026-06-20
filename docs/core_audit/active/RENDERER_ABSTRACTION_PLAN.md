# Renderer abstraction plan — one spectral frame model, many rendering strategies

Status: DESIGN (Stage 0). No code changes proposed here are landed. This is the approval
artifact for reframing "synthesis" as "rendering" and introducing a first-class renderer axis.

## 0. Thesis and scope

The original intention of this engine: **invent synthesizers in the spectral domain**, on the
premise that the spectral (STFT/Gabor frame) representation is a *universal substrate* — every
classical synthesis technique can be expressed as operations on a per-frame spectrum and rendered
back to audio. This plan makes that premise an architecture: a single **scene model** (the tracked
partials) consumed by interchangeable **renderers**, mixed by linear superposition.

**The duality, stated precisely (the thing the premise rests on).** The STFT is *exactly
invertible* whenever the analysis/synthesis window satisfies constant-overlap-add (COLA): inverse-FFT
each frame, overlap-add, and the input is reconstructed perfectly [7, 31, 5]. So any signal — hence
any synthesizer's output — is *representable* as a sequence of inverse-FFT frames. That is the trivial
half of universality. The non-trivial half is the distinction this plan is built on:

- **Representable** — the output *can* be written as per-frame spectra (always true, by invertibility).
- **Naturally expressible** — the technique's *control parameters* map to **cheap per-frame edits of
  the complex spectrum**, rather than requiring a per-sample recurrence or feedback loop.

The formal reason "a frequency-domain renderer and a time-domain oscillator bank are duals" is not a
metaphor: the STFT has two dual readings [32, 31] — the **overlap-add (OLA)** view (a time sequence of
per-frame inverse FFTs) and the **filter-bank-summation (FBS)** view (a bank of heterodyned bandpass
filters, i.e. an oscillator/filter bank). The two execution domains in this engine *are* these two
readings of one transform.

**The shared failure mode of every per-frame method:** intra-frame stationarity. A per-frame spectral
deposit freezes each component's frequency and amplitude at frame center; transients, fast chirps, and
a time-varying FM index all violate this and smear [28]. This is why fast `df`/`da` partials are routed
to the time-domain backend, not the IFFT.

**Scope of this plan (locked with the maintainer, 2026-06-20):**
- Supported renderers NOW: **additive, wavetable, subtractive**.
- Future renderers (catalogued, §8, not built here): FM/PM, granular, modal, waveguide, stochastic/noise.
- "Rendering" is the **strategy axis** (what sound), NOT a blanket `synth → render` rename of the
  embedded kernel. The m7-pinned `spectral_arm32_*` symbols keep their names (renaming them renumbers
  GCC labels and trips the byte-pinned `m7_baseline.json`).

## 1. The three orthogonal axes (the core model)

The recon found that today's `SynthBackend {AUTO, CPU, METAL, CUDA, EXPORT}` enum conflates concerns
that are actually independent. The clean model separates **three** axes:

| Axis | Question | Members | Chosen by |
|------|----------|---------|-----------|
| **Renderer** | *what sound* | additive · wavetable · subtractive · (future: FM, granular, modal, waveguide, stochastic) | the user — a creative choice |
| **Domain** | *how it is executed* | time-domain (oscillator/table read per sample) · frequency-domain (deposit + inverse FFT) | the engine — by density/stationarity (the existing hybrid crossover ≈ 7 partials) |
| **Device** | *on what silicon* | CPU · Metal · CUDA | availability |

The key consequences:
- **The oscillator bank and the IFFT path are not renderers — they are the two execution *domains*.**
  "Invent a synthesizer" = author a **renderer**; the engine then renders it in whichever domain/device
  is cheapest and admissible. This is the precise answer to "what is the oscillator bank conducive to":
  it is the *time-domain* backend, the home of every partial the IFFT cannot express per-frame
  (non-stationary `df`/`da`/cubic, and the future feedback/nonlinear renderers).
- **Renderers superpose.** Additive synthesis is linear, so a scene can be partitioned across renderers
  and domains and **summed** into one output buffer. This is exactly the deterministic-plus-stochastic
  model [9] generalized from two renderers to N — the architectural payoff.

## 2. The current code, named honestly

- **Scene model** = `SegmentArray` of `Segment {start, length, phase, omega, df, da, amp, da, width,
  cubic c2/c3}` (`spectral_common.h`). The universal intermediate every renderer consumes. `width` is a
  per-partial timbre-shape parameter (e.g. PWM duty); `df`/`da` carry non-stationarity; cubic c2/c3 the
  MQ cubic-phase annotation.
- **Existing renderers, today (informal):**
  - Additive (sine): `synth_cpu` time-domain (coupled-oscillator NCO) and the Rodet–Depalle IFFT path
    (`spectral_synth_ifft.c`, frequency-domain) — already the two domains of one renderer.
  - Wavetable: `synth_cpu_wavetable` + `SpectralWavetableBank` (time-domain only today).
  - Subtractive *source*: the bandlimited rich timbres (`timbre_synth_segment` in
    `spectral_osc_bandlimited.c`: saw/square/triangle/PWM via the `TIMBRE_*` enum + `width`).
    **No explicit filter stage exists yet** — `spectral_envelope.h` is the temporal amp/fade envelope,
    not a spectral filter.
- **Routing:** `spectral_synth_hybrid_try_render` (the density/stationarity crossover) is the embryonic
  *domain* router. `spectral_synth_dispatch_ex` + the `SpectralBackendVTable` is the *device* router —
  but it is keyed by the conflated enum.

## 3. The renderer abstraction (target design)

```
scene (SegmentArray)
      │
      ▼
 renderer router ── partitions partials by which renderer + which domain each needs
      │
      ├── additive    ─┐
      ├── wavetable    ├─ executed in time-domain (osc) OR frequency-domain (IFFT), on CPU/Metal/CUDA
      └── subtractive ─┘
      │
      ▼
 Σ overlap-add → output     (linear superposition; renderers mix)
```

**The interface.** A renderer supplies the *creative recipe for one partial* in up to two dual forms —
a frequency-domain **deposit** and/or a time-domain **eval** — and declares which `Segment` fields it
honors. The backend (domain × device) supplies the *loop* that applies the recipe.

```c
typedef enum { SPECTRAL_RENDERER_ADDITIVE, SPECTRAL_RENDERER_WAVETABLE,
               SPECTRAL_RENDERER_SUBTRACTIVE /* … future ids … */ } SpectralRendererId;

typedef struct {
    unsigned honors_chirp     : 1;  /* uses Segment.df  (non-stationary frequency) */
    unsigned honors_amp_ramp  : 1;  /* uses Segment.da  (non-stationary amplitude) */
    unsigned needs_filter     : 1;  /* consumes a spectral transfer function H(f)  */
    unsigned spectral_native  : 1;  /* a per-frame deposit exists (freq-domain admissible) */
} SpectralRendererCaps;

typedef struct SpectralRenderer {
    SpectralRendererId    id;
    const char*           name;
    SpectralRendererCaps  caps;
    /* Frequency-domain recipe: write this partial's contribution into a frame's half-spectrum.
       NULL ⇒ renderer is time-domain-only. */
    void  (*deposit)(const Segment* seg, SpectralFrameSpectrum* frame, const SpectralRenderCtx* ctx);
    /* Time-domain recipe: evaluate this partial over a sample block.
       NULL ⇒ renderer is frequency-domain-only. */
    void  (*eval_block)(const Segment* seg, float* out, size_t n, const SpectralRenderCtx* ctx);
    /* Routing predicate: may this renderer take this segment under the current context? */
    int   (*eligible)(const Segment* seg, const SpectralRenderCtx* ctx);
} SpectralRenderer;
```

The existing `SpectralBackendVTable` is retained but re-scoped to the **device** axis (CPU/Metal/CUDA
implementations of a domain), no longer overloaded to mean "renderer."

## 4. The three supported renderers (recipes, code mapping, citations)

Each recipe gives: the per-frame spectral deposit, the time-domain eval, the control→parameter map,
and the `Segment` fields used. All three are **spectral-native** (a per-frame deposit exists), so all
three can run in either domain — the reason they are the clean "supported now" set.

### 4.1 Additive
- **Deposit (freq-domain):** for partial `j`, add one window-transform main lobe centred at the
  fractional bin `f_j·N/SR`, scaled by the complex amplitude `a_j·e^{iφ_j}`: `A_j[k] += a_j·e^{iφ_j}·
  W[f_j/SR − k/N]`. Only ~K≈9 significant lobe samples are deposited per partial [28].
- **Eval (time-domain):** integer-NCO / coupled-oscillator recurrence per sample (the current
  `synth_cpu`), which additionally honors `df` (chirp) and `da` (amp ramp) exactly.
- **Control → parameter:** `omega → bin / phase-increment`; `amp → |deposit| / gain`;
  `phase → arg`; `df, da → non-stationary terms (time-domain only)`.
- **Segment fields:** `omega, phase, amp, df, da, cubic`.
- **Code today:** `synth_cpu` (`TIMBRE_SINE`), `spectral_synth_ifft.c`, routed by the hybrid crossover.
- **Refs:** McAulay–Quatieri [8]; Rodet–Depalle [28]; Serra–Smith [9]; J.O. Smith SASP additive [7].

### 4.2 Wavetable
- **Model:** a wavetable is a stored single-cycle waveform — equivalently a fixed **harmonic-amplitude
  vector** `{c_k}` (its Fourier series). Rendering a partial = reproducing that harmonic series at the
  partial's fundamental.
- **Deposit (freq-domain):** place one lobe per harmonic — at bins `k·f_0·N/SR`, amplitude
  `a·c_k·e^{iφ_k}`, for all `k` with `k·f_0 < SR/2` (Nyquist truncation = automatic band-limiting).
- **Eval (time-domain):** phase-accumulator table lookup (the current `synth_cpu_wavetable`).
- **Wavetable morphing:** interpolate the harmonic vectors `{c_k}` of two tables by a position control
  (linear in `c_k` = a spectral cross-fade). **Per-frequency-band timbre:** scale `c_k` by a band
  envelope before deposit.
- **Control → parameter:** `position → table / interpolation weight`; `omega → f_0`; the table → `{c_k}`.
- **Segment fields:** `omega, phase, amp` + the bank/position (an extension field).
- **Code today:** `synth_cpu_wavetable`, `SpectralWavetableBank`; `BACKEND_CPU_WAVETABLE_SUPPORT`.
  Freq-domain deposit form is the natural near-term addition (none today).
- **Refs:** Bristow-Johnson, "Wavetable Synthesis 101" [33]; Roads, *Computer Music Tutorial* [34]; SASP [7].

### 4.3 Subtractive
- **Model:** source–filter. A harmonically-**rich source** (saw `{1/k}`, square `{odd 1/k}`, pulse/PWM,
  triangle) shaped by a **filter** `H(f)`. The convolution theorem makes the filter a *per-bin multiply*
  in the spectral domain — subtractive's defining operation is *free* in the frequency domain [2].
- **Deposit (freq-domain):** deposit the source's harmonic template (as in §4.2), then multiply each
  deposited harmonic by the filter response: `A[k] *= H(k·f_0)`.
- **Eval (time-domain):** the bandlimited source oscillator (exists: `timbre_synth_segment`) followed by
  a time-domain filter (the time-domain dual of `H`).
- **Control → parameter:** `source select → template {c_k}`; `width → pulse duty`;
  `cutoff / Q / slope → H(f) envelope`.
- **Segment fields:** `omega, phase, amp, width` (source) + a filter descriptor (the addition).
- **Code today:** the **source half exists** (bandlimited `TIMBRE_SAW/SQUARE/TRIANGLE/PWM` + `width`).
  The **filter half** (`H(f)` multiply / its time-domain dual) is the explicit piece this renderer adds.
- **Refs:** Fant, source–filter / acoustic theory [35]; Oppenheim–Schafer, convolution theorem [2]; SASP [7].

## 5. Routing

Two independent decisions, made in order:
1. **Renderer** — chosen by the user / the segment's timbre tag (creative). Additive/wavetable/subtractive
   are not efficiency choices; they are *what the sound is*.
2. **Domain** — chosen by the engine via the existing stationarity/density crossover: dense + stationary
   (`df≈0`, `da≈0`, long) → frequency-domain (IFFT); sparse or non-stationary → time-domain (osc). This
   is `spectral_synth_hybrid` generalized to "pick the cheapest admissible domain for this renderer."
3. **Device** — CPU/Metal/CUDA by availability (the existing vtable, re-scoped).

Because renderers superpose (linear), one scene may use several renderers and both domains at once,
summed into one buffer (the §1 payoff).

## 6. Staged refactor (each stage independently green; m7 + parity gates are the guard)

- **Stage 0 — this document + canon.** Add the renderer × domain × device model to `AI_CANON` and a
  `CORE_CONTRACTS` row. Zero code. *(The approval artifact.)*
- **Stage 1 — introduce the renderer layer as a wrapper (low risk).** Define `SpectralRenderer` +
  `SpectralRendererId` and wrap the *existing* `synth_cpu` / wavetable / IFFT / hybrid behind it, with
  **no symbol renames and no change to the arm32 inner loop**. The hybrid router becomes the formal
  domain router; the `SpectralBackendVTable` is re-scoped to the device axis. Default render path stays
  byte-identical ⇒ `full_fused_parity` / `gpu_backend_parity` / `osc_parity` / m7 gate green by
  construction. New ctest: a renderer-dispatch table test (each id resolves to the right recipe).
- **Stage 2 — terminology sweep `synth → render`, gated, host-only.** Rename at the host/CLI/dispatch
  boundary only; the m7-pinned `spectral_arm32_*` kernel names are excluded. Each batch:
  build + `tests_all` + ctest + m7 gate before commit.
- **Stage 3 — formalize wavetable + subtractive as first-class renderers.**
  - 3a: wavetable gains a frequency-domain **deposit** path (harmonic-vector placement) so it runs on
    the IFFT domain, not only time-domain; parity test vs the time-domain table read.
  - 3b: subtractive gains the explicit **filter** stage — `H(f)` per-bin multiply (freq-domain) and its
    time-domain dual — with a source×filter parity test. This is the one genuinely new DSP in scope.

## 7. Risks and gates

- **m7 perf gate** (`tests/tools/test_perf_gate.py`, byte-pinned `m7_baseline.json`): any out-of-line
  change to `spectral_synth_arm32.c` renumbers GCC labels. Mitigation: the renderer layer is host-side
  dispatch; the arm32 inner loop and its symbols are untouched. If a label-bearing symbol must move,
  regenerate the baseline (documented procedure, as in passes 269/the rdc-daisy-01 work).
- **Parity:** `full_fused_parity`, `gpu_backend_parity`, `osc_parity` must stay green; Stage 1 keeps the
  default path byte-identical, Stage 3 adds *new* parity tests for the new domain/filter paths rather
  than perturbing existing ones.
- **Scene-model extensions** (wavetable position, filter descriptor): prefer the spare `Segment` pad
  words (`_pad_w`, already used for cubic c2/c3) over growing the 64-byte struct, to keep the m7 layout
  and the GPU/Q15 SoA packing unchanged.

## 8. Future renderers (catalogued, not in scope)

| Renderer | Deposit / eval | Class | Refs |
|----------|----------------|-------|------|
| FM/PM (static index) | Bessel-weighted partials `J_k(β)` at `c ± k·m` | spectral-native | Chowning [36]; SASP FM [7] |
| FM/PM (dynamic index) | sidebands move per-sample with `β(t)` | time-domain-native | Chowning [36] |
| Granular | one Gabor atom per grain (windowed-sinusoid transform) | spectral-native (grain OLA ≈ IFFT-OLA) | Roads, *Microsound* [34]; Gabor |
| Physical — modal | sum of decaying resonant modes (additive + decay) | spectral-native | Bilbao [37]; SASP/PASP [7] |
| Physical — waveguide / nonlinear | bidirectional delay line + nonlinear junction | time-domain-native | J.O. Smith, digital waveguides [38]; Bilbao [37] |
| Stochastic / noise (SMS residual) | per-frame magnitude envelope, randomized phase | spectral-native (the IFFT's cheapest job) | Serra–Smith [9] |

The stochastic renderer is the largest single capability gain (it renders noise/breath/transients the
pure partial model structurally cannot) and is the natural first future addition once §6 lands.

## 9. References

Shared with `reference/ACADEMIC_SOURCES.md` (numbers preserved where they overlap):

- [2] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed., Pearson, 2010.
  (convolution theorem: filtering = spectral multiplication)
- [5] D. W. Griffin and J. S. Lim, "Signal Estimation from Modified Short-Time Fourier Transform,"
  *IEEE TASSP*, 1984. (STFT consistency / reconstruction)
- [7] J. O. Smith, *Spectral Audio Signal Processing*, W3K Publishing, 2011. (COLA, OLA↔FBS dual views,
  inverse-FFT synthesis, additive synthesis chapters)
- [8] R. J. McAulay and T. F. Quatieri, "Speech Analysis/Synthesis Based on a Sinusoidal Representation,"
  *IEEE TASSP*, 1986. (the sinusoidal/partial model)
- [9] X. Serra and J. O. Smith, "Spectral Modeling Synthesis: … Deterministic plus Stochastic
  Decomposition," *Computer Music Journal*, 1990. (the linear-superposition / residual basis)
- [28] X. Rodet and P. Depalle, "Spectral Envelopes and Inverse FFT Synthesis," *Proc. AES 93rd Conv.*,
  1992. (the FFT⁻¹ deposit recipe, K≈9 lobe samples, ~order-of-magnitude gain, stationarity caveat)
- [29] A. Freed, X. Rodet, P. Depalle, "Synthesis and Control of Hundreds of Sinusoidal Partials …,"
  *Proc. ICSPAT*, 1992.
- [30] L. Savioja, V. Välimäki, J. O. Smith, "Real-Time Additive Synthesis with One Million Sinusoids
  Using a GPU," *Proc. AES 128th Conv.*, 2010 / *JAES*, 2011. (additive is data-parallel — the device axis)

New (to be appended to `ACADEMIC_SOURCES.md` under "Synthesis methods and rendering"):

- [31] J. B. Allen and L. R. Rabiner, "A Unified Approach to Short-Time Fourier Analysis and Synthesis,"
  *Proc. IEEE*, 1977. (OLA and filter-bank-summation as dual STFT interpretations — the formal duality)
- [32] J. O. Smith, "Dual Views of the Short-Time Fourier Transform," in SASP [7].
- [33] R. Bristow-Johnson, "Wavetable Synthesis 101, A Fundamental Perspective," *Proc. AES 101st Conv.*,
  1996. (a wavetable as a stored harmonic spectrum; band-limiting)
- [34] C. Roads, *The Computer Music Tutorial*, MIT Press, 1996; and *Microsound*, MIT Press, 2001.
  (wavetable; granular synthesis as time–frequency atoms)
- [35] G. Fant, *Acoustic Theory of Speech Production*, Mouton, 1960. (the source–filter model)
- [36] J. M. Chowning, "The Synthesis of Complex Audio Spectra by Means of Frequency Modulation,"
  *JAES*, 1973. (FM Bessel-sideband spectra; time-varying index breaks stationarity)
- [37] S. Bilbao, *Numerical Sound Synthesis*, Wiley, 2009. (modal vs finite-difference/waveguide physical models)
- [38] J. O. Smith, *Physical Audio Signal Processing*, W3K Publishing, 2010. (digital waveguides)
