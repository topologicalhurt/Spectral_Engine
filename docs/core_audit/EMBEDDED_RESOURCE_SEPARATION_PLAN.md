# Embedded Resource-Separation Plan (Cortex-M7 / Q15 path)

Scope: **the embedded path only** (`arch/arm`, `core/spectral_osc_q31.h`, `core/spectral_q.h`,
`core/spectral_mem.h`, `api/daisy_seed`). This is the maintainer's **S1** stream
(`REVIEWER_HANDOFF.md` §3: "cache coherence, memory bandwidth, DMA, saturating ALUs, and using
the MAC / FPU / DSP units") narrowed to four concerns:

1. Float/double/non-Q mixing with a **clean FPU-vs-ALU ownership model** (know which unit is busy when).
2. **Cache coherency.**
3. **Memory bandwidth.**
4. Clean **ARM-vs-desktop/x86/OS architectural separation.**

The enduring principles distilled here are canonized as `AI_CANON.md` rules **21–25** (and a clause
added to rule 6). This doc holds the per-finding ledger and the measure-first work plan.

---

## 0. The honest reality on "saturate FPU and ALU simultaneously"

The Cortex-M7 is in-order and dual-issue, but the synthesis recurrence is **serial** (each coupled
step depends on the previous — no cross-sample ILP, `spectral_osc_q31.h`). You therefore **cannot**
hide a floating-point burst behind the integer loop *within one voice's stream*. Today the design
has a **clean temporal ownership** boundary — FPU only at the per-voice seed, pure Q31/Q15 ALU for
the per-sample loop — but **zero overlap**, and the seed is paid **per block** (an unaccounted FPU
burst at every block boundary; see FPU thread).

Two distinct goals follow, and they must not be conflated:

- **Clean separation + minimal FPU (do this first).** Make the FPU cost honest and small: seed
  once at *activation*, carry state across blocks, bound drift with the cheap **ALU** renorm that
  already exists. Result: the audio steady state touches no FP at all → the cleanest possible
  "which unit when" contract (and an FP-context-free audio ISR, relevant to lazy VFP stacking).
- **Actual simultaneity (a separate, measure-first investigation).** True FPU+ALU overlap needs
  overlapping *work items*, e.g. a dual-domain voice split — some voices rendered in **float on the
  FPU**, others in **Q15 on the ALU**, software-pipelined so both units have work each cycle. This
  is what the reserved `SPECTRAL_EMBEDDED_FLOAT` gate was for; it is **unbuilt and unproven**.
  Engage it only behind a benchmark on the S1 model (the recurrence-serial + dual-issue constraints
  may or may not let it win — measure before building).

---

## 1. Findings ledger

Disposition legend — **DOC** = doc-truth fix (comment/markdown, no codegen, m7 gate untouched);
**PERF-GATED** = changes arm32 hot-path codegen → renumbers GCC labels → regenerate
`tests/fixtures/m7_baseline.json`; **MODEL/TEST** = perf-model or test change, not codegen-gated;
**API** = outward-facing behavior change (maintainer decision); **ON-TARGET** = needs hardware/BSP
to verify; **REFUTED** = candidate concern checked and dismissed (do not re-litigate).

| id | concern | sev | disposition | one-line |
|----|---------|-----|-------------|----------|
| placement-doc (bw-01/cache-01/fpu-06/arch-04) | honesty | med | **DOC ✅ landed** | arm32.c header called DTCM placement "INERT / `.dtcm_data`"; the Daisy BSP binds `SPECTRAL_MEM_FAST`→`.dtcmram_bss`, so accum+ctx ARE in DTCM. Rewritten. |
| osc-doc (contracts-01/honesty-03/wcet-init-04) | honesty | med | **DOC ✅ landed** | `spectral_coupled_init` "activation-time only, rare" + "float minimax" were false: it is **double** and runs **per block per voice** on arm32. Corrected. |
| fpu-04 unit-ownership | fpu-alu | med | **DOC ✅ landed** | No stated FPU/ALU ownership model; added to arm32.c header + this doc §0. |
| q63-width (bw-03) | bandwidth | info | **DOC ✅ landed** | Pinned the q63 contract at the accum decl (128·2³⁰=2³⁷>int32; SMLALD needs 64-bit dest). |
| stretch-doc (honesty-01) | honesty | high | **DOC ✅ landed** | `spectral_arm32_set_stretch` is a silent no-op; commented as intentionally inert + synth-time-only. |
| renorm-note (fpu-05/kr-03) | fpu-alu | med | **DOC ✅ landed** | `spectral_coupled_renorm` is test-only; noted at its def + the standing fix. |
| **fpu-01** carry-(c,s)+renorm | fpu-alu | high | **⛔ DECLINED on data** | After fpu-03, also carry the `(c,s)` state across blocks + one ALU `spectral_coupled_renorm`/block, dropping the f64 state-seed. Implemented + verified (digest IDENTICAL `07f868f5…` for the workload, SINAD 80.7 unchanged, perf gate green), but the measured gain on the STONE workload was only **−1.2% process_insns** (2.10M→2.08M) while it **costs +4 KB DTCM** (`osc_c[512]+osc_s[512]`) and swaps the exact-by-construction re-seed for a drift-bounded regime (only empirically byte-identical for tested inputs). Reverted per the minimal/decline-on-data discipline — the per-block seed was NOT the remaining bottleneck (fpu-03 already removed it). **Caveat (recreatable):** the benefit scales with partial *duration*; STONE is transient-heavy, so a held-note/drone workload would show more. Revisit only with a sustained-content WCET scenario (fpu-02) showing a gain that justifies the DTCM. |
| **fpu-03** invariant-constants | fpu-alu | med | **✅ LANDED** | `cos_w/sin_w` depend only on (constant) ω; were recomputed every block. Now computed ONCE at activation, stored in the active record (AoS+SoA), threaded into `synth_segment_m7`/`synth_core_pair_m7`; only the `(c,s)` state is re-seeded per block. **Byte-identical** (m7 baseline digest unchanged `07f868f5…`; host `arm_core_test` output identical) and **−44% process_insns** (3.76M→2.10M) — measuring that the per-block f64 re-seed *dominated* the embedded synth. Steady-state cyc/voice-sample 24.01 unchanged. m7 baseline re-signed (label keys + insns only); perf gate + ctest 32/32 green. |
| **kr-02** WCET cap unenforced | wiring | **high** | **API / PERF-GATED** | `DAISY_MAX_ACTIVE=128` (WCET budget) is enforced **nowhere**; runtime cap is `SPECTRAL_ARM32_MAX_ACTIVE=512` (4× the proven budget). Enforce a runtime `max_active`, or delete the budget comment. |
| bandwidth-01 dead LUT on M7 | bandwidth | med | **PERF-GATED** | M7 path never reads the 8 KB sine LUT yet `process()` requires it non-null and Daisy holds it resident. Gate the precondition on the LUT path; reclaim ~8 KB SRAM. |
| kr-01 fade-partition dup | dup | med | **PERF-GATED** (or parity test) | Three-region fade boundary math is hand-duplicated across M7 / non-M7 / SIMD; already one clamp out of sync. Hoist to one `static inline` or pin with a parity test. |
| fpu-02 48-sample WCET | fpu-alu | high | **MODEL/TEST** | m7 baseline prices the 256-sample buffer cap; the codec block is `DAISY_AUDIO_BLOCK_SIZE=48`, so the per-block seed is under-weighted ~5.3×. Add a 48-sample WCET scenario. |
| arch-03 / kr-03 SNR gate | honesty | med | **MODEL/TEST ✅ landed** | The 72 dB SNR contract was validated only under the *renorm* regime; the shipped *re-seed* regime had no SNR/THD gate. `arm_core_test::test_single_tone` now measures coherent SINAD over the sustain region at the kernel's exact rendered frequency — **measured 80.7 dB, floor 70.0** (fail-on-bug: 2% 2nd-harmonic → 28 dB, gate trips). This is the safety net required before the Phase-A drift-regime change. |
| cache-02 dormant SCB untested | cache | med | **MODEL/TEST ✅ landed** | The cacheable `SCB_InvalidateDCache_by_Addr` arm is compiled by no build/test. The line-round + overflow arithmetic is now extracted to `spectral_cache_invalidate_range` (`spectral_mem.h`) and pinned by `arm_core_test::test_cache_invalidate_range` (align/round-up/overflow/INT32_MAX cases, fail-on-bug verified). The dormant DMA path calls it (behaviorally identical; the SCB call stays firmware-only). No measured-codegen change (DMA path is `SPECTRAL_HAS_DMA`-dormant; helper unused in measured builds). |
| arch-05 firmware purity | arch | low | **MODEL/TEST** | The layer law checks include direction, not OS-contract-freedom. Add a test over `SPECTRAL_SOURCES_DAISY_ENGINE` for denylisted symbols (`omp.h`, `mmap`, `sysconf`, `fopen`…). |
| arch-02 sim renders other osc | arch | med | **MODEL/TEST** | `simulate`/`embedded_arm` host targets don't set `SPECTRAL_ARM_M7`, so they render the **LUT** oscillator while firmware renders the **coupled** one. Force M7 on the sim, or state the divergence at its surface. |
| fpu-07 Q-domain markers | mixing | low | **DOC** (verify test) | The RT integer loops carry no `SPECTRAL_Q_DOMAIN` markers, so "no float in the recurrence" is convention-only. Wrap the per-sample bodies (seed stays outside). |
| arch-01 capability name | arch | med | **DOC** | `#if SPECTRAL_ARM_M7` selects a whole *algorithm* (recurrence vs LUT-gather), i.e. a capability spelled as a CPU. Name the proxied capability at the gate (don't add the macro speculatively). |
| cache-04 redundant init barrier | cache | low | **DOC/clean** | `dsb` at the end of `spectral_arm32_init` fences nothing (pure same-core CPU writes). Delete or correct the comment. |
| sd-load coherency | cache | med | **ON-TARGET** | `daisy_spectral_load_sd` `f_read`s into **cacheable** SDRAM then issues only `dsb` (not invalidate). If libDaisy SDMMC/FatFS DMAs into the destination, the CPU can read stale cache. Confirm the FatFS transfer path on hardware; if DMA, add an invalidate. |
| dma-double-buffer | bandwidth | — | **ON-TARGET** | `SPECTRAL_HAS_DMA` segment prefetch is dormant; activating coherent double-buffering needs a board. |
| itcm-code | bandwidth | — | **ON-TARGET** | `SPECTRAL_MEM_FAST_CODE` is a no-op (no libDaisy ITCM section); pinning hot code needs a linker-script addition. |

### Refuted candidates (checked — do not re-open)

- **bw-02** "prefetch is a no-op on M7": refuted — `__builtin_prefetch` lowers to a real `PLD` on
  Cortex-M7 (adversarially re-verified by cross-compile). The hot-loop/segment prefetches are live.
- **bw-03 (the optimization)** "narrow q63→q31 to halve accumulator bandwidth": refuted — 128 voices
  overflow int32, SMLALD mandates a 64-bit destination, and the accumulator is DTCM-resident
  (zero-wait, not SDRAM bandwidth). q63 is load-bearing.
- **cache-03** clean-before-DMA on `dma_seg_buf` / output-buffer-vs-codec-DMA / false sharing:
  refuted — `dma_seg_buf` is CPU-read-only (invalidate-after-RX is complete); the output buffer is
  caller-owned (codec-DMA coherency is the BSP's); single-core single-thread → no false sharing.
- **chirp-05**: chirp rejection (`df_q15`) is loud and correct, not a silent drop.
- **boundary-06**: `load` and `load_in_place` share one validator — the SD trust boundary is
  validated once.
- **wcet-cap-07**: the "128 voices @ 256 samples" *number* is measurement-backed (the gap is that it
  is **unenforced** — see kr-02 — not that it is asserted).

---

## 2. Work plan (measure-first; maintainer sequences the gated items)

**Landed (DOC, safe — no codegen change, m7 gate untouched):** the placement, oscillator,
unit-ownership, q63-width, stretch, and renorm doc-truth fixes above; `AI_CANON.md` rules 21–25 +
the rule-6 clause.

**Landed (TEST, arch-03 — the Phase-A safety net):** `arm_core_test::test_single_tone` now gates
coherent SINAD of the shipped re-seed regime over the sustain region (measured 80.7 dB, floor 70;
fail-on-bug verified). Build this BEFORE Phase A so the drift-regime change has an audio-quality
gate to prove against. ctest 32/32 green.

**Landed (TEST, cache-02 — coherency arithmetic):** `spectral_cache_invalidate_range`
(`spectral_mem.h`) extracts the DMA-RX line-round + overflow guards into a host-testable inline,
pinned by `arm_core_test::test_cache_invalidate_range` (fail-on-bug verified). The firmware-only
SCB call is unchanged; the dormant DMA path calls the helper. ctest 32/32; measured codegen
untouched.

> **Phase A is blocked in toolchain-limited environments.** Any arm32 codegen change renumbers GCC
> labels → the m7 perf gate's kernel keys mismatch → `m7_baseline.json` must be regenerated
> (`benchmark_workflow m7-baseline --generate`). That regen AND the live gate
> (`test_perf_gate_live_stack_within_contract`) both require **`llvm-mca`** (for the `[modeled]`
> cycle numbers) and a newlib arm cross-gcc + qemu. Where `llvm-mca` is absent (e.g. an
> Apple-clang-only box — `brew install llvm` provides it), do Phase A only after installing it, or
> have the maintainer regenerate the baseline. Do NOT land a hot-path change with a stale baseline.

**Phase A — FPU/ALU separation (the core concern). PERF-GATED; m7-baseline regen + `llvm-mca`.**
1. **✅ LANDED (fpu-03, commit 26f8314):** extend the active record (SoA+AoS) with `cos_w/sin_w`
   computed ONCE at activation; render reuses them. Byte-identical (digest `07f868f5…`), **−44%
   process_insns**. The codegen-confirm / `arm32_process_correctness` / production-regime SNR gate
   (arch-03) / baseline re-sign loop is proven and documented (label-key re-pin + wrapper sync).
2. **⛔ DECLINED on data (fpu-01):** also carry `(c,s)` + per-block ALU renorm to drop the f64
   state-seed. Implemented + verified green, but only **−1.2%** more on STONE for **+4 KB DTCM** and
   a drift-regime swap — reverted (the seed was not the remaining bottleneck). Revisit only if a
   sustained-content WCET scenario (fpu-02) shows it pays for the DTCM.
3. **Open:** price the existing path at the real **48-sample** codec block (fpu-02) — model/bench,
   not codegen-gated; this also gives fpu-01 a fair (sustained) re-measurement if revisited.
4. Later: evaluate the dual-domain (FPU-voices + ALU-voices) split for *actual* simultaneity behind
   `SPECTRAL_EMBEDDED_FLOAT` — **benchmark on the S1 model first**; decline-on-data is a valid outcome.

**Phase B — real-time safety + bandwidth. PERF-GATED / API.**
- Enforce a runtime `max_active` (default 512; Daisy passes `DAISY_MAX_ACTIVE=128`) at the
  activation boundary, or delete the unenforced budget comment (kr-02).
- Gate the `osc_lut` non-null precondition on the LUT path; stop allocating/filling the LUT on the
  M7 build (bandwidth-01).
- Hoist the fade-region partition or add a parity test (kr-01).

**Phase C — coverage + arch separation. MODEL/TEST (not codegen-gated).**
- Extract the DMA-RX line-round arithmetic to a host-unit-tested inline; label the SCB arm
  firmware-only (cache-02).
- Add the firmware-purity test over `SPECTRAL_SOURCES_DAISY_ENGINE` (arch-05).
- Force `SPECTRAL_ARM_M7` on the `simulate`/`embedded_arm` host targets so the sim renders the
  firmware oscillator (re-baseline any sim-pinned audio), or document the divergence (arch-02).
- Wrap the RT integer loops in `SPECTRAL_Q_DOMAIN` markers (fpu-07); add the capability-proxy
  comment (arch-01); resolve the init barrier (cache-04).

**Phase D — on-target (hardware-gated; the standing S1 frontier).**
- Confirm the SDMMC/FatFS transfer path; add an invalidate to the SD-load path if it DMAs into
  cacheable SDRAM (sd-load coherency).
- Activate coherent DMA double-buffering; add an ITCM linker section + pin hot code.
- Replace modeled cycle/cache/bandwidth numbers with measured ones.

---

## 3. Verification anchors

- `arm32_process_correctness` (`arm-core-test.cmake`, forces `SPECTRAL_ARM_M7=1` +
  `SPECTRAL_HAS_DUAL_MAC=1`) — ground truth for any ARM change.
- `osc_recursive` SNR/drift gate; extend to the production re-seed regime (arch-03).
- `arm-none-eabi-gcc -mcpu=cortex-m7` codegen inspection (prove the intended instructions emit).
- `m7_baseline.json` perf gate — **regenerate** after any Phase-A/B hot-path change (and only then).
- `ctest` green on `embedded_arm` **and** `embedded_arm_float` **and** desktop (a symbol unused in
  one config is live in another).
