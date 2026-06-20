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
| placement-doc (bw-01/cache-01/fpu-06/arch-04) | honesty | med | **DOC ✅ landed** | arm32.c header called DTCM placement "INERT / `.dtcm_data`"; the Daisy BSP binds `SPECTRAL_MEM_FAST`→`.dtcmram_bss`, so the tagged `accum[]` IS in DTCM. Rewritten. (The re-audit later caught that this pass **over-claimed** "accum *and ctx*" — only `accum[]` carries the attribute; see **dtcm-ctx**.) |
| osc-doc (contracts-01/honesty-03/wcet-init-04) | honesty | med | **DOC ✅ landed** | `spectral_coupled_init` "activation-time only, rare" + "float minimax" were false: it is **double** and runs **per block per voice** on arm32. Corrected. |
| fpu-04 unit-ownership | fpu-alu | med | **DOC ✅ landed** | No stated FPU/ALU ownership model; added to arm32.c header + this doc §0. |
| q63-width (bw-03) | bandwidth | info | **DOC ✅ landed** | Pinned the q63 contract at the accum decl (128·2³⁰=2³⁷>int32; SMLALD needs 64-bit dest). |
| stretch-doc (honesty-01) | honesty | high | **DOC ✅ landed** | `spectral_arm32_set_stretch` is a silent no-op; commented as intentionally inert + synth-time-only. |
| renorm-note (fpu-05/kr-03) | fpu-alu | med | **DOC ✅ landed** | `spectral_coupled_renorm` is test-only; noted at its def + the standing fix. |
| **fpu-01** carry-(c,s)+renorm | fpu-alu | high | **✅ LANDED (w/ fpu-02)** | Carry `(c,s)` across blocks + one ALU `spectral_coupled_renorm`/block, dropping the per-block f64 state-seed entirely. The shipped drift regime is now carry+renorm (NOT byte-identical to re-seed — a validated audio change: `osc_recursive` 86–93 dB SNR over 8s, `arm_core_test` SINAD 80.7; new golden render checksum `0ed6e0ac`). **Measured directly at the real 48-sample block (fpu-02 fixture): −5.96%** process_insns (2,373,319→2,231,920); was only −1.2% at the old 256-block fixture (the artifact that first declined it). Cost +4 KB in the active context (`osc_c/osc_s[512]`), which lands in `.bss`/SRAM — **NOT DTCM** (the context is untagged; see **dtcm-ctx**). REQUIRED for the 512-voice real-time target. Steady-state cyc/voice-sample 24.01 unchanged. |
| **fpu-03** invariant-constants | fpu-alu | med | **✅ LANDED** | `cos_w/sin_w` depend only on (constant) ω; were recomputed every block. Now computed ONCE at activation, stored in the active record (AoS+SoA), threaded into `synth_segment_m7`/`synth_core_pair_m7`; only the `(c,s)` state is re-seeded per block. **Byte-identical** (m7 baseline digest unchanged `07f868f5…`; host `arm_core_test` output identical) and **−44% process_insns** (3.76M→2.10M) — measuring that the per-block f64 re-seed *dominated* the embedded synth. Steady-state cyc/voice-sample 24.01 unchanged. m7 baseline re-signed (label keys + insns only); perf gate + ctest 32/32 green. |
| **kr-02** WCET cap doc-stale | wiring | med | **REFRAMED by the 512-voice target** | Originally: `DAISY_MAX_ACTIVE=128` (old WCET cap) is enforced nowhere while the runtime cap is 512 → recommended enforcing 128. **The 512-voice target reverses this**: 512 is the goal, so the 512 cap is correct and must NOT be lowered. The defect is now doc-only: `DAISY_MAX_ACTIVE=128` and its "WCET-gated, 128 voices proven" comment are **stale** (the budget analysis above shows 512 is ~1.0–1.5× over on the oscillator bank → the real path to 512 is IFFT). Fix: update/retire `DAISY_MAX_ACTIVE` to state the 512 target + that 512-voice real-time rides the IFFT path, not assert a 128 guarantee. |
| bandwidth-01 dead LUT on M7 | bandwidth | med | **PERF-GATED** | M7 path never reads the 8 KB sine LUT yet `process()` requires it non-null and Daisy holds it resident. Gate the precondition on the LUT path; reclaim ~8 KB SRAM. |
| kr-01 fade-partition dup | dup | med | **PERF-GATED** (or parity test) | Three-region fade boundary math is hand-duplicated across M7 / non-M7 / SIMD; already one clamp out of sync. Hoist to one `static inline` or pin with a parity test. |
| fpu-02 48-sample fixture | fpu-alu | high | **✅ LANDED** | The counts fixture (`stagger9-8k`) rendered at **block=256**, but the real Daisy callback is **`DAISY_AUDIO_BLOCK_SIZE=48`** — under-pricing per-block overhead by **+12.9%** (2.10M→2.37M, measured). The canonical fixture now renders at block=48 (`stagger9-8k-b48`, total 16,416 = 48·342), so the perf model measures the real target and fpu-01 is gated at its true −5.96%. New input fixture digest + golden render checksum (re-signed baseline). |

### 512-voice real-time target (maintainer constraint: ~speech intelligibility)

The per-voice oscillator bank does **not** reach 512 real-time voices at 48 kHz on the M7 — the
**per-sample loop** (24.01 cyc/voice-sample, unchanged by fpu-01/fpu-03) is the wall, not the
per-block overhead. 512 voices need `512·48·cyc_vs` cycles per 48-sample block:

| clock | all-paired (20.0 cyc/vs) | budget | verdict | max voices (paired) |
|------:|--------------------------:|-------:|:-------:|:-------------------:|
| 400 MHz | 491,766 cyc/block | 400,000 | 1.23× over | ~416 |
| 480 MHz | 491,766 cyc/block | 480,000 | 1.02× over | ~499 |

fpu-01 + dual-MAC are necessary but **insufficient** for 512. 512 voices = a *dense* spectrum, which
is exactly the regime where **inverse-FFT (Rodet-Depalle) synthesis (~8×)** wins over the per-partial
oscillator (the F2 algorithm fork / IFFT_SYNTHESIS_PLAN). **Conclusion: the route to 512-voice
real-time is the IFFT synthesis path for dense frames, not further oscillator micro-opt.** The
512-slot active-record sizing (`SPECTRAL_ARM32_MAX_ACTIVE=512`) is therefore CORRECT — and kr-02's
"enforce 128" is OFF the table (128 was the old WCET cap; the target is 512).

### Block-size measurement (empirical, fpu-03 tree, qemu counts)

`process_insns(block) = 27.65·samples + 217.2·renders`, where `renders = 9·⌈8192/block⌉` (fit
validated: block=128 predicted to 0.02%). The synth is **loop-bound** (27.65 insns/sample); the
per-render overhead (seed + fade-setup + prologue) is 217 insns, of which fpu-01 removes ~89 (the
f64 state-seed net of renorm, from the direct fpu-01@256 delta 25,569/288).

| block | process_insns | renders | fpu-01 saving |
|------:|--------------:|--------:|--------------:|
| 256   | 2,101,357     | 288     | −1.2%         |
| 128   | 2,164,294     | 576     | ~−2.3%        |
| 64    | 2,289,042     | 1152    | ~−4.4%        |
| **48 (real Daisy)** | **2,373,319** | **1539** | **−5.76%** |
| arch-03 / kr-03 SNR gate | honesty | med | **MODEL/TEST ✅ landed** | The 72 dB SNR contract was validated only under the *renorm* regime; the shipped *re-seed* regime had no SNR/THD gate. `arm_core_test::test_single_tone` now measures coherent SINAD over the sustain region at the kernel's exact rendered frequency — **measured 80.7 dB, floor 70.0** (fail-on-bug: 2% 2nd-harmonic → 28 dB, gate trips). This is the safety net required before the Phase-A drift-regime change. |
| cache-02 dormant SCB untested | cache | med | **MODEL/TEST ✅ landed** | The cacheable `SCB_InvalidateDCache_by_Addr` arm is compiled by no build/test. The line-round + overflow arithmetic is now extracted to `spectral_cache_invalidate_range` (`spectral_mem.h`) and pinned by `arm_core_test::test_cache_invalidate_range` (align/round-up/overflow/INT32_MAX cases, fail-on-bug verified). The dormant DMA path calls it (behaviorally identical; the SCB call stays firmware-only). No measured-codegen change (DMA path is `SPECTRAL_HAS_DMA`-dormant; helper unused in measured builds). |
| arch-05 firmware purity | arch | low→**med** | **✅ LANDED (MODEL/TEST)** | The layer law checks include *direction*, not OS-contract-freedom (kernel→kernel is legal; system includes ignored). A re-audit found `spectral_wavetable.c` (a `SPECTRAL_SOURCES_DAISY_ENGINE` TU) compiled 4 host file loaders (`load`/`save`/`load_raw`/`load_hex`) UNGUARDED into firmware: `nm -u` (real-firmware mode) named `FILE*`/`sscanf`, the `spectral_fs_*` shim (NOT in the firmware link set → would be unresolved externals), and `malloc/calloc/free` — all inside those 4 fns (the bank is fixed arrays; `load_buffer` is the firmware path). Fixed: gated the `spectral_fs.h` include + the 4 loaders under the emulator guard `#if !SPECTRAL_EMBEDDED \|\| SPECTRAL_IS_EMBEDDED_SIM` (kept on desktop + host-sims, excluded ONLY on real firmware, which defines neither SIMULATION nor USE_EMBEDDED_SYNTH per daisy-config.cmake). NEW `tests/tools/test_firmware_purity.py` recompiles every engine TU in real-firmware mode and asserts `nm -u` names no host file I/O / heap / FS-shim / OS-threading symbol (fail-on-bug verified: neutering the guard trips it with the exact 10-symbol set). All 4 firmware TUs now CLEAN; all host targets link. |
| arch-02 sim renders other osc | arch | med | **✅ LANDED** | `simulate` rendered the **LUT** oscillator while firmware renders the **coupled** one. `simulation.cmake` now forces `SPECTRAL_ARM_M7=1 SPECTRAL_HAS_DUAL_MAC=1` on the `simulate` target (as arm_core_test does), so a host audio roundtrip (`spectral_*_simulation input.wav` → `output/<input>.wav`) renders the device's coupled path — bit-equivalent to QEMU/M7 (AoS-vs-SoA layout only). Verified: M7 render md5 differs from the prior LUT render; simulate_smoke green. (`embedded_arm`/`_restricted` left as-is — perf model, oscillator-agnostic.) |
| qemu-spq-roundtrip | tooling | — | **TRACKED (deferred)** | Extend the QEMU audio tool to load an arbitrary file's `.spq` segments via semihosting (mirroring `daisy_spectral_load_sd`) and render them on the LITERAL cross-compiled M7 ELF — a codegen-exact roundtrip, vs the simulate-M7 path which is code-exact (host build). Audio-equivalent to simulate-M7 (deterministic fixed-point), so the value is codegen fidelity, not a different sound. Feasible: mps2-an500 BULK = 15 MB > a real file's segment pool (~8 MB @ 7 s speech). Needs semihosting read (SYS_OPEN/READ) + a `.spq`-loading harness + the analyze→convert→render orchestration. Deferred per maintainer. |
| fpu-07 Q-domain markers | mixing | low | **✅ LANDED (DOC/TEST)** | The RT integer loops carried no `SPECTRAL_Q_DOMAIN` markers, so "no float in the recurrence" was convention-only. Wrapped the three per-sample bodies (synth_core_m7 / synth_fade_m7 / synth_core_pair_m7); `q_domain_contract` now ENFORCES it (9 pure regions). Seed stays outside. |
| arch-01 capability name | arch | med | **✅ LANDED (DOC)** | `#if SPECTRAL_ARM_M7` selects a whole *algorithm* (recurrence vs LUT-gather), i.e. a capability spelled as a CPU. Named the proxied capability at the render-dispatch gate (no speculative macro). |
| cache-04 redundant init barrier | cache | low | **✅ LANDED (DOC/clean)** | The `dsb` at the end of `spectral_arm32_init` fenced nothing (same-core CPU writes to ctx). Deleted; replaced with a comment on why no barrier is needed (the load/load_in_place barriers are the load-bearing ones). |
| dtcm-ctx placement | cache/bw | med | **DOC ✅ landed + ON-TARGET tracked** | Re-audit (pass-275 workflow): the arm32.c header + daisy_seed_mem.h claimed the active-voice context lands in DTCM, but only `accum[]` carries `SPECTRAL_MEM_FAST`. `DaisySpectralCtx ctx_` (incl. the fpu-01 `osc_c/osc_s` carries — read+written every block per voice) is an untagged C++ member → default `.bss`/SRAM. **Docs corrected (LANDED).** The actual win — pin the context (or just the hot SoA carries `osc_c/osc_s/cos_w/sin_w`) to DTCM (zero-wait) — is **ON-TARGET**: needs a BSP section attribute on the instance (or a SoA-static split), a DTCM-budget check (ctx ~24 KB of 128 KB DTCM), and hardware to validate the gain. The single best cache/bandwidth lever for the per-block-per-voice hot state. |
| stale-header-reseed | honesty | low | **✅ LANDED (DOC)** | The arm32.c file header still described the *pre*-fpu-01/fpu-03 behavior (synth_segment_m7 re-seeds the f64 oscillator every block; "standing fix"). Rewritten to the shipped truth: seed once at activation, (c,s) carries across blocks bounded by the ALU renorm, audio steady state FP-free. |
| redundant-arm-math-include | clean | info | **✅ LANDED** | arm32.c `#include "arm_math.h"` twice (each under its own `#if SPECTRAL_USE_CMSIS`); the 2nd is a header-guarded no-op. Removed. No codegen change (perf gate green). |
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
