# Determinism Surface — the canonical ARM-M performance & WCET contract

**Status:** PROPOSED (awaiting maintainer write-off). Branch `embedded-arch-audit`.
**Owner question:** *"Does device D deterministically meet a 512-voice synthesis budget
at its rated clock?"* This document defines what we mean by *determinism*, the single
surface that answers it, and the universal ARM-M debug contract shared between QEMU CI
and real hardware.

This is a **unify-and-extend** of the existing layered perf stack (M7_PERF_MODEL_PLAN
P0–P6), not a rebuild. CLAUDE.md §2/§3: reuse what exists, surgical changes.

---

## 0. Definition of determinism (the standard, from now on)

> **Determinism(device D) := the modeled worst-case execution time (WCET) of one audio
> block at the canonical workload (512 active voices) is ≤ the real-time budget of D, with
> the documented safety margin, under the device's worst-case memory/cache state.**

- Determinism is **always tethered to a device.** D ∈ {`daisy_seed` (STM32H750 @480 MHz),
  `teensy41` (i.MX RT1062 @600 MHz), …}. Same Cortex-M7 architecture ⇒ same counter
  contract; **different** clock, cache, TCM, and external-memory timing ⇒ different budget
  and different WCET. The gate runs per-device.
- The canonical workload is **512 voices** (the maintainer's intelligibility target).
- "Meets the budget" is a **PASS/FAIL** at the top of the report.

---

## 1. Research grounding (what is and isn't sound) — deep-research, 2026-06-21

Verified claims (adversarial 3-vote; sources cited). Full set in the session research log.

1. **Measurement-based WCET is UNSOUND.** End-to-end measurement over a subset of inputs
   *underestimates* the true WCET; only methods considering *all* executions give a sound
   bound (Wilhelm et al. TECS'07, 3-0; AbsInt aiT, 3-0; ECRTS'23, 3-0). Direct measurement
   determines time for *one* input and cannot infer all inputs.
2. **Local worst ≠ global worst (timing anomalies).** On cores with caches/pipelines you
   cannot greedily pick the worst case per instruction; *domino effects* mean a difference
   in pipeline start-state need not decay and cannot be bounded by a constant (TECS'07 3-0,
   AbsInt 3-0). ⇒ a "worst-case start state + sum of local worsts" is not automatically safe.
3. **Sound WCET = abstract interpretation (over-approx all cache/pipeline states) + ILP/IPET
   path analysis** (AbsInt 3-0). This is the aiT method; it is per-architecture, suffers
   state explosion, and is "time-consuming and error-prone without the processor's
   micro-architectural details" (ECRTS'23 3-0). **We are not building this.**
4. **The M7 is the hard case.** Tight-analysis error roughly *quadruples* from Cortex-M4
   (23.8% MAE) to Cortex-M7 (102% MAE) — 6-stage dual-issue in-order pipeline, **random-
   replacement** I/D caches, branch predictor (ECRTS'23, 2-0). Our cycle estimates carry
   real uncertainty; the answer is conservative bounding + variability elimination, not
   false precision.
5. **QEMU TCG is functional, not cycle-accurate** (qemu.org docs — primary, unverified only
   because the verifier hit the session limit). It gives exact *instruction/byte counts*,
   never cycles. Cycle estimates must come from a pipeline model (llvm-mca) or hardware.
6. **The QEMU+llvm-mca hybrid is a real technique** (MCAD, arXiv 2201.04804, <3% gmean error
   vs HW counters) — **but MCAD supports Cortex-A57/Skylake/POWER10, NOT Cortex-M7.** We use
   llvm-mca's `CortexM7Model` directly on extracted loops; this is why `mca_validation.py`
   (validate the model vs hand-derived TRM timings) is load-bearing and must stay.
7. **DWT erratum 850724 — counter mis-attribution.** On M7 r0p1/r0p2/r1p0 (fixed r1p1):
   LSU stall cycles increment **CPICNT** instead of LSUCNT; FPU lazy-stacking cycles land in
   **LSUCNT/CPICNT** instead of EXCCNT. No cycles are lost or double-counted — only
   mis-attributed — and **ARM gives no workaround** (Cortex-M7 errata notice, primary). ⇒ on
   affected silicon the per-unit split of stalls is unreliable; the contract must record the
   CPUID revision and either report the affected counters as a *combined* "stall" bucket or
   flag the attribution as erratum-suspect. Reading them cleanly is unsound.
8. **DWT profiling counters have inherently limited accuracy** and (CPICNT/EXCCNT/SLEEPCNT/
   LSUCNT/FOLDCNT) are **8-bit, saturating at 255** (ARMv7-M ARM). Reliable use = read +
   accumulate + clear frequently (per-block), never free-run across a long window.
9. **TCM placement / cache discipline is THE determinism lever on M7** (Microchip
   DS90003186A "deterministic code performance"; AN5740 TCM; Feabhas). ITCM/DTCM are
   single-cycle, not cached ⇒ no miss variability. The existing DTCM strategy is correct.
10. **Cortex-M7 (ARMv7E-M) has NO PMU.** The DWT profiling counters are the only HW perf
    counters (contrast ARMv8.1-M M55/M85, which add a real PMU). So per-unit (FPU/MAC/ALU)
    *utilization* has **no direct counter** on either target device.

**Consequence for our WCET stance (the honest framing):** we ship a **conservative hybrid
WCET bound**, explicitly labeled *not a formally-sound aiT-class proof*, made trustworthy by
(a) bounding each layer at its conservative extreme (cold cache = all-miss, CPI bound, one
mispredict per data-dependent branch), (b) **eliminating variability** by keeping the hot
state/code in TCM and the worst-case path well-characterized, and (c) a **named, justified
safety margin** on top. This is what `wcet.py` already does in spirit; we make the soundness
argument, the margin, and the device-tethering explicit.

---

## 2. What exists today (keep / extend / clean)

| Layer | Module | Verdict |
|---|---|---|
| L0 census (insns from codegen) | `embedded/codegen.py` | KEEP |
| L1 counts (insns/bytes/lines, QEMU TCG) | `embedded/counts.py` + `native/qemu/spectral_counts.c` | KEEP |
| L2 cycles (llvm-mca CortexM7Model) + validation | `embedded/codegen.py`, `embedded/mca_validation.py` | KEEP (the soundness anchor) |
| L3 stall bounds (modeled, from trace) | `embedded/memory_model.py` | KEEP, generalize device |
| WCET composition | `embedded/wcet.py` | EXTEND (device, BCET/ACET, margin) |
| Baseline + stone gate | `embedded/expectations.py` | EXTEND (512@480 stone, per-device) |
| Unified report + pretty tree | `core/report_output.py` → `BranchFormatter` | KEEP (the one reporting surface) |
| CLI entry | `testing/benchmark_workflow.py` (`m7-*`, `measure`) | EXTEND (device flag, `deterministic`, `hotspots`) |
| On-device DWT | `arch/arm/spectral_debug_embedded_arm.c` | RE-TETHER to the shared contract; erratum-aware |
| Device constants SSOT | `api/daisy_seed/daisy_seed_sdram.h` | GENERALIZE to per-device profiles |

**Already good and not to be reinvented:** the measured-vs-modeled provenance discipline,
the one report shape `{suite, context, tests:[{name,status,summary,details}]}`, the
generate/verify baseline pattern, the C-truth SSOT rule (no Python copies of C facts).

---

## 3. Design

### 3.1 Device abstraction (`DeviceProfile`)
Promote the Daisy-only `SPECTRAL_DAISY_*` set to a device-keyed profile. Each device is a
**C header SSOT** (C-truth rule) parsed at runtime; Python holds no copy:
- `devices/spectral_device_daisy_seed.h` (rename/extend of `daisy_seed_sdram.h`)
- `devices/spectral_device_teensy41.h` (NEW)

Carried params (research §7): `cpu_hz`, `sample_rate`, `block_size`; I-cache/D-cache bytes +
ways + line + **replacement policy** (random on M7 — affects the cache-model assumption);
ITCM/DTCM sizes; external-memory class + full timing set (the SDRAM/SEMC timings); CPUID
implementer/variant/**revision** (for erratum 850724). `memory_model.ModelConstants` becomes
`DeviceProfile.from_header(path)`; the gate iterates devices.

### 3.2 The canonical determinism gate
New stone scenario in `expectations.py`, **per device**:
```
{"device":"daisy_seed","active":512,"block":48,"scan_segments":<worst>,
 "max_budget_fraction": 1.0 - SAFETY_MARGIN}
```
`benchmark_workflow deterministic [--device daisy_seed|teensy41|all]` →
top-line **PASS/FAIL**: `WCET(512, worst-path, cold-cache) ≤ budget·(1−margin)`.
`SAFETY_MARGIN` is named + justified in `expectations.py` (research §1/§4 demand it; the M7
102% MAE is *why* the margin exists, sized from the mca-validation residual + a documented
engineering reserve). The existing 32/128-voice stones stay as regression history.

### 3.3 One ARM-M debug contract (QEMU ⇄ hardware)
A single schema — `SpectralPerfSample` — that **both** producers fill and the **same**
`render_report` emits:
- **QEMU/CI producer** (Python): insns/bytes/lines (L1), modeled cycles/IPC (L2), stall
  bounds (L3), WCET/BCET/ACET (composition).
- **On-device producer** (C, `spectral_debug_embedded_arm.c`): DWT CYCCNT (cycles/IPC vs the
  census insn count), CPICNT/LSUCNT/FOLDCNT/EXCCNT/SLEEPCNT (stalls/idle/interrupts/dual-
  issue), per-block min/avg/peak (BCET/ACET/WCET-observed), xruns. Emitted as the same
  key/value record over semihosting/ITM, parsed back into the **same Python report shape** →
  a real-hardware run is line-comparable to the model prediction (the consistency goal).

The contract is **versioned** and lives in one header (`spectral_perf_contract.h`) included
by the on-device code and mirrored by a Python dataclass — the field set cannot drift.

### 3.4 The metric set — method per metric (research-grounded; no toy heuristics)

| Metric | CI (QEMU/model) | On-device (DWT) | Soundness note |
|---|---|---|---|
| Instruction counts | L1 TCG exact | census insns | exact |
| Bandwidth (ld/st bytes, 32B lines) | L1 plugin exact | LSU bytes (approx) | exact in CI |
| Cycles | L2 mca (modeled) | CYCCNT (32-bit, exact) | CI modeled, HW exact |
| **IPC** | mca cyc / L1 insns | CYCCNT / census insns | both first-class now |
| **Stalls / CPI** | L3 bound + mca CPI | CPICNT (**erratum-aware**) | report as range; flag erratum |
| **Idle / wall** | n/a (no idle in render) | SLEEPCNT + wall via CYCCNT | HW only |
| **Interrupts** | n/a | EXCCNT (**erratum-aware**) | HW only; affected by 850724 |
| Dual-issue / fold | mca port pairing | FOLDCNT | corroborating |
| **FPU/MAC/ALU saturation** | **llvm-mca port pressure** (defensible) + instruction-mix from L0 disasm; CPICNT *stall attribution* as corroboration only | n/a (no per-unit counter, §10) | **the honest answer: there is NO direct counter; saturation = llvm-mca resource-pressure per port over the hot loop, cross-checked against the static instruction mix. We report it as "modeled port pressure," never as a measured utilization.** |
| **Hottest functions/lines** | new `hotspots` cmd: PC-histogram TCG plugin + `addr2line -afi` on a `-g` ELF | n/a | promote the by-hand technique to first-class |

The "not a toy" requirement for unit saturation is met by **honesty + method**: M7 has no FPU/
MAC counter (§10), so utilization is *modeled* from llvm-mca's per-port resource pressure on
the extracted hot loop (the same model `mca_validation.py` already trusts), corroborated by the
static instruction mix and the CPICNT stall signal — and **labeled modeled, with the erratum
caveat**, not presented as a hardware measurement.

### 3.5 WCET / BCET / ACET
- **WCET** (gates determinism): the existing conservative composition (t1 mca+backedge,
  t2 residual×CPI-bound+mispredict, t3 cold-all-miss) + the explicit margin + the soundness
  caveat from §1. Most documentation, most diligence.
- **ACET** (expected): the measured counts run (typical fixture) → typical cycles.
- **BCET** (best case): warm-cache / no-mispredict lower bound from the same mca bodies.
All three in the report; only WCET gates.

### 3.6 Cleanup (as we go)
- De-dup `qemu_main.c` / `qemu_render_main.c` (shared setup header; only the output sink
  differs) — flagged earlier this session.
- Audit: every Python perf tool emits through `render_report` (consistency requirement);
  list any straggler that prints ad-hoc and route it through.
- Fold `m7-census/counts/mca-validate/stalls/wcet/baseline` under a clear command taxonomy;
  keep them but document the layer each belongs to.

### 3.7 README
`tools/spectral_tools/performance/embedded/README.md`: the layer model, the determinism
definition, **how to generate each metric** (one command per row of §3.4), the device
abstraction, and the soundness/erratum caveats. Python wrapper = the single entry point.

---

## 4. Phasing (verifiable; maintainer sets order)

- **P1 — Device abstraction.** `DeviceProfile` + Daisy header generalization + Teensy header;
  `memory_model`/`wcet`/`expectations` consume it. Verify: existing gate green for Daisy
  (byte-identical numbers), Teensy profile parses + produces a budget.
- **P2 — Determinism gate.** `deterministic` command + the 512@480 stone (per device) + the
  named safety margin. Verify: PASS/FAIL is correct and re-derivable; current 512 number
  reproduced from §earlier-report (258 det / 420 sustain @400; recompute @480).
- **P3 — Unified contract + on-device re-tether.** `spectral_perf_contract.h` + Python mirror;
  `spectral_debug_embedded_arm.c` emits the contract record; erratum-revision handling.
  Verify: a QEMU report and a (host-simulated) on-device report share the schema; round-trip
  parse test.
- **P4 — Metric completion.** IPC + BCET/ACET surfaced; `hotspots` command; saturation =
  modeled port-pressure. Verify: each metric has a command + a test.
- **P5 — Cleanup + README.** De-dup, reporting-surface audit, README. Verify: docs build, all
  tools route through `render_report`.

Each phase: ctest/pytest green + the m7 perf gate green (re-sign only on a deliberate move).

---

## 5. Open decisions for the maintainer (write-off)

1. **Safety margin size.** What reserve below 100% budget = "deterministic"? (Proposal:
   derive a floor from the mca-validation residual, then a documented engineering reserve —
   e.g. gate at ≤85–90% of budget. The M7 102% MAE argues for a *generous* reserve unless we
   commit to cache-off / fully-TCM-resident determinism, which tightens it.)
2. **Clock of record.** Daisy @480 (hardware max) vs @400 (conservative). The model currently
   says 400; you've set 480 as the determinism standard — confirm we move the SSOT to 480
   (and that the firmware actually runs 480).
3. **Cache stance for the guarantee.** Accept the cold-all-miss bound (safe, loose), or commit
   to TCM-resident hot path + cache-off-or-warm determinism (tighter, more work, but the
   research says this is the *real* M7 determinism lever).
4. **Teensy scope now or stub.** Build the Teensy profile fully in P1, or land the abstraction
   with Daisy only and add Teensy when hardware is in hand.
5. **Re-run the research synthesis?** The harness died on a session limit after verifying 10
   claims; the substance is captured here. Re-run later for the polished cited report, or
   treat this section as the synthesis.

## 6. Status (live) + maintainer decisions (RESOLVED)

Write-offs (all RESOLVED): cache stance = **TCM-resident + cold-miss bound**; margin =
**derived from validation residual + reserve = 10%** (gate ≤90%); Teensy = **Daisy-first,
stub**; clock of record = **480 MHz (committed)**; worst case = **harshest reachable
production input, no test-softening**.

**LANDED:**
- **P1a (1c710865)** — Daisy clock → 480 MHz (SSOT + on-device aligned + baseline re-signed;
  audio checksum 7ed6e0ac unchanged; gate green). Firmware app must select libDaisy
  FREQ_480MHZ (documented in-header, HW-PENDING).
- **P2 (3895c5b7)** — the `deterministic` command: PASS/FAIL for N voices @ rated clock via
  the unified report, reusing the validated WCET stack + the derived margin + the
  deterministic ceiling.
- **note #2 PROVEN (d50710d3)** — `test_determinism_invariance.py`: per-voice cost is
  param-invariant (amplitude exact; frequency within ≤4 insns/voice — the bounded activation
  seed trig). The determinism guarantee holds across all user parameters.
- **note #1 ANALYSED** — see below.
- **P5 partial** — `README.md` (how to generate every metric + the soundness/erratum caveats).

**VERDICT (the honest headline):** 512 voices @ 480 MHz = **~154% of budget → FAIL** on the
exact oscillator; deterministic ceiling **~277 voices** (worst-case) / ~504 (all-sustain).

**Note #1 — path to 100% at 512 (analysis CORRECTED 2026-06-21 after reading synth_fade_m7):**
the worst case is 512 UNPAIRED fade kernels (t1 ≈ 565k). **CORRECTION to the earlier estimate:**
the fade kernel applies TWO multiplicative envelopes per sample — `faded = sample·fade_val`
then `accum += faded·am` (arm32.c:791-792) — whereas sustain applies ONE (`accum += sample·am`).
So a paired-fade canNOT just reuse the SMLALD dual-MAC the way sustain pairing does: the
per-lane `fade_val` multiply stays per-lane. Pairing therefore saves only the dual-MAC + the
loop/ramp overhead, NOT the fade multiply → ~30-40% on the fade kernel (worst_cyc 23→~15), so
t1 565k→~368k, WCET ~540k = **~112%** (NOT the ~95% first estimated). Combined with
**TCM-residence** (chosen cache stance: ctx/accum/kernels→DTCM/ITCM, t3 90k→~30k) → WCET
~480k = **~94-100%, MARGINAL** at the 90% gate. (For the da≈0 worst case the two envelopes
collapse to one linear ramp and pairing IS clean, but the adversary picks da≠0, so the strict
WCET needs the two-envelope paired kernel.) **Net: the exact-oscillator path to 512 is
marginal AND costly (a complex two-envelope paired-fade kernel + HW-gated TCM placement); the
IFFT is the clean route to 512** — confirming it as the right next phase. The fade-pairing
(partial, ~112%) and the SMMUL fade-split (byte-identical, 154→149%) remain available as
incremental worst-case reductions if 512-on-the-oscillator is pursued, but neither closes it
alone.

**ALSO LANDED:**
- **P4 (host metrics DONE)** — `hotspots` (per-PC histogram → addr2line inlined func+line;
  verified @512: smmul 27% / __qadd 20% / pair 18%); **WCET/ACET/BCET + sustain IPC** in the
  `deterministic` report (512@480: WCET 738k FAIL / ACET 505k ~105% / BCET 332k 69%);
  **`saturation`** = modeled FPU/MAC/ALU/SIMD per-unit utilization (llvm-mca port pressure;
  all kernels latency/dependency-bound, FPU 0% = integer recurrence). The only metric still
  pending is the on-device-measured set (idle/interrupts/HW cycles) = P3.
- **P5 (partial)** — README updated (every metric's command + soundness caveats).

**REMAINING:** P1b device abstraction (DeviceProfile + Teensy stub); P3 the unified
SpectralPerfSample contract (versioned header + Python mirror tethering on-device DWT for
idle/interrupts/HW cycles, erratum-aware); P5 finish (qemu_main/render dedup +
reporting-surface audit); and the note-#1 fade-pairing + TCM-residence engine work toward
512 (marginal — IFFT is the clean route).
