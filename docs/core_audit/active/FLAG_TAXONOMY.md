# Build-Flag Taxonomy — one rule for the preprocessor gates

**Status:** PROPOSED (awaiting maintainer sign-off + embedded-agent coordination). Branch
`embedded-arch-audit`. Answers the maintainer's headline concern #4: *"many precompiler flags are
erroneously named. There's no consistent way we do this or no rule we follow and there needs to be."*

This doc proposes the **rule**, inventories every gate by **axis**, and gives the **old→new map**.
**No renames land until this is blessed.** When it is, this moves to `reference/` as canon, AI.md's
inline flag list is replaced by a pointer here, and `test_build_flag_taxonomy` enforces it (see the
audit ledger). The semantic collapses (embedded-sim triplet, the "restricted" cluster, the host-guard
predicates) overlap the determinism workstream and are **propose-not-prescribe** — flagged below.

---

## 1. The problem

~25 gating identifiers with no convention. The same axis is spelled five different ways:

- **Capability** uses `USE_` (`SPECTRAL_USE_VDSP/CUDA/CMSIS`) *and* `HAS_` (`SPECTRAL_HAS_DMA/CHIRP/
  DUAL_MAC/FILE_IO`) *and* `*_HAS_*` (`SPECTRAL_HASH_HAS_HOST_FILE_API`, `SPECTRAL_FS_HAS_MMAP`).
- **Exec mode** uses `IS_` (`SPECTRAL_IS_EMBEDDED_SIM`), bare (`SPECTRAL_EMBEDDED_SIMULATION`),
  `USE_` (`SPECTRAL_USE_EMBEDDED_SYNTH`), `_MODE` (`SPECTRAL_RESTRICTED_MODE`), `_PROFILE`
  (`SPECTRAL_RESTRICTED_PROFILE`) — three of them name the *same* concept.
- **Approximation** uses `ENABLE_APPROX_` for five gates but `SPECTRAL_PRECISE_PHASE` (inverted sense)
  and `SPECTRAL_METAL_FAST_MATH` escape the family.
- **Target** mixes `SPECTRAL_EMBEDDED` (the master switch), `SPECTRAL_ARM_M7`, `SPECTRAL_PLATFORM_DAISY`,
  `SPECTRAL_EMBEDDED_FLOAT` with no stated relationship.

There is no SSOT doc and no test; AI.md's "Build flags that gate code" names 6 of ~25 and rots.

## 2. The axis model

Every gate belongs to exactly **one** of five axes. The prefix names the axis:

| Axis | Prefix | Question it answers | Value convention |
|---|---|---|---|
| **Target** | `SPECTRAL_TARGET_*` | Which silicon/board is this binary for? | always-defined 0/1 |
| **Mode** | `SPECTRAL_MODE_*` | How does this build run (real / host-sim / restricted)? | always-defined 0/1 |
| **Capability** | `SPECTRAL_HAS_*` | Is this device/library/HW feature available? | always-defined 0/1 |
| **Approximation** | `SPECTRAL_APPROX_*` | Is the gated fast approximation enabled (exact is the default reference)? | always-defined 0/1 |
| **Debug** | `SPECTRAL_DEBUG_*` | Instrumentation / diagnostics only — never changes audio. | presence or 0/1 |

**Sub-rules (each is a `test_build_flag_taxonomy` assertion):**
1. **One axis per flag.** Every consumed `SPECTRAL_*` `#if` token matches exactly one prefix.
2. **No synonyms.** At most one selector per concept (one embedded-sim entry flag, one "restricted"
   selector). The triplet/cluster collapses below enforce this.
3. **Defined, not presence.** Boolean gates are `#define X 0|1` and tested `#if X`, never
   `#ifdef X` — so a typo'd flag is a compile signal, not a silent-off. (Keeps the existing
   `#ifndef X / #define X default / #endif` overridable pattern; the audit's `config-unguarded-tunables`
   closes the two bare ones.)
4. **Approximation default is exact.** `SPECTRAL_APPROX_* == 0` means the exact reference path
   (AI_CANON correctness-before-performance). `PRECISE_PHASE` (inverted) folds in as `APPROX_PHASE`.
5. **Capability is observed, not chosen.** `HAS_` flags describe the toolchain/board; the build
   system sets them, code only reads them. `USE_VDSP/CUDA/CMSIS` become `HAS_VDSP/CUDA/CMSIS`.

## 3. Old → new map

Legend: **M** = mechanical grep-replace (define + every `#if` site, no semantic change).
**S** = semantic (collapses or re-senses flags — needs review). **E** = embedded-agent-owned
(determinism/restricted surface — propose only).

### Target — `SPECTRAL_TARGET_*`
| Current | Proposed | Kind | Note |
|---|---|---|---|
| `SPECTRAL_EMBEDDED` | `SPECTRAL_TARGET_FIRMWARE` (keep `SPECTRAL_EMBEDDED` as alias?) | S/E | The master "real firmware" switch; renaming it touches ~37 sites + the emulator-guard idiom. High blast radius — decide whether to alias or hard-rename. |
| `SPECTRAL_ARM_M7` | `SPECTRAL_TARGET_ARM_M7` | M/E | M7 codepath; perf-gated arm32 reads it. |
| `SPECTRAL_PLATFORM_DAISY` | `SPECTRAL_TARGET_DAISY` | M/E | Board binding. |
| `SPECTRAL_EMBEDDED_FLOAT` | `SPECTRAL_TARGET_ARM_FLOAT` | S/E | Currently a byte-identical no-op (audit `dead-opt-level-dup`): wire the FPU synth path or retire. |

### Mode — `SPECTRAL_MODE_*`
| Current | Proposed | Kind | Note |
|---|---|---|---|
| `SPECTRAL_IS_EMBEDDED_SIM` | `SPECTRAL_MODE_EMBEDDED_SIM` | S/E | The derived "host build modeling embedded" flag. |
| `SPECTRAL_EMBEDDED_SIMULATION` | *(collapse into the one entry flag)* | S/E | Audit `flag-embedded-sim-triplet`: 3 flags always set together. |
| `SPECTRAL_USE_EMBEDDED_SYNTH` | `SPECTRAL_MODE_ARM32_SYNTH` (the synth-redirect axis) | S/E | The only distinct semantic: gates synth_cpu→arm32 redirect. Make it the one master, derive the rest. |
| `SPECTRAL_RESTRICTED_MODE` | `SPECTRAL_MODE_RESTRICTED` | S/E | Audit `flag-restricted-noperf`: 4 "restricted" spellings — pick one. |
| `SPECTRAL_RESTRICTED_PROFILE` | *(collapse → MODE_RESTRICTED + a HAS_ for the profiler)* | S/E | |
| `SPECTRAL_NO_PERF` | `SPECTRAL_MODE_NO_PERF` or `SPECTRAL_HAS_PERF` (invert) | S/E | Inverted-sense; prefer a positive `HAS_PERF`. |
| `SPECTRAL_EXEC_MODE_*` (strings) | keep (diagnostic string vocabulary, not gates) | — | Move out of config.h to runtime diagnostics (audit `config-resolution-strings`); they mirror the mode flags 1:1. |

**Host-guard SSOT (audit `flag-host-guard-divergence`).** Three spellings of "host but not real
firmware" guard identical carve-outs: `#ifndef SPECTRAL_USE_EMBEDDED_SYNTH` (synth_cpu.c),
`#if !SPECTRAL_EMBEDDED` (oscillator.c), `#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM`
(renderer.c). Define **one** predicate pair with a one-line truth table:
`SPECTRAL_MODE_HOST` (desktop or embedded-sim — has an OS) and `SPECTRAL_MODE_FIRMWARE`
(real bare-metal). Convert the three guards to it. (E — overlaps the emulator-guard idiom the
embedded agent uses; coordinate before converting.)

### Capability — `SPECTRAL_HAS_*`
| Current | Proposed | Kind |
|---|---|---|
| `SPECTRAL_USE_VDSP` | `SPECTRAL_HAS_VDSP` | M |
| `SPECTRAL_USE_CUDA` | `SPECTRAL_HAS_CUDA` | M |
| `SPECTRAL_USE_CMSIS` | `SPECTRAL_HAS_CMSIS` | M |
| `SPECTRAL_HAS_DMA` / `_CHIRP` / `_DUAL_MAC` / `_FILE_IO` | unchanged (already `HAS_`) | — |
| `SPECTRAL_HASH_HAS_HOST_FILE_API` | `SPECTRAL_HAS_HOST_FILE_API` | M |
| `SPECTRAL_FS_HAS_MMAP` | `SPECTRAL_HAS_MMAP` | M |
| `SPECTRAL_LUT_IN_FLASH` | `SPECTRAL_HAS_FLASH_LUT` | M/E |

### Approximation — `SPECTRAL_APPROX_*`
| Current | Proposed | Kind |
|---|---|---|
| `SPECTRAL_ENABLE_APPROX_TRIG` / `_INV_SQRT` / `_ATAN2` / `_Q15_BOUNDARY` / `_PEAK_LOG` | `SPECTRAL_APPROX_TRIG` / `_INV_SQRT` / `_ATAN2` / `_Q15_BOUNDARY` / `_PEAK_LOG` | M |
| `SPECTRAL_PRECISE_PHASE` | `SPECTRAL_APPROX_PHASE` (invert sense: 0 = exact cubic) | S | Audit `precise-phase-unbuilt`: currently no build sets it. |
| `SPECTRAL_METAL_FAST_MATH` | `SPECTRAL_APPROX_METAL_MATH` | M |

> Note: `core/spectral_guarantees.h` + `test_guarantees.c` already pin the drift budget per approx
> gate. Renaming must keep that mapping (the gate name is part of the CORE_CONTRACTS manifest).

### Feature/layout (folds into Mode or stays a named choice)
| Current | Proposed | Kind | Note |
|---|---|---|---|
| `SPECTRAL_SOA_ACTIVE` | `SPECTRAL_MODE_SOA` (or derive from target) | S/E | Audit `flag-soa-cpu-derived`: auto-derived from the CPU; may not need to be a public flag. |
| `SPECTRAL_SYNTH_DETERMINISTIC_PARTITIONS` | unchanged (a numeric tunable, not a boolean axis) | — | Keep; it's a count, not a mode gate. |
| `SPECTRAL_COMPACT_SEG` / `SPECTRAL_Q15_COMPACT` | `SPECTRAL_MODE_COMPACT_SEG` / `_Q15` | S/E | Layout modes. |

### Debug — `SPECTRAL_DEBUG_*`
`SPECTRAL_DEBUG`, `SPECTRAL_DEBUG_ARM`, `SPECTRAL_DEBUG_RESTRICTED` already conform. The
`SPECTRAL_TRACK_*` instrumentation knobs (`_DEBUG_TIMING`, `_PREFETCH_PHASE`, …) are a sub-namespace —
keep `SPECTRAL_TRACK_*` as the peak-tracker tuning family (numeric/instrumentation), distinct from the
five gating axes. (Mostly embedded-perf-adjacent — E.)

## 4. Migration plan (after sign-off)

1. **Mechanical pass (M rows).** `HAS_` capability rename + `APPROX_` rename. Pure grep-replace of the
   define and every `#if` site; build-verify all targets byte-identical. Low risk, biggest readability win.
2. **Host-guard SSOT.** Land `SPECTRAL_MODE_HOST`/`_FIRMWARE`, convert the three divergent guards.
   Coordinate with the embedded agent (emulator-guard idiom).
3. **Semantic collapses (S/E rows).** Embedded-sim triplet → one master; "restricted" cluster → one;
   `PRECISE_PHASE`→`APPROX_PHASE`; `NO_PERF`→`HAS_PERF`. Owned-jointly with the determinism workstream.
4. **Target rename.** `SPECTRAL_EMBEDDED`→`SPECTRAL_TARGET_FIRMWARE` last (highest blast radius); decide
   alias-vs-hard-rename.
5. **Land the test + docs.** `test_build_flag_taxonomy`, repoint AI.md, move this doc to `reference/`.

## 5. Open questions (maintainer + embedded agent)

1. **`SPECTRAL_EMBEDDED` rename:** alias (keep the name, add `TARGET_FIRMWARE` as the canonical) or
   hard-rename ~37 sites + the emulator-guard idiom? (Recommend: alias first, deprecate over time.)
2. **Embedded-sim collapse:** confirm `USE_EMBEDDED_SYNTH` is the one true master and the other two
   derive from it (the audit found all three set together at `CMakeLists.txt:131-133`).
3. **`NO_PERF` / `PRECISE_PHASE` sense inversion:** OK to flip to positive `HAS_PERF` / `APPROX_PHASE`?
4. **Scope split:** which rows does the determinism agent own vs this campaign? (Proposed: all `TARGET_*`
   + `MODE_RESTRICTED*` + `SOA`/`COMPACT` + `TRACK_*` are theirs; `HAS_*` capability + `APPROX_*` are this
   campaign's mechanical pass.)
