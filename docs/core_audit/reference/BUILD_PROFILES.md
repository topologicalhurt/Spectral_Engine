# Build profiles — the flag philosophy (canon, no plan)

The single source of truth for compiler flags is **`spectral_engine/cmake/profiles.cmake`**:
it defines the flag GROUPS (with rationale); `host-config.cmake` (desktop + host-embedded
sim + tests) and `daisy-config.cmake` (real Cortex-M7 firmware) assemble each target's
flag list from those groups. No flag is decided anywhere else. This doc states the
*philosophy* the groups encode; the groups themselves are the authority for exact tokens.

## Two profile philosophies

| Profile | Targets | Philosophy |
|---|---|---|
| **HOST** | `desktop`, `cuda`, the test/bench exes | **Quality but well-optimized.** `-O3` + aggressive auto-vectorization (`-funroll-loops -ftree-vectorize`, clang `-fvectorize -fslp-vectorize`) + host fast-math + `-march=native` + LTO, with full `-Wall -Wextra` kept on. AVX-512 capped OFF (see below). Accuracy preserved where the engine's gated approximations require it; speed taken where free. |
| **EMBEDDED (host-sim)** | `simulate`, `simulate_daisy`, `embedded_arm{,_float,_restricted}` | **Aggressively minimal, deterministic** — the desktop *mirror* of the firmware, built with the host toolchain. `-O3` + section GC + the embedded fast-math group; differs from real firmware only in toolchain/arch. |
| **FIRMWARE** | `daisy` (real STM32H7) | **Aggressively minimal, fastest-for-the-target, deterministic.** Board `-O` level + section GC + no unwind/exception tables + MCU arch flags (`-mcpu=cortex-m7 -mthumb -mfpu=… -mfloat-abi=…`) + SAFE fast-math → small, WCET-stable code. |

## The math/precision knob (one per profile)

Math precision is decided **once per profile**, not re-chosen per flag:

- **`SPECTRAL_REPRO_BUILD`** (host + host-sim) — ON drops the host/embedded fast-math
  group for a bit-reproducible build. Defaults ON when `SPECTRAL_PRODUCTION_BUILD` is ON.
- **`SPECTRAL_DAISY_SAFE_MATH`** (firmware) — ON (default) drops the firmware unsafe-math
  group for deterministic firmware.

The group definitions live in `profiles.cmake`; the conditional assembly that reads these
knobs lives in `host-config.cmake` / `daisy-config.cmake` (they run after `options.cmake`
sets the knobs).

## AVX-512 is OFF by default

`-mno-avx512f` caps AVX-512 off on x86 (`profiles.cmake`, `spectral_profile_host_native_isa`).
Rationale: a wider SIMD lane that **down-clocks** the core can net-lose on real workloads,
and the engine has **no 512-bit kernel** to enable — lifting the cap alone changes nothing.
Turning it on is a deliberate, *measured* decision (a whole-program win net of downclock on
real AVX-512 silicon), tracked as x86-CI-gated work (QTYPE Thread C). The two x86 ISA flags
(`-mavx2 -mno-avx512f`) are **arch-gated**: emitted only when `CMAKE_SYSTEM_PROCESSOR` is
x86 — on arm64 they are `-Wunused-command-line-argument` noise (NEON is already on via
`-march=native`). See the SIMD-width policy (KERNEL_HARDENING_PLAN §VII): widest lane
available **unless** its latency is worse.

## CMAKE_BUILD_TYPE is NOT the optimization control

The engine always compiles at its **profile** `-O` level. `CMAKE_BUILD_TYPE` only adds
`-g`/drops `NDEBUG`; a `Debug` configure is still `-O3` fast-math + `-g`. Do not expect a
debuggable `-O0` build from `-DCMAKE_BUILD_TYPE=Debug` today — that would be a separate
real Debug profile (a tracked decision). This is intentional: the perf and parity gates
measure the optimized engine, so the default build must stay optimized.

## Build reproducibility caveat (ThinLTO + fast-math)

The default host build uses **ThinLTO** (`-flto=thin`) + fast-math. ThinLTO's parallel
backend is **not bit-reproducible across builds under load**, and combined with fast-math
this has been observed to produce a *grossly* wrong result on rare draws (the
`arm32_process_correctness` re-seed SINAD test fell to ~28 dB vs a 70 dB floor on ~1 build
in 13, with byte-identical compiler inputs). This is a latent reproducibility/UB hazard,
NOT a flag-policy issue — tracked as a correctness finding in
`../KERNEL_HARDENING_PLAN.md` (build-reproducibility). `SPECTRAL_REPRO_BUILD` exists to
get a bit-reproducible build when that matters.

## Adding a profile or flag

Add the group to `profiles.cmake` with a one-line rationale; have the consuming
`*-config.cmake` reference it. Test executables get dead-strip via the shared
`spectral_apply_dead_strip(<target>)` helper — never re-inline the `-Wl,-dead_strip` /
`--gc-sections` idiom.
