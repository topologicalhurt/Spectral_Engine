# Patch notes — Pass 147: device-agnostic memory-class placement (Phase E, increment 1)

## Problem

Device-specific detail lived in the core: `spectral_config.h` hardcoded STM32H7
linker sections (`.dtcm_data` / `.itcm_text` / `.sdram_data`), sizes (128 KB
DTCM), and the Cortex-M cache line, used directly by `arm32.c` via
`SPECTRAL_DTCM` / `SPECTRAL_ITCM` / `SPECTRAL_SDRAM`. The core must be
device-agnostic — it may distinguish embedded/low-level from host, but must not
name a specific device (ULTRAPLAN Phase E).

## Change

Introduce `core/port/spectral_mem.h`, a device-agnostic memory-class interface by
intent:

```text
SPECTRAL_MEM_FAST       (was DTCM)   hot data, fastest/zero-wait memory
SPECTRAL_MEM_FAST_CODE  (was ITCM)   hot code, fastest instruction memory
SPECTRAL_MEM_BULK       (was SDRAM)  large/bulk data, possibly external
SPECTRAL_CACHE_LINE                  alignment
```

Binding resolution (asm-generic / CMSIS Device-Family-Pack pattern): a BSP
overrides via `SPECTRAL_BSP_MEM_HEADER`; else a built-in Cortex-M default applies
on a real Cortex-M cross-compile; else portable no-ops (host/sim). `spectral_config.h`
includes the port header instead of defining sections; `arm32.c` uses the intent
macros. Dropped the unused `SPECTRAL_DTCM_SIZE` / `_KB`.

## Verification

Behavior-preserving on every host build (the macros resolve identically to before):

```text
- desktop / simulate (ARM_M7=0): no-op placement, CACHE_LINE 64 — as before.
- forced-M7 harness (ARM_M7=1 host): no-op placement, CACHE_LINE 32 — as before;
  ctest green, audio unchanged (peak 0.499, 990 Hz); sim oracle green.
- real Cortex-M cross-compile: identical .dtcm_data/.itcm_text/.sdram_data binding
  (gate unchanged) — not built here (no ARM toolchain in this environment).
```

## Scope (Phase E increment 1 of N)

This is the device-coupling-removal increment. The built-in Cortex-M section
default still resides in `spectral_mem.h`; moving it fully into the `api/` Daisy
BSP (via `SPECTRAL_BSP_MEM_HEADER` + the embedded/daisy CMake) is the follow-on,
gated on exercising the embedded toolchain build. Remaining Phase E work: split
the ~22 `#ifdef`-interleaved embedded/host core files into build-selected
per-profile implementation files behind shared interfaces.
