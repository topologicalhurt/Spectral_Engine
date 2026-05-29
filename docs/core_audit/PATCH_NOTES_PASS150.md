# Patch notes — Pass 150: fs.c — drop dead embedded stub branch + guard (Phase E)

`spectral_fs.c` was `#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM` (real host
file I/O) `#else` (embedded stubs returning `SPECTRAL_ERR_BACKEND_UNAVAIL`)
`#endif`. The `#else` stub branch is **dead**: it compiles only on a real
Cortex-M cross-compile (`EMBEDDED && !SIM`), but the only such target (daisy)
excludes `fs.c` from its source set (`DAISY_ENGINE`); every target that does
compile `fs.c` takes the host branch.

Removed the guard and deleted the dead stub branch — `fs.c` is now an
unconditional host-only translation unit (build-selected via `CORE` membership;
daisy excludes it). Internal `SPECTRAL_FS_HAS_MMAP` / platform capability guards
remain (capability, not device — allowed by the Phase E criteria). If a real
embedded build ever needs an `fs` stub, it belongs in a BSP port file, not a dead
`#else` in core.

Verified: the six green targets build; sim oracle green. `embedded_arm_restricted`
= the pre-existing link bug, unchanged.
