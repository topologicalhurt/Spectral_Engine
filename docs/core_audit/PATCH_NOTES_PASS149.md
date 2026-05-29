# Patch notes — Pass 149: remove redundant host/sim whole-file guards (Phase E)

`spectral_seg_cache.c`, `spectral_seg_cache_fs.c`/`.h`, and
`spectral_segment_parser.c`/`.h` were each wrapped entirely in
`#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM`. That guard is true on every
target that compiles them (desktop: `!EMBEDDED`; sim/embedded host builds:
`IS_EMBEDDED_SIM`), and they are excluded from the real-embedded (daisy) source
set (`DAISY_ENGINE`). So the guard was redundant — build-selection (`CORE`
membership vs `DAISY_ENGINE`) already gates them. Removed the guards; the bodies
no longer branch on `SPECTRAL_EMBEDDED`.

Verified: the six green targets build (desktop, simulate, embedded_arm,
embedded_arm_float, simulate_daisy, arm_core_test); sim oracle green.
`embedded_arm_restricted` = the pre-existing link bug, unchanged.
