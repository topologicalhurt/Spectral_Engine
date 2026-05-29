# Patch notes — Pass 148: build-select desktop-only core modules (Phase E)

## Change

`segment_mt.c` (pthread thread-safe segment array) and `segment_pool.c` (analysis
segment pooling) were whole-file `#if !SPECTRAL_EMBEDDED` — desktop/analysis-only,
compiled empty on the sim/embedded host targets. Moved them to a build-selected
source set `SPECTRAL_SOURCES_CORE_DESKTOP` (desktop target only) and dropped the
`.c` guards, so the algorithm bodies no longer branch on `SPECTRAL_EMBEDDED`. The
headers keep their `#if !SPECTRAL_EMBEDDED` guard — it gates `<pthread.h>` and the
`pthread_mutex_t` member (declaration-level, and bare-metal protection), not an
algorithm body.

## Verification

All six green targets build (desktop, simulate, embedded_arm, embedded_arm_float,
simulate_daisy, arm_core_test); sim oracle green. The embedded/sim targets no
longer compile these modules — they were compiled empty there before (no symbol
references, per the baseline), so behavior is unchanged.

`embedded_arm_restricted` still fails at link on a PRE-EXISTING bug (`analyze_audio`
/ `perf_*` undefined — its source set omits ANALYSIS+MONITORING but the shared CLI
pipeline references them); it still compiles to the link stage. Tracked separately.
