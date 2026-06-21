/* spectral_kernel.h - Spectral Engine public kernel API (v0.0.1 — PRE-RELEASE, surface-guarded)
 *
 * THE public API surface of the Spectral Engine: analyze audio into sinusoidal segments, and
 * resynthesize them (desktop float — CPU/Metal/CUDA/vDSP; Cortex-M7 Q15). Include THIS one header
 * to get the public API; everything not reachable from here is internal.
 *
 * STABILITY: this is 0.0.1 — PRE-1.0 and NOT frozen. The surface is expected to still change (the
 * S5 maturity audit lists the gaps to 1.0: a real library target, an analyze_audio error channel,
 * symbol namespacing). What the version + `tests/core_contracts/test_kernel_api_freeze.c` DO buy at
 * 0.0.1: the guard pins the CURRENT surface (version, struct layouts, enum values, curated
 * signatures) so every API change is DELIBERATE — you bump the version + update the guard when you
 * change the surface. It becomes a hard FREEZE (breaking change ⇒ MAJOR bump) at 1.0.
 *
 * The 0.0.1 surface:
 *   - scene model: Segment / SegmentArray / SegmentActive / SpectralHalf (spectral_common.h)
 *   - errors: SpectralError + domains + message/ok helpers (spectral_error.h)
 *   - analysis: analyze_audio() — audio -> SegmentArray (spectral_analysis.h)
 *   - synthesis: synth_cpu / synth_cpu_wavetable + the backend dispatch
 *     (spectral_synth.h, spectral_backend.h)
 *   - audio + wavetable I/O (spectral_io.h, spectral_wavetable.h)
 *   - the renderer registry: renderer x domain x device (spectral_renderer.h; AI_CANON #31)
 *
 * NOT in the guarded surface: the opt-in frequency-domain fast paths (spectral_synth_ifft.h,
 * spectral_synth_hybrid.h, spectral_synth_hybrid_wavetable.h) — F3-gated, no production caller; the
 * analysis internals; the embedded arm32 kernels; perf instrumentation. Include those directly.
 */
#ifndef SPECTRAL_KERNEL_H
#define SPECTRAL_KERNEL_H

#define SPECTRAL_KERNEL_VERSION_MAJOR 0
#define SPECTRAL_KERNEL_VERSION_MINOR 0
#define SPECTRAL_KERNEL_VERSION_PATCH 1
#define SPECTRAL_KERNEL_VERSION_STRING "0.0.1"

/* True if the compiled-against kernel API is at least (maj.min). */
#define SPECTRAL_KERNEL_VERSION_AT_LEAST(maj, min)                          \
    ((SPECTRAL_KERNEL_VERSION_MAJOR > (maj)) ||                             \
     (SPECTRAL_KERNEL_VERSION_MAJOR == (maj) &&                            \
      SPECTRAL_KERNEL_VERSION_MINOR >= (min)))

/* The frozen 1.0 public surface. */
#include "spectral_common.h"      /* Segment, SegmentArray, SpectralHalf — the scene model */
#include "spectral_error.h"       /* SpectralError, error domains + message/ok helpers */
#include "spectral_analysis.h"    /* analyze_audio: audio -> SegmentArray */
#include "spectral_synth.h"       /* synth_cpu / synth_cpu_wavetable */
#include "spectral_backend.h"     /* spectral_synth_dispatch[_ex]: backend selection + fallback */
#include "spectral_renderer.h"    /* renderer registry: renderer x domain x device (AI_CANON #31) */
#include "spectral_io.h"          /* audio read/write, normalize, mono<->stereo */
#include "spectral_wavetable.h"   /* SpectralWavetableBank: load/get wavetables */

#ifdef __cplusplus
extern "C" {
#endif

/* The component-contract versions the 1.0 kernel aggregates (each tracks a specific internal
 * contract, orthogonal to the public-API semver above): SPECTRAL_SYNTH_API_VERSION_* (synth ABI),
 * SPECTRAL_OSC_FORMULAS_VERSION (waveform formulas), SPECTRAL_WAVETABLE_VERSION (table format),
 * SPECTRAL_SEGMENT_MATH_VERSION (segment math). They are declared in their own headers. */

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_KERNEL_H */
