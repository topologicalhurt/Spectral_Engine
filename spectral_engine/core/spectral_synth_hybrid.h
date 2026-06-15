/* spectral_synth_hybrid.h - hybrid IFFT/oscillator synthesis fast path.
 *
 * F-stream F2b (IFFT_SYNTHESIS_PLAN.md). A density router that renders a dense
 * set of stationary sine partials via the IFFT path (spectral_synth_ifft.h)
 * instead of the per-partial oscillator loop — measured ~8x at 64 partials.
 *
 * This is an OPT-IN FAST PATH, not a replacement: try_render() either renders
 * the whole buffer via the IFFT (returns RENDERED) or DECLINES untouched, and
 * the caller falls back to synth_cpu(). The default render path is byte-for-byte
 * unchanged — nothing here is wired into the engine dispatch until the golden
 * sign-off (F3). The IFFT is an approximation within MEASURED, gated floors.
 *
 * v1 scope (the all-eligible dense case): the fast path engages only when EVERY
 * segment is an in-domain stationary sine partial long enough that the IFFT's
 * COLA coverage ramp is a negligible fraction of its life (so that ramp serves
 * as the segment fade), the timbre is sine, and stretch/pitch are identity.
 * Anything else DECLINES. Mixed interior/edge routing is the F2b Step-2 sequel.
 */
#ifndef SPECTRAL_SYNTH_HYBRID_H
#define SPECTRAL_SYNTH_HYBRID_H

#include <stddef.h>

#include "spectral_common.h"
#include "spectral_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SPECTRAL_HYBRID_RENDERED = 0,   /* out[] filled via the IFFT fast path */
    SPECTRAL_HYBRID_DECLINED = 1    /* not applicable; caller should use synth_cpu */
} SpectralHybridResult;

/* True if `seg` is a stationary sine partial the IFFT can place: no chirp
 * (|df|~0), no cubic phase, ~constant amplitude over its life, fractional bin
 * (omega*n_fft/2pi) inside the renderer domain (1, n_fft/2-1), and long enough
 * (>= SPECTRAL_HYBRID_MIN_FRAMES frames) that the COLA fade is negligible.
 * The sine-timbre and stretch/pitch-identity gates are global (try_render). */
int spectral_synth_hybrid_segment_eligible(const Segment* seg, size_t n_fft);

/* Attempt the IFFT fast path. Returns SPECTRAL_HYBRID_RENDERED with the whole
 * `out` buffer (out_len samples) filled, or SPECTRAL_HYBRID_DECLINED with `out`
 * UNTOUCHED — in which case the caller must render via synth_cpu(). Declines on
 * non-sine timbre, non-identity stretch/pitch, an empty/oversized set, too few
 * partials to beat the oscillator loop, or any ineligible segment. */
SpectralHybridResult spectral_synth_hybrid_try_render(
    SegmentArray sa, float* out, size_t out_len,
    float stretch, float pitch, SpectralTimbre timbre);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_SYNTH_HYBRID_H */
