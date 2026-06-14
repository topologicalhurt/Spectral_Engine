#ifndef SPECTRAL_ENVELOPE_H
#define SPECTRAL_ENVELOPE_H

#include <stddef.h>
#include "spectral_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fade envelope parameters for segment synthesis */
typedef struct {
    size_t fade_len;
    size_t fade_out_start;
    float inv_fade;
} FadeParams;

/* The span-COMPUTATION of the three-region (fade-in / sustain / fade-out over
 * [0,len)) geometry is factored HERE and nowhere else: fade_params_init() is the
 * single source of fade_len / fade_out_start / inv_fade, and fade_envelope() is
 * the single source of the per-sample region branch. Every backend reuses these.
 *
 * The three-region WALK is, by contrast, intentionally NOT factored into a shared
 * driver (review item D3/D5 — accepted duplication). The four inner loops are
 * irreconcilable rather than boilerplate: the scalar float walk recomputes phase
 * from j; the scalar-Q15 and host pack8-SIMD walks each carry a STATEFUL phase
 * generator (integer NCO / vectorized SpectralPhaseNco8) that must be stepped once
 * per sample in sample order, so their spans share a single running cursor; the
 * CMSIS path has no three-span walk at all (one [0,len) loop over fade_envelope()).
 * A generic driver would have to thread that stateful generator through a callback,
 * which would deoptimize the hot sustain loop and move the embedded codegen the m7
 * perf gate pins (process_insns). So only the spans (this struct) are shared; the
 * walk stays inline per backend. */
FadeParams fade_params_init(size_t segment_len, size_t max_fade);

float fade_envelope_in(size_t j, float inv_fade);
float fade_envelope_out(size_t j, size_t len, float inv_fade);
float fade_envelope(size_t j, const FadeParams* fp, size_t len);

#ifdef __cplusplus
}
#endif

#endif
