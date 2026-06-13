/* spectral_osc_bandlimited.h - Anti-aliased ("musical") oscillator quality modes.
 *
 * Spectral resynthesis renders every detected partial as the chosen timbre AT
 * that partial's frequency.  The naive waveforms in spectral_osc_formulas.h are
 * point-sampled, so any harmonic a hard edge or sharp cusp generates above
 * Nyquist folds back as inharmonic aliasing — the "harsh" timbres (square, saw,
 * pwm, quantized, asin).  This module adds three opt-in, band-limited renderers
 * that trade speed for a cleaner spectrum, selectable per run:
 *
 *   POLYBLEP   - PolyBLEP step + polyBLAMP slope correction at the
 *                discontinuities (saw / square / pwm / triangle).
 *   ADDITIVE   - Fourier reconstruction with a Nyquist-limited harmonic count
 *                per partial (saw / square / triangle / parabola).
 *   OVERSAMPLE - render the naive waveform N x, FIR low-pass, decimate.  The
 *                universal fallback; the only mode that tames asin / quantized.
 *
 * The DEFAULT is NAIVE, which never enters this module, so the default build
 * output is bit-identical to the point-sampled path.  These modes affect the
 * CPU float synthesis path only (timbre_synth_segment); the GPU/Q15 backends are
 * unchanged.  This is an audition/quality feature, not a golden contract.
 */
#ifndef SPECTRAL_OSC_BANDLIMITED_H
#define SPECTRAL_OSC_BANDLIMITED_H

#include "spectral_config.h" /* SpectralTimbre */

#ifdef __cplusplus
extern "C" {
#endif

struct SegmentLoopParams;

typedef enum SpectralOscQuality {
    SPECTRAL_OSC_QUALITY_NAIVE      = 0, /* point-sampled (default, bit-identical) */
    SPECTRAL_OSC_QUALITY_POLYBLEP   = 1, /* PolyBLEP / polyBLAMP edge correction   */
    SPECTRAL_OSC_QUALITY_ADDITIVE   = 2, /* Nyquist-capped harmonic sum            */
    SPECTRAL_OSC_QUALITY_OVERSAMPLE = 3  /* N x oversample + FIR decimate          */
} SpectralOscQuality;

/* Process-global oscillator quality, mirroring spectral_osc_set_dispatch() in spectral_oscillator.c.
 * Read once per segment by timbre_synth_segment(); set before synthesis. */
void spectral_osc_set_quality(SpectralOscQuality quality);
SpectralOscQuality spectral_osc_get_quality(void);

/* Human-readable name (for logging/usage). Never NULL. */
const char* spectral_osc_quality_name(SpectralOscQuality quality);

/* Parse "naive|polyblep|additive|oversample" (aliases: blep, add, os) or an
 * integer 0..3.  Returns 1 and writes *out on success, 0 on failure. */
int spectral_osc_quality_parse(const char* text, SpectralOscQuality* out);

/* Band-limited synthesis of one segment.  Accumulates (dst[j] += amp*wave) using
 * the same fade/amplitude envelope as the naive scalar path, so it is a drop-in
 * quality swap.  Returns 1 if it fully rendered the segment, or 0 if this
 * timbre/quality pair is not handled (the caller must fall back to naive).
 * Never call with quality == NAIVE. */
int spectral_osc_bandlimited_synth_segment(float* dst, const struct SegmentLoopParams* lp,
                                  SpectralTimbre timbre, SpectralOscQuality quality);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_OSC_BANDLIMITED_H */
