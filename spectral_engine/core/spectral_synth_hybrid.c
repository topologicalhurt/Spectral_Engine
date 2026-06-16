/* spectral_synth_hybrid.c - hybrid IFFT/oscillator synthesis fast path (F2b). */
#include "spectral_synth_hybrid.h"

#include <math.h>

#include "spectral_consts.h"
#include "spectral_contracts.h"      /* spectral_segment_valid_for_synth — the osc-path gate */
#include "spectral_synth_ifft.h"

/* v1 operating point. n_fft is the IFFT frame length (hop = n_fft/2); 512
 * matches the F2 measured point. The COLA coverage of a partial active over
 * [start, end] ramps up/down over ~one frame, so requiring length >= MIN_FRAMES
 * frames keeps that ramp a small fraction of the partial's life and lets it
 * stand in for the segment fade. MIN_PARTIALS is the F1/P6 crossover below which
 * the oscillator loop is cheaper. */
#define HYBRID_NFFT          512u
#define HYBRID_MIN_FRAMES    4u
#define HYBRID_MIN_PARTIALS  8u

/* Stationarity tolerances: a segment whose frequency/amplitude moves less than
 * these over its life renders within the IFFT's frame-center approximation. */
#define HYBRID_DF_EPS        1.0e-9f   /* |df| (rad/sample^2) ~ 0: no chirp */
#define HYBRID_C_EPS         1.0e-9f   /* |c2|,|c3| ~ 0: no cubic phase */
#define HYBRID_DAMP_EPS      1.0e-3f   /* |da*length| (linear) ~ 0: const amp */

int spectral_synth_hybrid_segment_eligible(const Segment* seg, size_t n_fft) {
    if (!seg) return 0;
    /* Gate on the SAME validity contract the oscillator fallback enforces, so an
     * out-of-contract segment (amp<0, start<0, length<=0, omega<0, or any non-finite
     * field) DECLINES to the osc path rather than being IFFT-rendered into a partial
     * that diverges from the only sanctioned reference. Subsumes the per-field finite
     * checks; the hybrid-specific stationarity/domain gates follow. */
    if (!spectral_segment_valid_for_synth(seg)) return 0;
    if (!(fabsf(seg->df) <= HYBRID_DF_EPS)) return 0;             /* no chirp (also rejects NaN) */
    if (spectral_segment_has_cubic(seg) &&
        (fabsf(spectral_segment_cubic_c2(seg)) > HYBRID_C_EPS ||
         fabsf(spectral_segment_cubic_c3(seg)) > HYBRID_C_EPS)) {
        return 0;                                                 /* no cubic phase */
    }
    if (!(fabsf(seg->da * seg->length) <= HYBRID_DAMP_EPS)) return 0;   /* ~constant amplitude */
    if (!(seg->length >= (float)(HYBRID_MIN_FRAMES * n_fft))) return 0; /* COLA fade negligible */

    /* Fractional bin must sit strictly inside the renderer domain (1, n_fft/2-1)
     * so the partial is placed, never silently dropped at DC/Nyquist. */
    size_t half = n_fft / 2u;
    float bin = seg->omega * (float)n_fft / (float)(2.0 * SPECTRAL_PI_D);
    float bin_max = (float)half - 1.0f;
    if (!(bin > 1.0f) || !(bin < bin_max)) return 0;
    return 1;
}

/* Per-frame partial provider: emit every eligible segment active in the frame
 * centered at `center`. The onset is FLOORED to match the oscillator fallback,
 * which renders from start_idx = (size_t)start: the osc phase at output sample t is
 * phase + omega*(t - floor(start)), so the renderer's omega*center + phase0 must use
 * phase0 = phase - omega*floor(start) to agree (a fractional start would otherwise
 * phase-shift the partial by up to ~pi). */
typedef struct { const Segment* segs; size_t count; size_t n_fft; } HybridSource;

static size_t hybrid_frame_partials(void* vctx, double center,
                                    SpectralIfftPartial* out, size_t cap) {
    const HybridSource* h = (const HybridSource*)vctx;
    size_t k = 0;
    for (size_t i = 0; i < h->count && k < cap; i++) {
        const Segment* s = &h->segs[i];
        double start = floor((double)s->start);   /* osc start_idx = (size_t)start */
        double end = start + (double)s->length;
        if (center < start || center > end) continue;            /* not active this frame */
        double omega = (double)s->omega;
        out[k].bin = (float)(omega * (double)h->n_fft / (2.0 * SPECTRAL_PI_D));
        out[k].amp = s->amp;
        /* Wrap the absolute t=0 phase to [-pi, pi] in DOUBLE before the float cast:
         * omega*start grows with a late onset, so a raw float cast of phase - omega*start
         * would lose up to ~1 rad. The renderer reduces mod 2pi anyway, so wrapping here
         * is exact and keeps the float phase0 to ~1e-7 rad regardless of onset. */
        double p0 = (double)s->phase - omega * start;
        p0 -= (2.0 * SPECTRAL_PI_D) * floor(p0 / (2.0 * SPECTRAL_PI_D) + 0.5);
        out[k].phase0 = (float)p0;
        k++;
    }
    return k;
}

SpectralHybridResult spectral_synth_hybrid_try_render(
    SegmentArray sa, float* out, size_t out_len,
    float stretch, float pitch, SpectralTimbre timbre) {

    if (!out || out_len == 0) return SPECTRAL_HYBRID_DECLINED;
    if (timbre != TIMBRE_SINE) return SPECTRAL_HYBRID_DECLINED;   /* motif is one cosine bin */
    if (stretch != 1.0f || pitch != 1.0f) return SPECTRAL_HYBRID_DECLINED;  /* v1: identity only */
    if (sa.count < HYBRID_MIN_PARTIALS) return SPECTRAL_HYBRID_DECLINED;    /* below crossover */
    if (sa.count > HYBRID_NFFT / 2u) return SPECTRAL_HYBRID_DECLINED;       /* would exceed frame cap */

    for (uint32_t i = 0; i < sa.count; i++) {
        if (!spectral_synth_hybrid_segment_eligible(&sa.segs[i], HYBRID_NFFT)) {
            return SPECTRAL_HYBRID_DECLINED;                      /* v1: all-or-decline */
        }
    }

    /* v1 builds the synth per call (a malloc + the ~132k-cos motif/window table). The
     * tables are n_fft-constant, so before F3 wires this into the dispatch the synth
     * must be hoisted to setup and reused — via the static-pool spectral_ifft_synth_init
     * path (caller-owned, no malloc) — rather than rebuilt every render. Acceptable for
     * the opt-in v1 (no production caller); a real perf bug once dense renders stream. */
    SpectralIfftSynth* s = spectral_ifft_synth_create(HYBRID_NFFT);
    if (!s) return SPECTRAL_HYBRID_DECLINED;

    HybridSource src = { sa.segs, sa.count, HYBRID_NFFT };
    int rc = spectral_ifft_synth_render_dynamic(s, hybrid_frame_partials, &src, out, out_len);
    spectral_ifft_synth_destroy(s);

    return (rc == 0) ? SPECTRAL_HYBRID_RENDERED : SPECTRAL_HYBRID_DECLINED;
}
