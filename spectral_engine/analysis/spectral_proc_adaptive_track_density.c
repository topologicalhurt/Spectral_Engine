/* spectral_proc_adaptive_track_density.c - MQ track linkage + cubic phase stage
 *
 * McAulay & Quatieri (1986), "Speech Analysis/Synthesis Based on a Sinusoidal
 * Representation," IEEE ASSP-34(4).  This optional stage reconstructs partial
 * tracks across hops (peak continuation) and annotates each predecessor segment
 * with cubic-phase coefficients (MQ eq. 33-38) so the synth path renders a
 * C1-continuous phase across the hop boundary instead of independent grains.
 *
 * Double-gated: it only does work under SPECTRAL_PRECISE_PHASE (compile flag,
 * default 0) AND when selected via the adaptive_track_density processing mask.
 * With the flag off the stage is a no-op and the synth path is bit-identical to
 * the legacy per-grain quadratic model. */
#include "spectral_proc_adaptive_track_density.h"

#if SPECTRAL_PRECISE_PHASE
#include "spectral_common.h"
#include <limits.h>
#include <math.h>

/* Frequency-matching tolerance for linking a partial across one hop, expressed
 * in FFT bins (an MQ peak-continuation matching interval).  Conservative: ~1
 * bin keeps spurious cross-links rare while admitting normal partial drift. */
#ifndef SPECTRAL_TRACK_LINK_BINS
#define SPECTRAL_TRACK_LINK_BINS 1.0f
#endif

static const float ATD_TWO_PI = 6.28318530717958647692f;

/* Frame index of a segment.  start == frame*hop and length == hop (constant),
 * so start/length recovers the integer frame.  Returns -1 if length invalid. */
static long atd_frame_index(const Segment* s) {
    if (!(s->length > 0.0f)) return -1;
    return lroundf(s->start / s->length);
}

/* Annotate predecessors in [pa,pb) with MQ cubic phase coefficients computed
 * against their nearest-frequency successor in [ca,cb).  Coefficients are in
 * the analysis domain (rad/sample^n); the synth applies pitch/stretch scaling. */
static void atd_link_frames(Segment* segs,
                            uint32_t pa, uint32_t pb,
                            uint32_t ca, uint32_t cb,
                            float tol_omega)
{
    for (uint32_t p = pa; p < pb; p++) {
        Segment* sp = &segs[p];
        const float w_k = sp->omega;
        const float T = sp->length;
        if (!(T > 0.0f)) continue;

        const float tol = (tol_omega > 0.0f)
                            ? tol_omega
                            : (0.05f * fabsf(w_k) + 1e-6f);

        /* Nearest-frequency successor within tolerance (track continuation). */
        uint32_t best = UINT32_MAX;
        float best_d = tol;
        for (uint32_t c = ca; c < cb; c++) {
            const float d = fabsf(segs[c].omega - w_k);
            if (d < best_d) { best_d = d; best = c; }
        }
        if (best == UINT32_MAX) continue; /* track death: keep quadratic model */

        const Segment* sc = &segs[best];
        const float theta_k  = sp->phase;
        const float theta_k1 = sc->phase;
        const float w_k1     = sc->omega;

        /* MQ maximally-smooth unwrap factor M* (eq. 38). */
        const float x = ((theta_k + w_k * T - theta_k1) + 0.5f * T * (w_k1 - w_k))
                        / ATD_TWO_PI;
        const float M = roundf(x);

        const float dtheta = theta_k1 + ATD_TWO_PI * M - theta_k - w_k * T;
        const float dw = w_k1 - w_k;
        const float invT = 1.0f / T;

        /* theta(t) = theta_k + w_k*t + a2*t^2 + a3*t^3 (MQ eq. 36-37). */
        const float a2 = (3.0f * dtheta * invT - dw) * invT;
        const float a3 = (-2.0f * dtheta * invT + dw) * (invT * invT);

        if (!isfinite(a2) || !isfinite(a3)) continue;
        spectral_segment_set_cubic(sp, a2, a3);
    }
}
#endif /* SPECTRAL_PRECISE_PHASE */

SpectralError spectral_proc_adaptive_track_density_apply(
    SegmentArray* sa,
    int sample_rate,
    const SpectralProcessParams* params)
{
#if SPECTRAL_PRECISE_PHASE
    (void)sample_rate;
    if (!sa) return SPECTRAL_ERR_PARAM;
    if (!sa->segs || sa->count < 2u) return SPECTRAL_OK;

    /* One MQ matching interval ~ bin spacing (2*pi / n_fft rad/sample). */
    float tol_omega = 0.0f;
    if (params && params->n_fft > 0) {
        tol_omega = SPECTRAL_TRACK_LINK_BINS * (ATD_TWO_PI / (float)params->n_fft);
    }

    Segment* segs = sa->segs;
    const uint32_t n = sa->count;

    /* Walk maximal runs of equal frame index (segments are globally
     * frame-ordered by the tracker finalize step).  Link consecutive frames. */
    uint32_t pa = 0u, pb = 0u;
    long prev_frame = LONG_MIN;
    uint32_t i = 0u;
    while (i < n) {
        const long f = atd_frame_index(&segs[i]);
        const uint32_t run_start = i;
        while (i < n && atd_frame_index(&segs[i]) == f) i++;
        const uint32_t run_end = i;

        if (f >= 0 && prev_frame != LONG_MIN && f == prev_frame + 1) {
            atd_link_frames(segs, pa, pb, run_start, run_end, tol_omega);
        }
        pa = run_start;
        pb = run_end;
        prev_frame = f;
    }
    return SPECTRAL_OK;
#else
    (void)sa;
    (void)sample_rate;
    (void)params;
    return SPECTRAL_OK;
#endif
}
