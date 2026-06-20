/* spectral_synth_hybrid_wavetable.c - opt-in wavetable IFFT fast path (F-stream / renderer 3a-3). */
#include "spectral_synth_hybrid_wavetable.h"

#include <math.h>

#include "spectral_consts.h"
#include "spectral_synth_ifft.h"
#include "spectral_wavetable_harmonics.h"

/* Same operating point as the sine hybrid: 512-sample IFFT frame, hop n_fft/2, the MIN_PARTIALS
 * crossover below which the per-partial table read is cheaper. MAX_HARMONICS bounds the harmonic
 * stack extracted per table (and so the amp_k/phase_k scratch). */
#define WT_HYBRID_NFFT          512u
#define WT_HYBRID_MIN_PARTIALS  8u
#define WT_HYBRID_MAX_HARMONICS 32u

typedef struct {
    const Segment* segs;
    size_t         count;
    size_t         n_fft;
    const float*   amp_k;
    const float*   phase_k;
    size_t         n_harm;
} WavetableHybridSource;

/* Per-frame partial provider: for every segment active in the frame, deposit its wavetable
 * harmonics. The fundamental's onset is FLOORED to match the table-read fallback (which renders
 * from start_idx = (size_t)start), so the t=0 fundamental phase is (phase - omega*floor(start)),
 * wrapped in double. The table series is in COSINES (spectral_wavetable_harmonics convention), and
 * the IFFT deposits cosines, so — unlike the sine hybrid — there is no -pi/2 conversion: harmonic
 * k of fundamental phase p is cos(k*omega*t + k*p + phase_k[k]), exactly what spectral_wavetable_expand
 * places (bin k*f0, amp*amp_k[k], phase k*p + phase_k[k]). */
static size_t wt_hybrid_frame_partials(void* vctx, double center,
                                       SpectralIfftPartial* out, size_t cap) {
    const WavetableHybridSource* h = (const WavetableHybridSource*)vctx;
    size_t k = 0;
    for (size_t i = 0; i < h->count && k < cap; i++) {
        const Segment* s = &h->segs[i];
        double start = floor((double)s->start);
        double end = start + (double)s->length;
        double omega, f0_bin, base;
        if (center < start || center > end) continue;          /* not active this frame */
        omega = (double)s->omega;
        f0_bin = omega * (double)h->n_fft / (2.0 * SPECTRAL_PI_D);
        base = (double)s->phase - omega * start;
        base -= (2.0 * SPECTRAL_PI_D) * floor(base / (2.0 * SPECTRAL_PI_D) + 0.5);  /* wrap to [-pi,pi] */
        k += spectral_wavetable_expand(h->amp_k, h->phase_k, h->n_harm,
                                       (float)f0_bin, s->amp, (float)base,
                                       h->n_fft, out + k, cap - k);
    }
    return k;
}

SpectralHybridResult spectral_synth_hybrid_try_render_wavetable(
    SegmentArray sa, const SpectralWavetable* table, float* out, size_t out_len,
    float stretch, float pitch) {

    float amp_k[WT_HYBRID_MAX_HARMONICS + 1];
    float phase_k[WT_HYBRID_MAX_HARMONICS + 1];
    size_t n_harm;
    SpectralIfftSynth* s;
    WavetableHybridSource src;
    int rc;

    if (!out || out_len == 0) return SPECTRAL_HYBRID_DECLINED;
    if (!table || !table->valid) return SPECTRAL_HYBRID_DECLINED;             /* no valid table */
    /* identity transform only (pitch is SEMITONES -> identity 0.0; stretch is a ratio -> 1.0) */
    if (stretch != 1.0f || pitch != 0.0f) return SPECTRAL_HYBRID_DECLINED;
    if (sa.count < WT_HYBRID_MIN_PARTIALS) return SPECTRAL_HYBRID_DECLINED;   /* below crossover */

    /* all-or-decline: every segment must be a stationary, in-domain-fundamental partial (the
     * SAME contract the sine hybrid + the osc fallback enforce). */
    for (uint32_t i = 0; i < sa.count; i++) {
        if (!spectral_synth_hybrid_segment_eligible(&sa.segs[i], WT_HYBRID_NFFT)) {
            return SPECTRAL_HYBRID_DECLINED;
        }
    }

    n_harm = spectral_wavetable_harmonics(table, amp_k, phase_k, WT_HYBRID_MAX_HARMONICS);
    if (n_harm < 2u) return SPECTRAL_HYBRID_DECLINED;                         /* no harmonics */

    /* Guarantee no per-frame truncation: with every segment active (eligibility requires
     * length >= 4 frames) the deposited partial count = sum over segments of the harmonics that
     * fit below the renderer edge. It must not exceed the frame buffer (n_fft/2). */
    {
        const size_t cap = WT_HYBRID_NFFT / 2u;
        const size_t bin_max_i = WT_HYBRID_NFFT / 2u - 1u;
        const double bin_max = (double)bin_max_i;
        size_t total = 0;
        for (uint32_t i = 0; i < sa.count; i++) {
            double f0_bin = (double)sa.segs[i].omega * (double)WT_HYBRID_NFFT / (2.0 * SPECTRAL_PI_D);
            size_t kfit = (f0_bin > 0.0) ? (size_t)floor(bin_max / f0_bin) : 0u;  /* k*f0 < bin_max */
            total += (n_harm - 1u < kfit) ? (n_harm - 1u) : kfit;
        }
        if (total == 0u || total > cap) return SPECTRAL_HYBRID_DECLINED;
    }

    s = spectral_ifft_synth_create(WT_HYBRID_NFFT);
    if (!s) return SPECTRAL_HYBRID_DECLINED;
    src.segs = sa.segs; src.count = sa.count; src.n_fft = WT_HYBRID_NFFT;
    src.amp_k = amp_k; src.phase_k = phase_k; src.n_harm = n_harm;
    rc = spectral_ifft_synth_render_dynamic(s, wt_hybrid_frame_partials, &src, out, out_len);
    spectral_ifft_synth_destroy(s);

    return (rc == 0) ? SPECTRAL_HYBRID_RENDERED : SPECTRAL_HYBRID_DECLINED;
}
