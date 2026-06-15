/* spectral_synth_ifft.c - IFFT synthesis frame builder + OLA renderer.
 *
 * Backend-independent half: the F1-validated construction (see the contract
 * header and IFFT_SYNTHESIS_PLAN.md). The motif table is computed once at
 * create() from the same periodic-Hann window the OLA uses, in double, then
 * stored float — the F1 harness measured interpolation error negligible at
 * O=16 and float storage sits ~25 dB below the K=8 truncation floor.
 */
#include "spectral_synth_ifft.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "spectral_consts.h"
#include "spectral_config.h"          /* SPECTRAL_EMBEDDED / SPECTRAL_ENABLE_APPROX_TRIG */
#include "spectral_osc_formulas.h"    /* spectral_fast_sin_inline — the trig SSOT */

/* F6 host SIMD trig: the per-partial sin/cos is the renderer's largest non-FFT
 * term, and it is scalar+interleaved in the frame loop so the compiler cannot
 * vectorize it. A microbench (bench_ifft_synth) measured a 4-wide minimax sin at
 * ~6.7× the scalar sinf, so the frame loop batches the trig into a SIMD Phase A.
 * Host only (SIMDe → SSE2/NEON) AND only when the poly is the active trig; exact
 * (APPROX_TRIG=0) and embedded builds take the scalar SSOT path. */
#if !SPECTRAL_EMBEDDED && SPECTRAL_ENABLE_APPROX_TRIG
#  include "simde/x86/sse2.h"
#  include "simde/x86/sse4.1.h"        /* floor_ps */
#  define SPECTRAL_IFFT_TRIG_SIMD 1
/* 4-wide minimax sin, SSOT coefficients (mirrors osc_vfastsin: q=round(x/pi) fold). */
static inline simde__m128 ifft_fast_sin4(simde__m128 x) {
    const simde__m128 inv_pi = simde_mm_set1_ps(SPECTRAL_INV_PI);
    const simde__m128 pi     = simde_mm_set1_ps(SPECTRAL_PI);
    const simde__m128 half   = simde_mm_set1_ps(0.5f);
    const simde__m128 two    = simde_mm_set1_ps(2.0f);
    const simde__m128 one    = simde_mm_set1_ps(1.0f);
    simde__m128 q  = simde_mm_floor_ps(simde_mm_add_ps(simde_mm_mul_ps(x, inv_pi), half));
    simde__m128 xr = simde_mm_sub_ps(x, simde_mm_mul_ps(q, pi));
    simde__m128 x2 = simde_mm_mul_ps(xr, xr);
    simde__m128 p  = simde_mm_set1_ps(SPECTRAL_MINIMAX_SIN_C9);
    p = simde_mm_add_ps(simde_mm_set1_ps(SPECTRAL_MINIMAX_SIN_C7), simde_mm_mul_ps(x2, p));
    p = simde_mm_add_ps(simde_mm_set1_ps(SPECTRAL_MINIMAX_SIN_C5), simde_mm_mul_ps(x2, p));
    p = simde_mm_add_ps(simde_mm_set1_ps(SPECTRAL_MINIMAX_SIN_C3), simde_mm_mul_ps(x2, p));
    p = simde_mm_mul_ps(xr, simde_mm_add_ps(one, simde_mm_mul_ps(x2, p)));
    simde__m128 frac = simde_mm_sub_ps(q, simde_mm_mul_ps(two, simde_mm_floor_ps(simde_mm_mul_ps(q, half))));
    simde__m128 sign = simde_mm_sub_ps(one, simde_mm_mul_ps(two, frac));
    return simde_mm_mul_ps(p, sign);
}
#else
#  define SPECTRAL_IFFT_TRIG_SIMD 0
#endif

/* F1-measured operating point (IFFT_SYNTHESIS_PLAN error budget): K=8 taps
 * per side -> -55 dBFS frame / -83 dBFS stream RMS floor; O=16 oversampling
 * is within 1 dB of O=64. K is an explicit F2/F3 knob: raising it buys
 * ~8 dB per +4 taps at ~2K complex MACs per partial. */
#define IFFT_MOTIF_K 8
#define IFFT_MOTIF_O 16

struct SpectralIfftSynth {
    size_t n_fft;
    size_t hop;
    int owns_pool;    /* create() owns + frees the pool/backend; init() does not */
    SpectralIfftBackend* backend;
    float* motif;     /* 2*K*O+1 samples of the centered window spectrum */
    float* window;    /* periodic Hann, n_fft (kept for tests/inspection) */
    float* re;        /* half-spectrum scratch, n_fft/2 */
    float* im;
    float* frame;     /* time-domain frame scratch, n_fft */
    SpectralIfftPartial* frame_partials;  /* per-frame partial scratch (cap n_fft/2)
                                             for the dynamic render path */
};

static int is_pow2(size_t v) { return v && (v & (v - 1)) == 0; }

/* Pool carving, 16-byte aligned so the re/im/frame SIMD loads stay aligned. */
#define IFFT_POOL_ALIGN 16u
static size_t ifft_align(size_t x) {
    return (x + (IFFT_POOL_ALIGN - 1u)) & ~(size_t)(IFFT_POOL_ALIGN - 1u);
}

size_t spectral_ifft_synth_pool_bytes(size_t n_fft) {
    if (!is_pow2(n_fft) || n_fft < 64) return 0;
    size_t mlen = 2u * IFFT_MOTIF_K * IFFT_MOTIF_O + 1u;
    size_t half = n_fft / 2u;
    return ifft_align(sizeof(struct SpectralIfftSynth))
         + ifft_align(mlen * sizeof(float))                 /* motif */
         + ifft_align(n_fft * sizeof(float))                /* window */
         + ifft_align(half * sizeof(float))                 /* re */
         + ifft_align(half * sizeof(float))                 /* im */
         + ifft_align(n_fft * sizeof(float))                /* frame */
         + ifft_align(half * sizeof(SpectralIfftPartial));  /* frame_partials */
}

/* Carve the struct + scratch pointers from a contiguous pool (no compute). */
static void ifft_synth_place(SpectralIfftSynth* s, char* pool, size_t n_fft) {
    size_t mlen = 2u * IFFT_MOTIF_K * IFFT_MOTIF_O + 1u;
    size_t half = n_fft / 2u;
    char* p = pool + ifft_align(sizeof(*s));
    s->n_fft = n_fft;
    s->hop = half;
    s->motif = (float*)p;   p += ifft_align(mlen * sizeof(float));
    s->window = (float*)p;  p += ifft_align(n_fft * sizeof(float));
    s->re = (float*)p;      p += ifft_align(half * sizeof(float));
    s->im = (float*)p;      p += ifft_align(half * sizeof(float));
    s->frame = (float*)p;   p += ifft_align(n_fft * sizeof(float));
    s->frame_partials = (SpectralIfftPartial*)p;
}

/* Compute the build-constant window + motif tables into the placed scratch. Double
 * cos at INIT only; the embedded F4 port commits this table so the libc-free runtime
 * never calls libm (and the memsets here become spectral_mem_zero). */
static void ifft_synth_compute_tables(SpectralIfftSynth* s) {
    size_t n_fft = s->n_fft;
    size_t mlen = 2u * IFFT_MOTIF_K * IFFT_MOTIF_O + 1u;
    memset(s->re, 0, (n_fft / 2u) * sizeof(float));
    memset(s->im, 0, (n_fft / 2u) * sizeof(float));
    for (size_t n = 0; n < n_fft; n++) {
        s->window[n] = (float)(0.5 - 0.5 * cos(2.0 * SPECTRAL_PI_D * (double)n / (double)n_fft));
    }
    /* M(d) = sum_m v[m] cos(2 pi d m / N), v = window circularly centered at 0.
     * Double accumulation; the table is tiny (257 floats at K=8/O=16). */
    for (size_t i = 0; i < mlen; i++) {
        double d = ((double)i - (double)(IFFT_MOTIF_K * IFFT_MOTIF_O)) / IFFT_MOTIF_O;
        double acc = 0.0;
        for (long m = -(long)n_fft / 2; m < (long)n_fft / 2; m++) {
            double v = 0.5 - 0.5 * cos(2.0 * SPECTRAL_PI_D
                                       * (double)((m + (long)n_fft + (long)n_fft / 2) % (long)n_fft)
                                       / (double)n_fft);
            acc += v * cos(2.0 * SPECTRAL_PI_D * d * (double)m / (double)n_fft);
        }
        s->motif[i] = (float)acc;
    }
}

/* Static-allocation init (the embedded path: no malloc). Places the synth in a
 * caller pool (>= pool_bytes, 16-byte aligned) with a caller-owned backend. The
 * caller owns the pool + backend for the synth's lifetime; destroy() is a no-op. */
SpectralIfftSynth* spectral_ifft_synth_init(void* pool, size_t pool_bytes,
                                            size_t n_fft, SpectralIfftBackend* backend) {
    if (!pool || !backend || !is_pow2(n_fft) || n_fft < 64) return NULL;
    if (pool_bytes < spectral_ifft_synth_pool_bytes(n_fft)) return NULL;
    SpectralIfftSynth* s = (SpectralIfftSynth*)pool;
    ifft_synth_place(s, (char*)pool, n_fft);
    s->backend = backend;
    s->owns_pool = 0;
    ifft_synth_compute_tables(s);
    return s;
}

#if !SPECTRAL_EMBEDDED
SpectralIfftSynth* spectral_ifft_synth_create(size_t n_fft) {
    if (!is_pow2(n_fft) || n_fft < 64) return NULL;
    void* pool = malloc(spectral_ifft_synth_pool_bytes(n_fft));
    if (!pool) return NULL;
    SpectralIfftBackend* backend = spectral_ifft_backend_create(n_fft);
    if (!backend) { free(pool); return NULL; }
    SpectralIfftSynth* s = (SpectralIfftSynth*)pool;
    ifft_synth_place(s, (char*)pool, n_fft);
    s->backend = backend;
    s->owns_pool = 1;
    ifft_synth_compute_tables(s);
    return s;
}
#endif

void spectral_ifft_synth_destroy(SpectralIfftSynth* s) {
    if (!s || !s->owns_pool) return;   /* init() pools are caller-owned (no-op) */
#if !SPECTRAL_EMBEDDED
    spectral_ifft_backend_destroy(s->backend);
    free(s);   /* s == the malloc'd pool start; scratch is carved from it */
#endif
}

size_t spectral_ifft_synth_n_fft(const SpectralIfftSynth* s) {
    return s ? s->n_fft : 0;
}

static inline float motif_eval(const float* t, float d) {
    float x = (d + (float)IFFT_MOTIF_K) * (float)IFFT_MOTIF_O;
    int i0 = (int)x;
    int len = 2 * IFFT_MOTIF_K * IFFT_MOTIF_O + 1;
    if (i0 < 0) i0 = 0;
    if (i0 >= len - 1) i0 = len - 2;
    float f = x - (float)i0;
    return t[i0] * (1.0f - f) + t[i0 + 1] * f;
}

/* Scatter one partial's precomputed (cr, ci) into the packed half-spectrum
 * (Hermitian implicit). The trig that produces cr/ci is batched into the frame
 * loop's Phase A (F6 SIMD) so it can vectorize across partials. */
static void place_partial_cri(SpectralIfftSynth* s, float bin, float cr, float ci) {
    int k0 = (int)floorf(bin);
    int half = (int)(s->n_fft / 2);
    const int n_tap = 2 * IFFT_MOTIF_K;
    const int klo = k0 - IFFT_MOTIF_K + 1;

    /* Interior fast path: the motif index advances by exactly O per tap and the lerp
     * weight f = x - i0 is tap-INDEPENDENT, so hoist the lookup setup out of the loop.
     * Bit-identical to the per-tap motif_eval (i0_j = i0_0 + j*O, same f) with no
     * DC/Nyquist skip (window interior) and no table-edge clamp — the latter needs
     * frac > 0: an exact-integer bin (frac=0, measure-zero) would push the last tap to
     * the table end, so it falls to the per-tap path which clamps the edge correctly. */
    const float frac = bin - (float)k0;
    if (klo >= 1 && klo + n_tap - 1 <= half - 1 && frac > 0.0f) {
        const float* t = s->motif;
        float xf = (1.0f - frac) * (float)IFFT_MOTIF_O;   /* x at the first tap */
        int i0 = (int)xf;
        float f = xf - (float)i0;
        float w0 = 1.0f - f;
        for (int j = 0; j < n_tap; j++) {
            int k = klo + j;
            float m = t[i0] * w0 + t[i0 + 1] * f;
            float sgn = (k & 1) ? -1.0f : 1.0f;
            s->re[k] += sgn * m * cr;
            s->im[k] += sgn * m * ci;
            i0 += IFFT_MOTIF_O;
        }
        return;
    }

    /* Boundary: per-tap motif_eval with the DC/Nyquist skip. */
    for (int tap = -IFFT_MOTIF_K + 1; tap <= IFFT_MOTIF_K; tap++) {
        int k = k0 + tap;
        if (k <= 0 || k >= half) continue;     /* DC/Nyquist kept clear */
        float m = motif_eval(s->motif, (float)k - bin);
        float sgn = (k & 1) ? -1.0f : 1.0f;    /* (-1)^k centering twiddle */
        s->re[k] += sgn * m * cr;
        s->im[k] += sgn * m * ci;
    }
}

/* Domain check for one partial: the header's bin in (1, n_fft/2 - 1) plus finite
 * amp/phase. place_partial's float->int floor is int-conversion UB on a garbage
 * bin, so every partial is checked before it reaches the spectrum; the negated
 * comparisons also reject NaN. */
static int ifft_partial_valid(const SpectralIfftSynth* s, const SpectralIfftPartial* p) {
    const size_t half = s->n_fft / 2;
    const float bin_max = (float)half - 1.0f;
    return (p->bin > 1.0f) && (p->bin < bin_max) &&
           isfinite(p->amp) && isfinite(p->phase0);
}

/* Build frame m (center = m*hop + n_fft/2) from pre-validated partials, each
 * placed at its frame-center phase; inverse-FFT and overlap-add into out. The
 * single home for the framing/OLA/motif math, shared by the static and dynamic
 * render paths so they cannot drift. */
static void ifft_render_frame(SpectralIfftSynth* s,
                              const SpectralIfftPartial* partials, size_t n,
                              long m, float* out, size_t total) {
    double center = (double)m * (double)s->hop + (double)s->n_fft / 2.0;

    /* Full half-spectrum clear per frame, though only the K-bin neighborhood of
     * each partial gets written: at N=512 this is 2 KB per frame against ~2K-tap
     * placement work per partial, and the path still measured 7.5x over the
     * oscillator loop — not the bottleneck. Revisit with a dirty-region clear in
     * the ARM port, where the memory system is the constraint. */
    memset(s->re, 0, (s->n_fft / 2) * sizeof(float));
    memset(s->im, 0, (s->n_fft / 2) * sizeof(float));

    /* Phase A: (cr, ci) for every partial. The phase reduction stays scalar+double
     * (phi can be thousands of rad → float would lose ~1e-3), but the sin/cos is
     * batched into a SIMD minimax pass (F6). Phase B scatters. Blocked so the
     * per-block scratch is bounded regardless of n. cos(x)=sin(x+pi/2). */
    enum { IFFT_TRIG_BLK = 64 };
    float phi[IFFT_TRIG_BLK], cr[IFFT_TRIG_BLK], ci[IFFT_TRIG_BLK];
    for (size_t b0 = 0; b0 < n; b0 += IFFT_TRIG_BLK) {
        size_t bn = (n - b0 < IFFT_TRIG_BLK) ? (n - b0) : (size_t)IFFT_TRIG_BLK;
        const double tau = 2.0 * SPECTRAL_PI_D;
        for (size_t j = 0; j < bn; j++) {
            double omega = tau * (double)partials[b0 + j].bin / (double)s->n_fft;
            double x = omega * center + (double)partials[b0 + j].phase0;
            /* Reduce to [-pi, pi] by round-subtract (libm-free, and tighter than the
             * old fmod's [0, 2pi) so the minimax sin stays in its 1.4-ULP range).
             * |x/tau| < ~4000 here, so the long long round is exact. */
            double q = (double)(long long)(x / tau + (x >= 0.0 ? 0.5 : -0.5));
            phi[j] = (float)(x - tau * q);
        }
        size_t j = 0;
#if SPECTRAL_IFFT_TRIG_SIMD
        const simde__m128 hpi = simde_mm_set1_ps(SPECTRAL_HALF_PI);
        for (; j + 4 <= bn; j += 4) {
            simde__m128 ph = simde_mm_loadu_ps(&phi[j]);
            simde__m128 ha = simde_mm_setr_ps(0.5f * partials[b0 + j].amp,
                                              0.5f * partials[b0 + j + 1].amp,
                                              0.5f * partials[b0 + j + 2].amp,
                                              0.5f * partials[b0 + j + 3].amp);
            simde_mm_storeu_ps(&cr[j], simde_mm_mul_ps(ha, ifft_fast_sin4(simde_mm_add_ps(ph, hpi))));
            simde_mm_storeu_ps(&ci[j], simde_mm_mul_ps(ha, ifft_fast_sin4(ph)));
        }
#endif
        for (; j < bn; j++) {   /* SIMD tail + the scalar/exact/embedded path */
            float ha = 0.5f * partials[b0 + j].amp;
            cr[j] = ha * spectral_fast_sin_inline(phi[j] + SPECTRAL_HALF_PI);
            ci[j] = ha * spectral_fast_sin_inline(phi[j]);
        }
        for (j = 0; j < bn; j++) {
            place_partial_cri(s, partials[b0 + j].bin, cr[j], ci[j]);
        }
    }

    spectral_ifft_backend_inverse(s->backend, s->re, s->im, s->frame);

    long base = m * (long)s->hop;
    size_t n0 = (base < 0) ? (size_t)(-base) : 0u;
    for (size_t i = n0; i < s->n_fft; i++) {
        size_t t = (size_t)(base + (long)i);
        if (t >= total) break;
        out[t] += s->frame[i];
    }
}

int spectral_ifft_synth_render(SpectralIfftSynth* s,
                               const SpectralIfftPartial* partials, size_t n,
                               float* out, size_t total) {
    if (!s || (!partials && n > 0) || !out || total == 0) return 1;

    /* Validate the whole (stationary) set at the boundary, ONCE — the static
     * contract rejects any out-of-domain partial outright. */
    for (size_t p = 0; p < n; p++) {
        if (!ifft_partial_valid(s, &partials[p])) return 1;
    }
    memset(out, 0, total * sizeof(float));

    /* Frames at m*hop for m = -1.. so every output sample has full COLA
     * coverage (frame m covers [m*hop, m*hop + n_fft)). */
    long m_last = (long)((total - 1) / s->hop);
    for (long m = -1; m <= m_last; m++) {
        ifft_render_frame(s, partials, n, m, out, total);
    }
    return 0;
}

int spectral_ifft_synth_render_dynamic(SpectralIfftSynth* s,
                                       SpectralIfftFramePartialsFn fn, void* ctx,
                                       float* out, size_t total) {
    if (!s || !fn || !out || total == 0) return 1;

    /* Per-frame-activity path for time-localized sources (the F2b hybrid
     * router): `fn` supplies the partials live in each frame — the engine maps
     * the active stationary-sine segment interiors to partials at frame center.
     * The framing/OLA/motif stay renderer-owned (ifft_render_frame); only WHICH
     * partials are present changes per frame. Partials a frame returns that fall
     * outside the domain are skipped here (UB-safety), so a caller bug degrades
     * to a dropped partial, never an out-of-bounds spectrum write. */
    const size_t cap = s->n_fft / 2;
    memset(out, 0, total * sizeof(float));
    long m_last = (long)((total - 1) / s->hop);
    for (long m = -1; m <= m_last; m++) {
        double center = (double)m * (double)s->hop + (double)s->n_fft / 2.0;
        size_t got = fn(ctx, center, s->frame_partials, cap);
        if (got > cap) got = cap;
        size_t valid = 0;
        for (size_t p = 0; p < got; p++) {
            if (ifft_partial_valid(s, &s->frame_partials[p])) {
                if (valid != p) s->frame_partials[valid] = s->frame_partials[p];
                valid++;
            }
        }
        ifft_render_frame(s, s->frame_partials, valid, m, out, total);
    }
    return 0;
}
