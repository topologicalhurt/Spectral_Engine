/* spectral_oscillator.h - oscillator kernel API: the canonical timbre -> L0-waveform dispatch (L1) */
#ifndef SPECTRAL_OSCILLATOR_H
#define SPECTRAL_OSCILLATOR_H

#include "spectral_oscillator_dispatch.h"
#include "spectral_osc_formulas.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define OSC_HOT __attribute__((hot))
#else
#define OSC_HOT
#endif

/* SegmentLoopParams defined in spectral_synth_internal.h */
struct SegmentLoopParams;

/* THE canonical timbre -> L0-waveform map. Single source of truth, expanded in
 * two places that must never drift: the spectral_osc_eval() switch below
 * (single-sample + CUDA device path) and the host-side osc_waveform_fn() pointer
 * selector in spectral_oscillator.c (hot per-segment scalar loop). One list, two
 * expansions, zero divergence. */
#define SPECTRAL_OSC_TIMBRE_LIST(X)        \
    X(TIMBRE_SINE,      spectral_osc_sine)      \
    X(TIMBRE_SAW,       spectral_osc_saw)       \
    X(TIMBRE_SQUARE,    spectral_osc_square)    \
    X(TIMBRE_TRIANGLE,  spectral_osc_triangle)  \
    X(TIMBRE_ASIN,      spectral_osc_asin)      \
    X(TIMBRE_PARABOLA,  spectral_osc_parabola)  \
    X(TIMBRE_QUANTIZED, spectral_osc_quantized) \
    X(TIMBRE_PWM,       spectral_osc_pwm)

/* L1 per-sample oscillator kernel — THE single dispatch point shared by the
 * single-sample timbre_oscillator(), the CUDA tile kernel, and (mirrored) Metal.
 * Normalizes the phase once via the canonical L0 formula, then selects the L0
 * waveform.  Same float op sequence the former per-backend switches ran, so the
 * CUDA path is bit-faithful and the host single-sample API is byte-identical.
 * Compiles as `static inline` in C and `__device__ __forceinline__` under nvcc
 * via OSC_FORMULA_FUNC.
 *
 * NOTE: the hot host *segment* loop does NOT call this per sample — it hoists
 * the timbre dispatch out of the loop via osc_waveform_fn() (spectral_oscillator.c), so
 * the inner loop is a tight call to one fixed L0 waveform.  Calling this switch
 * per sample measured ~1.6x slower on the scalar/asin paths (the loop-invariant
 * dispatch was not unswitched).  Both forms are byte-identical; this one is the
 * portable single source, that one is its host-hoisted specialization. */
OSC_FORMULA_FUNC float spectral_osc_eval(float phase, int timbre, float width) {
    float rads = spectral_normalize_phase(phase);
    switch (timbre) {
#define SPECTRAL_OSC_EVAL_CASE(T, FN) case T: return FN(rads, width);
        SPECTRAL_OSC_TIMBRE_LIST(SPECTRAL_OSC_EVAL_CASE)
#undef SPECTRAL_OSC_EVAL_CASE
        default: return spectral_osc_sine(rads, width);
    }
}

OSC_HOT float timbre_oscillator(float p, float a, SpectralTimbre timbre, float width);

/* Segment synthesis with backend dispatch */
OSC_HOT void timbre_synth_segment(float* __restrict__ dst, const struct SegmentLoopParams* lp, SpectralTimbre timbre);

void spectral_osc_set_dispatch(OscDispatchWord dispatch);

/* Per-timbre opt-in Q15 *compute* domain (QTYPE_DOMAIN_PLAN.md §5).
 * Bit (1u << timbre) set => that timbre's point-sampled sustain renders via the
 * Q15 evaluators (spectral_osc_q15.h) instead of float.  Float is the DEFAULT
 * (mask 0); Q15 is lossy (~-85..-92 dBFS vs float; floors pinned by
 * test_q15_compute_precision) and engaged only
 * when explicitly enabled.  Only sine/saw/square/triangle/parabola have a Q15
 * path (spectral_osc_q15_available); the bit is ignored for any other timbre. */
#define OSC_Q15_BIT(timbre) ((uint16_t)(1u << (timbre)))
void spectral_osc_set_q15_enable(uint16_t mask);
int spectral_osc_q15_available(SpectralTimbre timbre);

/* CUDA oscillator — thin wrapper over the shared L1 kernel above. */
#ifdef __CUDACC__

__device__ __forceinline__ float oscillator_cuda(float phase, int timbre) {
    return spectral_osc_eval(phase, timbre, 0.0f);
}

#endif

#ifdef __cplusplus
}
#endif

#endif
