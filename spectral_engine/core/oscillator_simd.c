#include "oscillator_dispatch.h"
#include "oscillator.h"
#include "spectral_config.h"
#include "spectral_synth_internal.h"
#include "spectral_envelope.h"
#include <math.h>
#include <stdlib.h>

static inline float compute_phase(float phase0, float alpha, float beta, size_t j) {
    float jf = (float)j;
    return phase0 + jf * (alpha + beta * jf);
}

static void build_phase_array(float* phases, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    
    for (size_t j = 0; j < len; j++) {
        phases[j] = phase_to_rads(compute_phase(phase0, alpha, beta, j));
    }
}

static void build_amp_envelope(float* amps, const SegmentLoopParams* lp, const FadeParams* fp) {
    const size_t len = lp->length;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;
    
    for (size_t j = 0; j < len; j++) {
        amps[j] = (amp0 + d_amp * (float)j) * fade_envelope(j, fp, len);
    }
}

static void apply_envelope_accumulate(float* dst, const float* wave, const float* amps, size_t len) {
    for (size_t j = 0; j < len; j++) {
        dst[j] += wave[j] * amps[j];
    }
}

typedef struct { float* a; float* b; float* c; size_t len; } TempBuffers;

static TempBuffers alloc_temp(size_t len, int count) {
    TempBuffers t = {0};
    t.len = len;
    t.a = (float*)malloc(len * sizeof(float));
    if (count >= 2) t.b = (float*)malloc(len * sizeof(float));
    if (count >= 3) t.c = (float*)malloc(len * sizeof(float));
    return t;
}

static int temp_valid(const TempBuffers* t, int count) {
    if (!t->a) return 0;
    if (count >= 2 && !t->b) return 0;
    if (count >= 3 && !t->c) return 0;
    return 1;
}

static void free_temp(TempBuffers* t) {
    free(t->a); free(t->b); free(t->c);
}

#if defined(OSC_SIMD_VDSP)

void osc_simd_segment_sine(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;
    
    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
    TempBuffers t = alloc_temp(len, 3);
    if (!temp_valid(&t, 3)) { free_temp(&t); return; }
    
    build_phase_array(t.a, lp);
    for (size_t j = 0; j < len; j++) t.a[j] += PI;
    
    int n = (int)len;
    vvsinf(t.b, t.a, &n);
    
    build_amp_envelope(t.c, lp, &fp);
    vDSP_vma(t.b, 1, t.c, 1, dst, 1, dst, 1, len);
    
    free_temp(&t);
}

void osc_simd_segment_saw(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;
    
    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
    TempBuffers t = alloc_temp(len, 3);
    if (!temp_valid(&t, 3)) { free_temp(&t); return; }
    
    build_phase_array(t.a, lp);
    
    float neg_inv_pi = -SPECTRAL_INV_PI;
    vDSP_vsmul(t.a, 1, &neg_inv_pi, t.b, 1, len);
    
    build_amp_envelope(t.c, lp, &fp);
    apply_envelope_accumulate(dst, t.b, t.c, len);
    
    free_temp(&t);
}

void osc_simd_segment_square(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;
    
    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;
    
    for (size_t j = 0; j < len; j++) {
        float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
        float wave = rads < 0.0f ? -1.0f : 1.0f;
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
}

void osc_simd_segment_triangle(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;
    
    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
    TempBuffers t = alloc_temp(len, 3);
    if (!temp_valid(&t, 3)) { free_temp(&t); return; }
    
    build_phase_array(t.a, lp);
    
    float inv_pi = SPECTRAL_INV_PI, two = 2.0f, neg_one = -1.0f;
    vDSP_vsmul(t.a, 1, &inv_pi, t.b, 1, len);
    vDSP_vabs(t.b, 1, t.b, 1, len);
    vDSP_vsmsa(t.b, 1, &two, &neg_one, t.b, 1, len);
    
    build_amp_envelope(t.c, lp, &fp);
    apply_envelope_accumulate(dst, t.b, t.c, len);
    
    free_temp(&t);
}

void osc_simd_segment_parabola(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;
    
    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
    TempBuffers t = alloc_temp(len, 3);
    if (!temp_valid(&t, 3)) { free_temp(&t); return; }
    
    build_phase_array(t.a, lp);
    
    float inv_pi_sq = SPECTRAL_INV_PI_SQ, neg_one = -1.0f;
    vDSP_vsq(t.a, 1, t.b, 1, len);
    vDSP_vsmsa(t.b, 1, &inv_pi_sq, &neg_one, t.b, 1, len);
    
    build_amp_envelope(t.c, lp, &fp);
    apply_envelope_accumulate(dst, t.b, t.c, len);
    
    free_temp(&t);
}

int osc_simd_available(SpectralTimbre timbre) {
    return (timbre <= TIMBRE_PARABOLA);
}

#elif defined(OSC_SIMD_CMSIS)

void osc_simd_segment_sine(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;
    
    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;
    
    size_t j = 0;
    float32_t phases[4], sines[4], amps_v[4], results[4];
    
    for (; j + 4 <= len; j += 4) {
        for (int k = 0; k < 4; k++) {
            phases[k] = phase_to_rads(compute_phase(phase0, alpha, beta, j + k)) + PI;
            amps_v[k] = (amp0 + d_amp * (float)(j + k)) * fade_envelope(j + k, &fp, len);
        }
        for (int k = 0; k < 4; k++) sines[k] = arm_sin_f32(phases[k]);
        arm_mult_f32(sines, amps_v, results, 4);
        arm_add_f32(dst + j, results, dst + j, 4);
    }
    
    for (; j < len; j++) {
        float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
        float wave = arm_sin_f32(rads + PI);
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
}

void osc_simd_segment_saw(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;
    
    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;
    
    size_t j = 0;
    float32_t rads_v[4], waves[4], amps_v[4], results[4];
    
    for (; j + 4 <= len; j += 4) {
        for (int k = 0; k < 4; k++) {
            rads_v[k] = phase_to_rads(compute_phase(phase0, alpha, beta, j + k));
            amps_v[k] = (amp0 + d_amp * (float)(j + k)) * fade_envelope(j + k, &fp, len);
        }
        arm_scale_f32(rads_v, -SPECTRAL_INV_PI, waves, 4);
        arm_mult_f32(waves, amps_v, results, 4);
        arm_add_f32(dst + j, results, dst + j, 4);
    }
    
    for (; j < len; j++) {
        float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
        float wave = rads * -SPECTRAL_INV_PI;
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
}

void osc_simd_segment_square(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;
    
    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;
    
    for (size_t j = 0; j < len; j++) {
        float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
        float wave = rads < 0.0f ? -1.0f : 1.0f;
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
}

void osc_simd_segment_triangle(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;
    
    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;
    
    size_t j = 0;
    float32_t rads_v[4], abs_v[4], waves[4], amps_v[4], results[4];
    
    for (; j + 4 <= len; j += 4) {
        for (int k = 0; k < 4; k++) {
            rads_v[k] = phase_to_rads(compute_phase(phase0, alpha, beta, j + k)) * SPECTRAL_INV_PI;
            amps_v[k] = (amp0 + d_amp * (float)(j + k)) * fade_envelope(j + k, &fp, len);
        }
        arm_abs_f32(rads_v, abs_v, 4);
        arm_scale_f32(abs_v, 2.0f, waves, 4);
        arm_offset_f32(waves, -1.0f, waves, 4);
        arm_mult_f32(waves, amps_v, results, 4);
        arm_add_f32(dst + j, results, dst + j, 4);
    }
    
    for (; j < len; j++) {
        float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
        float wave = fabsf(rads * SPECTRAL_INV_PI) * 2.0f - 1.0f;
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
}

void osc_simd_segment_parabola(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;
    
    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;
    
    size_t j = 0;
    float32_t rads_v[4], sq_v[4], waves[4], amps_v[4], results[4];
    
    for (; j + 4 <= len; j += 4) {
        for (int k = 0; k < 4; k++) {
            rads_v[k] = phase_to_rads(compute_phase(phase0, alpha, beta, j + k));
            amps_v[k] = (amp0 + d_amp * (float)(j + k)) * fade_envelope(j + k, &fp, len);
        }
        arm_mult_f32(rads_v, rads_v, sq_v, 4);
        arm_scale_f32(sq_v, SPECTRAL_INV_PI_SQ, waves, 4);
        arm_offset_f32(waves, -1.0f, waves, 4);
        arm_mult_f32(waves, amps_v, results, 4);
        arm_add_f32(dst + j, results, dst + j, 4);
    }
    
    for (; j < len; j++) {
        float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
        float wave = rads * rads * SPECTRAL_INV_PI_SQ - 1.0f;
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
}

int osc_simd_available(SpectralTimbre timbre) {
    return (timbre <= TIMBRE_PARABOLA && timbre != TIMBRE_SQUARE);
}

#else

void osc_simd_segment_sine(float* dst, const SegmentLoopParams* lp) { (void)dst; (void)lp; }
void osc_simd_segment_saw(float* dst, const SegmentLoopParams* lp) { (void)dst; (void)lp; }
void osc_simd_segment_square(float* dst, const SegmentLoopParams* lp) { (void)dst; (void)lp; }
void osc_simd_segment_triangle(float* dst, const SegmentLoopParams* lp) { (void)dst; (void)lp; }
void osc_simd_segment_parabola(float* dst, const SegmentLoopParams* lp) { (void)dst; (void)lp; }

int osc_simd_available(SpectralTimbre timbre) { (void)timbre; return 0; }

#endif

static int g_native_backend_available = 0;
void osc_set_native_available(int available) { g_native_backend_available = available; }
int osc_native_available(void) { return g_native_backend_available; }
