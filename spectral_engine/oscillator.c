/* oscillator.c - Oscillator Implementation
 * 
 * Implements sine LUT generation and lookup for Q15 synthesis,
 * plus float-based timbre oscillator for desktop.
 * 
 * LUT Details:
 *   - Size: 2^SPECTRAL_OSC_LUT_BITS entries + 1 (for wrap interpolation)
 *   - Range: [-32700, +32700] (slightly below Q15_MAX for headroom)
 *   - Lookup uses linear interpolation between adjacent samples
 * 
 * Phase Accumulator:
 *   - 16-bit unsigned (0-65535 maps to 0-2pi)
 *   - Upper bits select LUT index, lower bits are interpolation fraction
 *   - Cosine lookup adds 16384 (quarter-wave offset)
 * 
 * Timbre Oscillator:
 *   Uses function pointer table for branch-free dispatch.
 *   Each waveform is a separate inline function.
 */
#include "oscillator.h"
#include "spectral_synth_internal.h"
#include "spectral_utils.h"
#include <math.h>

/* Generate sine LUT with +1 entry for wraparound interpolation */
void spectral_osc_lut_init_sine(q15_t* lut) {
    if (!lut) return;
    
    const float scale = 32700.0f;  /* Slightly below Q15_MAX for headroom */
    
    for (uint32_t i = 0; i <= SPECTRAL_OSC_LUT_SIZE; i++) {
        float phase = (float)i / (float)SPECTRAL_OSC_LUT_SIZE * TWO_PI;
        float val = sinf(phase);
        lut[i] = (q15_t)(val * scale);
    }
    
    /* Force exact values at cardinal points to eliminate rounding error */
    lut[0] = 0;                                             /* sin(0) = 0 */
    lut[SPECTRAL_OSC_LUT_SIZE / 2] = 0;                     /* sin(pi) = 0 */
    lut[SPECTRAL_OSC_LUT_SIZE / 4] = (q15_t)scale;          /* sin(pi/2) = 1 */
    lut[3 * SPECTRAL_OSC_LUT_SIZE / 4] = (q15_t)(-scale);   /* sin(3pi/2) = -1 */
}

/* LUT lookup: upper bits = index, lower bits = interpolation fraction */
q15_t spectral_osc_lut_lookup(uq16_t phase_u16, const q15_t* lut) {
    uint32_t idx = phase_u16 >> (16 - SPECTRAL_OSC_LUT_BITS);
    uint32_t frac_raw = phase_u16 & SPECTRAL_OSC_FRAC_MASK;
    uint32_t frac = (SPECTRAL_OSC_FRAC_BITS >= 8) 
        ? (frac_raw >> (SPECTRAL_OSC_FRAC_BITS - 8)) 
        : (frac_raw << (8 - SPECTRAL_OSC_FRAC_BITS));
    
    q15_t s0 = lut[idx];
    q15_t s1 = lut[idx + 1];  /* LUT has SIZE+1 entries for wraparound interpolation */
    return (q15_t)(s0 + ((((q31_t)s1 - (q31_t)s0) * (int32_t)frac) >> 8));
}

q15_t spectral_osc_lut_lookup_cos(uq16_t phase_u16, const q15_t* lut) {
    /* Cosine = sine with 90deg (quarter-wave) phase offset = 16384 in 16-bit */
    return spectral_osc_lut_lookup(phase_u16 + 16384, lut);
}

/* Initialize per-sample loop parameters from segment definition */
SegmentLoopParams segment_loop_params_init(const Segment* s, const SynthParams* p, size_t out_len) {
    SegmentLoopParams lp;
    
    lp.start_idx = (size_t)(s->start * p->stretch);
    lp.length = (size_t)(s->length * p->stretch);
    
    /* Bounds checking */
    if (lp.start_idx >= out_len) {
        lp.valid = 0;
        return lp;
    }
    if (lp.start_idx + lp.length > out_len) {
        lp.length = out_len - lp.start_idx;
    }
    
    /* Frequency: Hz * 2pi / sr, then scaled by pitch and inverse stretch */
    lp.alpha = s->freq_hz * p->pitch_factor * p->inv_stretch;
    /* Chirp: df/dt scaled by pitch and inverse stretch squared */
    lp.beta = s->df * p->pitch_factor * p->inv_stretch_sq;
    /* Amplitude change per stretched sample */
    lp.d_amp = s->da * p->inv_stretch;
    lp.phase = s->phase;
    lp.amp = s->amp;
    lp.width = s->width;
    lp.valid = 1;
    
    return lp;
}

/*
 * Individual waveform generators
 * Each takes normalized phase (rads in [-pi, pi)) and width parameter
 */

static inline float osc_sine(float rads, float width) {
    (void)width;
    return fast_sin(rads + PI);  /* Shift to match original phase convention */
}

static inline float osc_saw(float rads, float width) {
    (void)width;
    return rads * -SPECTRAL_INV_PI;
}

static inline float osc_square(float rads, float width) {
    (void)width;
    return (rads > 0.0f) ? 1.0f : -1.0f;
}

static inline float osc_triangle(float rads, float width) {
    (void)width;
    return (1.0f - fabsf(rads * SPECTRAL_TWO_INV_PI)) * 2.0f - 1.0f;
}

static inline float osc_asin(float rads, float width) {
    (void)width;
    return asinf(rads * SPECTRAL_INV_PI);
}

static inline float osc_parabola(float rads, float width) {
    (void)width;
    return 1.0f - (rads * rads * SPECTRAL_INV_PI_SQ);
}

static inline float osc_quantized(float rads, float width) {
    return (width > 0.0f) ? ((int)(rads * width) / width) : 0.0f;
}

static inline float osc_pwm(float rads, float width) {
    return (width > 0.0f) ? (((rads + PI) * INV_TWO_PI < width) ? 1.0f : -1.0f) : 1.0f;
}

/* Function pointer type for waveform generators */
typedef float (*TimbreFunc)(float rads, float width);

/* Jump table indexed by SpectralTimbre enum */
static const TimbreFunc timbre_table[TIMBRE_COUNT] = {
    osc_sine,       /* TIMBRE_SINE     = 0 */
    osc_saw,        /* TIMBRE_SAW      = 1 */
    osc_square,     /* TIMBRE_SQUARE   = 2 */
    osc_triangle,   /* TIMBRE_TRIANGLE = 3 */
    osc_asin,       /* TIMBRE_ASIN     = 4 */
    osc_parabola,   /* TIMBRE_PARABOLA = 5 */
    osc_quantized,  /* TIMBRE_QUANTIZED= 6 */
    osc_pwm         /* TIMBRE_PWM      = 7 */
};

/*
 * timbre_oscillator: Branch-free waveform generation via function pointer table
 * 
 * Parameters:
 *   p        - Phase in radians
 *   a        - Amplitude
 *   timbre   - Waveform type (SpectralTimbre enum, must be < TIMBRE_COUNT)
 *   width    - Width parameter (for PWM and quantized)
 * 
 * Returns: Waveform sample value (amplitude-scaled)
 * 
 * Note: Invalid timbre values are clamped to TIMBRE_SINE for safety.
 */
float timbre_oscillator(float p, float a, SpectralTimbre timbre, float width) {
    /* Bounds check - clamp invalid values to sine (safest default) */
    unsigned int idx = (unsigned int)timbre;
    if (idx >= TIMBRE_COUNT) {
        idx = TIMBRE_SINE;
    }
    
    /* Normalize phase to [-0.5, 0.5) then scale to [-pi, pi) */
    float norm = p * INV_TWO_PI;
    float rads = TWO_PI * (norm - (int)norm + (norm < 0.0f) - 0.5f);
    
    /* Branch-free dispatch through function pointer table */
    return a * timbre_table[idx](rads, width);
}

/*
 * timbre_synth_segment: SIMD-optimized segment rendering
 * 
 * Renders an entire segment, allowing resynthesis with waveforms
 * other than sine. We implement SIMD vectorization for select 'primitive' waveforms. 
 * Complex timbres fall back to the function pointer table.
 * 
 * Parameters:
 *   dst    - Output buffer (samples are ADDED, not overwritten)
 *   lp     - Segment loop parameters (phase, freq, amplitude, etc.)
 *   timbre - Waveform type (SpectralTimbre enum, must be < TIMBRE_COUNT)
 * 
 */
void timbre_synth_segment(float* __restrict__ dst, const SegmentLoopParams* lp, SpectralTimbre timbre) {
    if ((unsigned int)timbre >= TIMBRE_COUNT) {
        SPECTRAL_WARN_ONCE(TIMBRE_COUNT, "Invalid timbre %d, using sine", (int)timbre);
        timbre = TIMBRE_SINE;
    }
    
    const size_t len = lp->length;
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;
    const float width = lp->width;
    
    /* 
     * Phase Calculation Strategy:
     * 
     * Mathematical formula: p(j) = phase0 + j * (alpha + beta * j)
     *                           = phase0 + alpha*j + beta*j^2
     * 
     * For SIMD timbres: Use direct formula (compiler can vectorize)
     * For non-SIMD timbres: Use incremental accumulation (more accurate for long segments)
     * 
     * The incremental form is:
     *   p[j+1] = p[j] + freq[j]
     *   freq[j+1] = freq[j] + 2*beta
     * 
     * This avoids floating-point precision loss from multiplying large j values.
     */
    
    switch (timbre) {
    case TIMBRE_SINE:
        #pragma omp simd
        for (size_t j = 0; j < len; j++) {
            float p = phase0 + j * (alpha + beta * j);
            dst[j] += (amp0 + d_amp * j) * fast_sin(p);
        }
        break;
    case TIMBRE_SAW:
        #pragma omp simd
        for (size_t j = 0; j < len; j++) {
            float p = phase0 + j * (alpha + beta * j);
            float norm = p * INV_TWO_PI;
            float rads = TWO_PI * (norm - (int)norm + (norm < 0.0f) - 0.5f);
            dst[j] += (amp0 + d_amp * j) * (rads * -SPECTRAL_INV_PI);
        }
        break;
    case TIMBRE_SQUARE:
        #pragma omp simd
        for (size_t j = 0; j < len; j++) {
            float p = phase0 + j * (alpha + beta * j);
            float norm = p * INV_TWO_PI;
            float rads = TWO_PI * (norm - (int)norm + (norm < 0.0f) - 0.5f);
            dst[j] += (amp0 + d_amp * j) * ((rads > 0.0f) ? 1.0f : -1.0f);
        }
        break;
    case TIMBRE_TRIANGLE:
        #pragma omp simd
        for (size_t j = 0; j < len; j++) {
            float p = phase0 + j * (alpha + beta * j);
            float norm = p * INV_TWO_PI;
            float rads = TWO_PI * (norm - (int)norm + (norm < 0.0f) - 0.5f);
            float abs_r = rads < 0.0f ? -rads : rads;
            dst[j] += (amp0 + d_amp * j) * ((1.0f - abs_r * SPECTRAL_TWO_INV_PI) * 2.0f - 1.0f);
        }
        break;
    case TIMBRE_PARABOLA:
        #pragma omp simd
        for (size_t j = 0; j < len; j++) {
            float p = phase0 + j * (alpha + beta * j);
            float norm = p * INV_TWO_PI;
            float rads = TWO_PI * (norm - (int)norm + (norm < 0.0f) - 0.5f);
            dst[j] += (amp0 + d_amp * j) * (1.0f - rads * rads * SPECTRAL_INV_PI_SQ);
        }
        break;
        
    /* Non-SIMD timbres: use incremental phase accumulation for better precision */
    case TIMBRE_ASIN:
    case TIMBRE_QUANTIZED:
    case TIMBRE_PWM: {
        /* Incremental phase accumulation avoids precision loss for long segments.
         * 
         * Given p(j) = phase0 + alpha*j + beta*j^2, the increment is:
         *   p(j+1) - p(j) = alpha + beta*(2j+1)
         * 
         * So we track: freq = alpha + beta*(2j+1), d_freq = 2*beta
         */
        float phase = phase0;
        float freq = alpha + beta;     /* Initial frequency increment */
        float d_freq = 2.0f * beta;    /* Frequency change per sample */
        float amp = amp0;
        
        for (size_t j = 0; j < len; j++) {
            dst[j] += timbre_oscillator(phase, amp, timbre, width);
            phase += freq;
            freq += d_freq;
            amp += d_amp;
        }
        break;
    }
        
    default:
        __builtin_unreachable();
    }
}
