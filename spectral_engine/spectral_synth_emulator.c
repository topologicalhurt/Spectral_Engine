/* spectral_synth_emulator.c - Desktop emulation of embedded Q15 synthesis
 * Allows testing embedded synthesis path without hardware.
 */

#define SPECTRAL_EMBEDDED 1
#define SPECTRAL_EMBEDDED_EMULATION 1

#include "spectral_synth_embedded.h"
#include "spectral_synth_internal.h"
#include "spectral_error.h"
#include "spectral_common.h"
#include "spectral_q15.h"
#include "spectral_wavetable.h"
#include "spectral_perf.h"
#include "spectral_utils.h"
#include "oscillator.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Emulator configuration */

static EmbeddedTargetConfig g_emulator_config;
static int g_emulator_config_initialized = 0;

static EmbeddedTargetConfig* get_emulator_config(void) {
    if (!g_emulator_config_initialized) {
        g_emulator_config = embedded_perf_default_config();
        g_emulator_config_initialized = 1;
    }
    return &g_emulator_config;
}

void emulator_set_config(uint32_t cpu_mhz, uint32_t sample_rate, 
                         uint32_t block_size, uint32_t max_mem_kb) {
    EmbeddedTargetConfig* cfg = get_emulator_config();
    if (cpu_mhz > 0) cfg->cpu_freq_mhz = cpu_mhz;
    if (sample_rate > 0) cfg->sample_rate = sample_rate;
    if (block_size > 0) cfg->block_size = block_size;
    if (max_mem_kb > 0) cfg->max_memory_kb = max_mem_kb;
}

void emulator_set_verbose(int verbose) {
    get_emulator_config()->verbose = verbose;
}

/* Emulator Segment - matches SpectralSegmentQ15 in representation but uses
 * q31_t freq_inc for higher precision frequency during desktop emulation.
 * 
 * Phase representation: phase_q15 is signed Q15 where [-32768, 32767] maps to [-pi, pi).
 * This matches SpectralSegmentQ15.phase_q15 exactly.
 * 
 * The embedded synth converts phase_q15 to unsigned Q31 accumulator:
 *   phase_acc = (phase_q15 + 32768) << 16
 */
typedef struct {
    uint32_t start;
    uint16_t length;
    q31_t    freq_inc;
#if SPECTRAL_HAS_CHIRP
    q31_t    df_inc;
#endif
    int16_t  phase_q15;
    int16_t  amp_q15;
    int16_t  da_q15;
} EmulatorSegment;

static void segment_to_emulator(const Segment* src, EmulatorSegment* dst,
                                uint32_t sample_rate, float amp_scale,
                                float pitch_factor, float inv_stretch) {
    (void)sample_rate;
    
    dst->start = (uint32_t)src->start;
    dst->length = (uint16_t)fminf(65535.0f, src->length);
    
    /* Frequency: compute Q31 phase increment per sample
     * Apply pitch_factor (2^(semitones/12)) and inv_stretch to match desktop synth.
     * omega is already in radians/sample, so: freq_inc = omega * pitch * inv_stretch * 2^32 / 2pi
     * We use double precision to avoid overflow */
    double freq_scaled = src->omega * pitch_factor * inv_stretch;
    double freq_inc = freq_scaled * SPECTRAL_Q31_PER_RAD;
    dst->freq_inc = (q31_t)CLAMP(freq_inc, (double)Q31_MIN, (double)Q31_MAX);
    
#if SPECTRAL_HAS_CHIRP
    /* Chirp (df): frequency change per sample, converted to Q31 increment delta
     * Apply pitch_factor and inv_stretch^2 to match desktop synth */
    double df_scaled = src->df * pitch_factor * inv_stretch * inv_stretch;
    double df_inc = df_scaled * SPECTRAL_Q31_PER_RAD;
    dst->df_inc = (q31_t)CLAMP(df_inc, (double)Q31_MIN, (double)Q31_MAX);
#endif
    
    /* Phase: convert [0, 2pi) to signed Q15 [-pi, pi) representation */
    dst->phase_q15 = PHASE_RAD_TO_Q15(src->phase);
    
    /* Amplitude: scale and saturate to Q15 */
    float amp_scaled = CLAMP(src->amp * amp_scale, 0.0f, 1.0f);
    dst->amp_q15 = FLOAT_TO_Q15(amp_scaled);
    
    /* Amplitude delta per sample (also scaled by inv_stretch like desktop) */
    float da_scaled = src->da * amp_scale * inv_stretch;
    dst->da_q15 = FLOAT_TO_Q15(CLAMP(da_scaled, -1.0f, 1.0f));
}

/* Oscillator LUT */

static const q15_t* get_emulation_lut(void) {
    static q15_t lut[SPECTRAL_OSC_LUT_SIZE + 1];
    static int initialized = 0;
    
    if (!initialized) {
        spectral_osc_lut_init_sine(lut);
        initialized = 1;
    }
    return lut;
}

/* Active segment state - runtime state for emulation */

typedef struct {
    uint32_t seg_idx;
    q31_t    phase_acc;
    q31_t    freq_inc;
#if SPECTRAL_HAS_CHIRP
    q31_t    df_inc;
#endif
    q15_t    amp;
    q15_t    da;
} EmulatorActiveSegment;

#define EMULATOR_MAX_ACTIVE SPECTRAL_EMBEDDED_MAX_ACTIVE

/* Desktop emulation entry point */

SpectralError synth_embedded_emulation(SegmentArray sa, float* out_buffer, size_t out_len,
                                      float stretch, float pitch, SpectralTimbre timbre,
                                      int n_threads, double* t_synth) {
    (void)n_threads;
    (void)timbre;  /* TODO: support non-sine timbres in emulator */
    
    if (!SYNTH_VALIDATE_FLOAT(out_buffer, out_len, sa, &t_synth)) {
        return SPECTRAL_OK;
    }

    double start_time = spectral_get_time_sec();
    
    /* Compute pitch factor: 2^(semitones/12) */
    float pitch_factor = powf(2.0f, pitch / 12.0f);
    float inv_stretch = 1.0f / stretch;
    
    /* Find maximum amplitude for scaling all segments to fit in Q15 */
    float max_amp = 0.0f;
    for (size_t i = 0; i < sa.count; i++) {
        if (sa.segs[i].amp > max_amp) max_amp = sa.segs[i].amp;
    }
    /* amp_scale brings max_amp to 1.0, with a bit of headroom */
    float amp_scale = (max_amp > 0.0f) ? (0.99f / max_amp) : 1.0f;
    
    /* Convert segments to emulator format */
    EmulatorSegment* emu_segs = (EmulatorSegment*)malloc(sa.count * sizeof(EmulatorSegment));
    if (!emu_segs) {
        SPECTRAL_WARN("Emulator: segment buffer allocation failed (%zu segments)", sa.count);
        memset(out_buffer, 0, out_len * sizeof(float));
        if (t_synth) *t_synth = 0;
        return SPECTRAL_ERR_MEMORY;
    }
    
    for (size_t i = 0; i < sa.count; i++) {
        segment_to_emulator(&sa.segs[i], &emu_segs[i], 44100, amp_scale, pitch_factor, inv_stretch);
        /* Apply stretch to timing */
        emu_segs[i].start = (uint32_t)(emu_segs[i].start * stretch);
        emu_segs[i].length = (uint16_t)fminf(65535.0f, emu_segs[i].length * stretch);
    }
    
    /* Debug: print first few segments */
    EmbeddedTargetConfig* cfg = get_emulator_config();
#if SPECTRAL_DEBUG && !defined(NDEBUG)
    if (cfg->verbose) {
        SPECTRAL_DBG("max_amp=%.3f amp_scale=%.6f", max_amp, amp_scale);
        SPECTRAL_DBG("First 5 segments:");
        for (size_t i = 0; i < 5 && i < sa.count; i++) {
            Segment* s = &sa.segs[i];
            EmulatorSegment* e = &emu_segs[i];
            SPECTRAL_DBG("  [%zu] desktop: start=%.0f len=%.0f omega=%.6f phase=%.3f amp=%.6f da=%.6f",
                   i, s->start, s->length, s->omega, s->phase, s->amp, s->da);
            SPECTRAL_DBG("       emulator: start=%u len=%u freq_inc=%d phase_q15=%d amp_q15=%d da_q15=%d",
                   e->start, e->length, e->freq_inc, e->phase_q15, e->amp_q15, e->da_q15);
        }
    }
    
    /* Check amplitude distribution */
    if (cfg->verbose) {
        float sum_amp = 0;
        for (size_t i = 0; i < sa.count; i++) {
            sum_amp += sa.segs[i].amp;
        }
        float avg_amp = (sa.count > 0) ? (sum_amp / sa.count) : 0.0f;
        SPECTRAL_DBG("Segment amp stats: max=%.6f avg=%.6f total=%.3f",
               max_amp, avg_amp, sum_amp);
    }
#endif
    
    /* Get oscillator LUT */
    const q15_t* osc_lut = get_emulation_lut();
#if SPECTRAL_DEBUG && !defined(NDEBUG)
    if (cfg->verbose) {
        SPECTRAL_DBG("Oscillator LUT size=%d (%d bits)", 
               SPECTRAL_OSC_LUT_SIZE, SPECTRAL_OSC_LUT_BITS);
    }
#endif
    
    /* Allocate 64-bit accumulator buffer (prevents overflow with many segments) */
    const uint32_t block_size = 256;
    int64_t* accum = (int64_t*)calloc(block_size, sizeof(int64_t));
    if (!accum) {
        SPECTRAL_WARN("Emulator: accumulator buffer allocation failed (%u samples)", block_size);
        free(emu_segs);
        memset(out_buffer, 0, out_len * sizeof(float));
        if (t_synth) *t_synth = 0;
        return SPECTRAL_ERR_MEMORY;
    }
    
    /* Active segment tracking */
    EmulatorActiveSegment active[EMULATOR_MAX_ACTIVE];
    uint32_t num_active = 0;
    uint32_t peak_active = 0;
    uint32_t next_seg_idx = 0;
    
    /* Operation counting for accurate performance estimation */
    EmbeddedOpCounts ops = {0};
    
    /* Process output in blocks */
    size_t out_pos = 0;
    while (out_pos < out_len) {
        uint32_t block_len = (out_len - out_pos > block_size) ? block_size : (uint32_t)(out_len - out_pos);
        memset(accum, 0, block_len * sizeof(int64_t));
        
        uint32_t block_end = (uint32_t)out_pos + block_len;
        
        /* Activate new segments that start in this block */
        while (next_seg_idx < sa.count && num_active < EMULATOR_MAX_ACTIVE) {
            EmulatorSegment* seg = &emu_segs[next_seg_idx];
            if (seg->start >= block_end) break;
            
            uint32_t seg_end = seg->start + seg->length;
            if (seg_end > out_pos) {
                /* Activate this segment */
                EmulatorActiveSegment* act = &active[num_active];
                act->seg_idx = next_seg_idx;
                act->freq_inc = seg->freq_inc;
#if SPECTRAL_HAS_CHIRP
                act->df_inc = seg->df_inc;
#endif
                act->amp = seg->amp_q15;
                act->da = seg->da_q15;
                
                /* Initialize phase from stored Q15 value */
                act->phase_acc = ((q31_t)seg->phase_q15 + 32768) << 16;
                
                /* Advance phase if segment started before this block */
                if (seg->start < out_pos) {
                    uint32_t samples_in = (uint32_t)out_pos - seg->start;
#if SPECTRAL_HAS_CHIRP
                    /* With chirp: phase += n*freq + n*(n-1)/2 * df */
                    q31_t chirp_contrib = (q31_t)(((int64_t)samples_in * (samples_in - 1) / 2) * act->df_inc >> 16);
                    act->phase_acc += samples_in * act->freq_inc + chirp_contrib;
                    act->freq_inc += samples_in * act->df_inc;
#else
                    act->phase_acc += samples_in * act->freq_inc;
#endif
                    /* Advance amplitude: use 32-bit intermediate to prevent overflow */
                    q31_t amp_advance = (q31_t)act->da * (q31_t)samples_in;
                    act->amp = spectral_ssat16((q31_t)act->amp + amp_advance);
                }
                
                num_active++;
            }
            next_seg_idx++;
        }
        
        if (num_active > peak_active) peak_active = num_active;
        
        /* Process all active segments */
        uint32_t i = 0;
        while (i < num_active) {
            EmulatorActiveSegment* act = &active[i];
            EmulatorSegment* seg = &emu_segs[act->seg_idx];
            uint32_t seg_end = seg->start + seg->length;
            
            /* Remove expired segments */
            if (out_pos >= seg_end) {
                active[i] = active[--num_active];
                continue;
            }
            
            /* Compute block range for this segment */
            uint32_t blk_start = (seg->start > out_pos) ? (seg->start - (uint32_t)out_pos) : 0;
            uint32_t blk_end = (seg_end < block_end) ? (seg_end - (uint32_t)out_pos) : block_len;
            uint32_t len = blk_end - blk_start;
            
            /* Count operations for performance estimation
             * This models what spectral_synth_embedded.c does on real hardware */
            ops.lut_lookups += len;           /* 1 LUT lookup per sample */
            ops.mac_operations += len;        /* 1 MAC per sample */
            ops.phase_updates += len;         /* 1 phase update per sample */
            ops.loop_iterations += (len + 3) >> 2;
            
            q31_t phase = act->phase_acc;
            q31_t freq_inc = act->freq_inc;
#if SPECTRAL_HAS_CHIRP
            q31_t df_inc = act->df_inc;
#endif
            q15_t amp = act->amp;
            q15_t da = act->da;
            
            for (uint32_t j = blk_start; j < blk_end; j++) {
                uq16_t lut_idx = (uq16_t)(phase >> 16);
                q15_t sample = spectral_osc_lut_lookup(lut_idx, osc_lut);
                accum[j] += (int64_t)sample * amp;
                phase += freq_inc;
#if SPECTRAL_HAS_CHIRP
                freq_inc += df_inc;
#endif
                amp = spectral_qadd16(amp, da);
            }
            
            act->phase_acc = phase;
            act->freq_inc = freq_inc;
            act->amp = amp;
            i++;
        }
        
        /* Convert accumulator to float output
         * accum contains sum of (Q15 * Q15) = Q30 products */
        const double scale = 1.0 / 1073741824.0;
        for (uint32_t j = 0; j < block_len; j++) {
            out_buffer[out_pos + j] = (float)((double)accum[j] * scale);
        }
        
#if SPECTRAL_DEBUG && !defined(NDEBUG)
        /* Debug: print first block accumulator stats */
        if (out_pos == 0 && cfg->verbose) {
            int64_t min_acc = 0, max_acc = 0;
            for (uint32_t j = 0; j < block_len; j++) {
                if (accum[j] < min_acc) min_acc = accum[j];
                if (accum[j] > max_acc) max_acc = accum[j];
            }
            SPECTRAL_DBG("first block: num_active=%u min_acc=%lld max_acc=%lld",
                   num_active, min_acc, max_acc);
        }
#endif
        
        out_pos += block_len;
    }
    
#if SPECTRAL_DEBUG && !defined(NDEBUG)
    if (cfg->verbose) {
        SPECTRAL_DBG("peak_active=%u (max allowed=%d)", peak_active, EMULATOR_MAX_ACTIVE);
    }
#endif
    
    free(accum);
    free(emu_segs);
    
    double elapsed = spectral_get_time_sec() - start_time;
    *t_synth = elapsed;
    
    /* Calculate and print embedded target performance estimates using actual op counts */
    EmbeddedPerfEstimate est = embedded_perf_estimate(
        cfg, &ops, out_len, sa.count, peak_active, elapsed);
    
    embedded_perf_print(cfg, &est);
    
    /* Print exact memory usage */
    EmbeddedMemoryUsage mem = embedded_memory_usage(
        sa.count,
        block_size,
        SPECTRAL_OSC_LUT_BITS,
        EMULATOR_MAX_ACTIVE,
        cfg->max_memory_kb
    );
    embedded_memory_print(&mem);
    
    return SPECTRAL_OK;
}

/* synth_cpu is provided via macro in spectral_synth_embedded.h
 * when SPECTRAL_USE_EMBEDDED_SYNTH is defined, redirecting to synth_embedded_emulation */

/* Wavetable version - falls back to emulation (wavetables not yet supported) */
#ifdef SPECTRAL_USE_EMBEDDED_SYNTH
SpectralError synth_cpu_wavetable(SegmentArray sa, float* out_buffer, size_t out_len,
                                  float stretch, float pitch,
                                  const SpectralWavetableBank* bank, SpectralTimbre timbre,
                                  int n_threads, double* t_synth) {
    (void)bank;
    if (bank != NULL) {
        SPECTRAL_WARN_ONCE(TIMBRE_COUNT + 1, "Wavetable not supported in embedded emulation, using default sine");
    }
    return synth_embedded_emulation(sa, out_buffer, out_len, stretch, pitch,
                                    timbre, n_threads, t_synth);
}
#endif
