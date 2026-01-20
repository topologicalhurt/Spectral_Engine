/* spectral_synth_arm32.h - Q15 Fixed-Point Synthesis for ARM Cortex-M
 * 
 * Block-based streaming synthesizer for real-time ARM32 embedded use.
 * Processes audio in fixed-size blocks, manages active segment state
 * across calls to avoid re-scanning the full segment list each block.
 * 
 * Memory Model:
 *   - SpectralArm32Ctx: main context (~32 bytes + active array)
 *   - SpectralSegmentQ15: 16-byte packed segments (stored in flash/SDRAM)
 *   - SpectralActiveSegQ15: runtime state for currently-playing segments
 *   - Q15 LUT: provided externally, typically in DTCM or flash
 * 
 * API Flow:
 *   1. spectral_arm32_init() - set up context
 *   2. spectral_arm32_load() - load segment data
 *   3. spectral_arm32_process() - render audio blocks
 *   4. Repeat step 3 until playback complete
 */
#ifndef SPECTRAL_SYNTH_ARM32_H
#define SPECTRAL_SYNTH_ARM32_H

#include "spectral_config.h"
#include "spectral_q15.h"
#include "oscillator.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SPECTRAL_EMBEDDED

/* Polyphony limit - see spectral_config.h for SPECTRAL_ARM32_MAX_ACTIVE */
#ifndef SPECTRAL_ARM32_MAX_ACTIVE
#define SPECTRAL_ARM32_MAX_ACTIVE    512
#endif

typedef SpectralActiveSegQ15 SpectralActiveSegment;

typedef struct SpectralArm32Ctx {
    const SpectralSegmentQ15* segments;
    uint32_t num_segments;
    uint32_t segments_capacity;
    
    uint32_t output_position;
    uint32_t output_length;
    uint32_t next_seg_idx;
    
    SpectralActiveSegment active[SPECTRAL_ARM32_MAX_ACTIVE];
    uint16_t num_active;
    uint16_t peak_active;
    
    const q15_t* osc_lut;
    
    uint32_t sample_rate;
    uint32_t freq_inc_scale_q24;
    q15_t amplitude_q15;
    q15_t stretch_q214;
} SpectralArm32Ctx;

void spectral_arm32_init(SpectralArm32Ctx* ctx,
                         SpectralSegmentQ15* segments,
                         uint32_t capacity,
                         const q15_t* osc_lut,
                         uint32_t sample_rate);

void spectral_arm32_reset(SpectralArm32Ctx* ctx);
void spectral_arm32_seek(SpectralArm32Ctx* ctx, uint32_t sample_pos);

SpectralError spectral_arm32_load(SpectralArm32Ctx* ctx,
                                  const SpectralSegmentQ15* data,
                                  uint32_t num_segments,
                                  uint32_t output_len);

uint32_t spectral_arm32_process(SpectralArm32Ctx* ctx,
                                q15_t* out_left,
                                q15_t* out_right,
                                uint32_t num_samples);

uint32_t spectral_arm32_process_interleaved(SpectralArm32Ctx* ctx,
                                            q15_t* out_interleaved,
                                            uint32_t num_samples);

static inline int spectral_arm32_is_complete(const SpectralArm32Ctx* ctx) {
    return (!ctx) ? 1 : (ctx->output_position >= ctx->output_length);
}

static inline uint32_t spectral_arm32_get_position(const SpectralArm32Ctx* ctx) {
    return ctx ? ctx->output_position : 0;
}

static inline uint32_t spectral_arm32_get_duration(const SpectralArm32Ctx* ctx) {
    return ctx ? ctx->output_length : 0;
}

static inline uint16_t spectral_arm32_get_peak_active(const SpectralArm32Ctx* ctx) {
    return ctx ? ctx->peak_active : 0;
}

void spectral_arm32_set_amplitude(SpectralArm32Ctx* ctx, float amplitude);
void spectral_arm32_set_stretch(SpectralArm32Ctx* ctx, float stretch);

#if defined(SPECTRAL_RESTRICTED_MODE) && defined(SPECTRAL_DEBUG_RESTRICTED)
void arm32_synth_profile_start(void);
uint32_t arm32_synth_profile_end(void);
uint32_t arm32_synth_get_peak_cycles(void);
void arm32_synth_reset_profile(void);

uint32_t arm32_synth_get_total_cycles(void);
uint32_t arm32_synth_get_call_count(void);
uint32_t arm32_synth_get_avg_cycles(void);
uint32_t arm32_synth_get_min_cycles(void);
uint32_t arm32_synth_get_max_cycles(void);
uint32_t arm32_synth_get_samples_processed(void);
#endif

#endif /* SPECTRAL_EMBEDDED */

/* Desktop emulation for testing ARM32 synthesis path */
#if defined(SPECTRAL_USE_EMBEDDED_SYNTH) || defined(SPECTRAL_EMBEDDED_EMULATION)

#include "spectral_common.h"

void emulator_set_config(uint32_t cpu_mhz, uint32_t sample_rate,
                         uint32_t block_size, uint32_t max_mem_kb);
void emulator_set_verbose(int verbose);

SpectralError synth_arm32_emulation(SegmentArray sa, float* out_buffer, size_t out_len,
                                    float stretch, float pitch, SpectralTimbre timbre,
                                    int n_threads, double* t_synth);

#ifdef SPECTRAL_USE_EMBEDDED_SYNTH
#define synth_cpu synth_arm32_emulation
#endif

#endif /* SPECTRAL_USE_EMBEDDED_SYNTH || SPECTRAL_EMBEDDED_EMULATION */

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_SYNTH_ARM32_H */
