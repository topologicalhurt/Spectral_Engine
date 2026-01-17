/* spectral_synth_embedded.h - Q15 Fixed-Point Synthesis for ARM Cortex-M
 * 
 * Block-based streaming synthesizer designed for real-time embedded use.
 * Processes audio in fixed-size blocks, managing active segment state
 * across calls to avoid re-scanning the full segment list each block.
 * 
 * Memory Model:
 *   - SpectralEmbeddedCtx: main context (~32 bytes + active array)
 *   - SpectralSegmentQ15: 16-byte packed segments (stored in flash/SDRAM)
 *   - SpectralActiveSegQ15: runtime state for currently-playing segments
 *   - Q15 sine LUT: provided externally, typically in flash
 * 
 * API Flow:
 *   1. spectral_embedded_init() - set up context
 *   2. spectral_embedded_load() - load segment data
 *   3. spectral_embedded_process() - render audio blocks
 *   4. Repeat step 3 until playback complete
 */
#ifndef SPECTRAL_SYNTH_EMBEDDED_H
#define SPECTRAL_SYNTH_EMBEDDED_H

#include "spectral_config.h"
#include "spectral_q15.h"
#include "oscillator.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SPECTRAL_EMBEDDED

/*
 * Maximum concurrent active segments (polyphony limit)
 * 
 * CPU budget per block (256 samples @ 48kHz, 480MHz M7):
 *   Available cycles: (480MHz / 48kHz) * 256 = 2,560,000 cycles
 *   Per-segment cost: ~13 cycles/sample * 256 = ~3,328 cycles/segment
 *   Max voices: ~769 for 100% CPU (use 512 for ~67% headroom)
 * 
 * Memory: 16 bytes per active segment = 8KB for 512 voices
 */
#ifndef SPECTRAL_EMBEDDED_MAX_ACTIVE
#define SPECTRAL_EMBEDDED_MAX_ACTIVE    512
#endif

typedef SpectralActiveSegQ15 SpectralActiveSegment;

typedef struct SpectralEmbeddedCtx {
    const SpectralSegmentQ15* segments;
    uint32_t num_segments;
    uint32_t segments_capacity;
    
    uint32_t output_position;
    uint32_t output_length;
    uint32_t next_seg_idx;
    
    SpectralActiveSegment active[SPECTRAL_EMBEDDED_MAX_ACTIVE];
    uint16_t num_active;   /* Can reach SPECTRAL_EMBEDDED_MAX_ACTIVE (512) */
    uint16_t peak_active;  /* Peak polyphony tracking */
    
    const q15_t* osc_lut;
    
    uint32_t sample_rate;
    q15_t amplitude_q15;
    q15_t stretch_q214;
} SpectralEmbeddedCtx;

void spectral_embedded_init(SpectralEmbeddedCtx* ctx,
                            SpectralSegmentQ15* segments,
                            uint32_t capacity,
                            const q15_t* osc_lut,
                            uint32_t sample_rate);

void spectral_embedded_reset(SpectralEmbeddedCtx* ctx);
void spectral_embedded_seek(SpectralEmbeddedCtx* ctx, uint32_t sample_pos);

int spectral_embedded_load(SpectralEmbeddedCtx* ctx,
                           const SpectralSegmentQ15* data,
                           uint32_t num_segments,
                           uint32_t output_len);

uint32_t spectral_embedded_process(SpectralEmbeddedCtx* ctx,
                                   q15_t* out_left,
                                   q15_t* out_right,
                                   uint32_t num_samples);

uint32_t spectral_embedded_process_interleaved(SpectralEmbeddedCtx* ctx,
                                               q15_t* out_interleaved,
                                               uint32_t num_samples);

static inline int spectral_embedded_is_complete(const SpectralEmbeddedCtx* ctx) {
    return (!ctx) ? 1 : (ctx->output_position >= ctx->output_length);
}

static inline uint32_t spectral_embedded_get_position(const SpectralEmbeddedCtx* ctx) {
    return ctx ? ctx->output_position : 0;
}

static inline uint32_t spectral_embedded_get_duration(const SpectralEmbeddedCtx* ctx) {
    return ctx ? ctx->output_length : 0;
}

static inline uint16_t spectral_embedded_get_peak_active(const SpectralEmbeddedCtx* ctx) {
    return ctx ? ctx->peak_active : 0;
}

void spectral_embedded_set_amplitude(SpectralEmbeddedCtx* ctx, float amplitude);
void spectral_embedded_set_stretch(SpectralEmbeddedCtx* ctx, float stretch);

#if defined(SPECTRAL_RESTRICTED_MODE) && defined(SPECTRAL_DEBUG_RESTRICTED)
void restricted_synth_profile_start(void);
uint32_t restricted_synth_profile_end(void);
uint32_t restricted_synth_get_peak_cycles(void);
void restricted_synth_reset_profile(void);

/* Get formatted performance report
 * Returns bytes written (0 if buf_len < 256) */
int restricted_synth_get_perf_report(char* buf, int buf_len);
#endif

#endif /* SPECTRAL_EMBEDDED */

/* Desktop emulation for testing embedded synthesis path */
#if defined(SPECTRAL_USE_EMBEDDED_SYNTH) || defined(SPECTRAL_EMBEDDED_EMULATION)

#include "spectral_common.h"

void emulator_set_config(uint32_t cpu_mhz, uint32_t sample_rate,
                         uint32_t block_size, uint32_t max_mem_kb);
void emulator_set_verbose(int verbose);

/* Desktop wrapper: converts float segments to Q15, runs embedded synth.
 * n_threads is IGNORED - embedded is single-threaded.
 * Returns: SPECTRAL_OK on success, negative error code on failure */
int synth_embedded_emulation(SegmentArray sa, float* out_buffer, size_t out_len,
                             float stretch, float pitch, SpectralTimbre timbre,
                             int n_threads, double* t_synth);

#ifdef SPECTRAL_USE_EMBEDDED_SYNTH
#define synth_cpu synth_embedded_emulation
#endif

#endif /* SPECTRAL_USE_EMBEDDED_SYNTH || SPECTRAL_EMBEDDED_EMULATION */

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_SYNTH_EMBEDDED_H */
