/* spectral_synth_internal.h - Internal Synthesis Helpers
 * 
 * Common includes and utilities for synthesis backends (CPU, Metal, CUDA, embedded).
 * Provides unified access to:
 *   - spectral_common.h: Fast math, segments, core types
 *   - spectral_config.h: Build-time configuration
 *   - oscillator.h: SegmentLoopParams, timbre functions, Q15 types
 *   - Shared input validation to avoid code duplication
 * 
 * Not intended for external use - include spectral_synth.h instead.
 */
#ifndef SPECTRAL_SYNTH_INTERNAL_H
#define SPECTRAL_SYNTH_INTERNAL_H

#include "spectral_common.h"
#include "spectral_config.h"
#include "spectral_error.h"
#include "oscillator.h"
#include <string.h>

/* Precomputed segment parameters for inner synthesis loop */
typedef struct SegmentLoopParams {
    size_t start_idx;
    size_t length;
    float alpha;    /* phase increment per sample */
    float beta;     /* chirp rate (phase acceleration) */
    float d_amp;    /* amplitude delta per sample */
    float phase;
    float amp;
    float width;
    int valid;
} SegmentLoopParams;

SegmentLoopParams segment_loop_params_init(const Segment* s, const SynthParams* p, size_t out_len);

/* Common entry-point validation for all synth backends.
   Handles: null t_synth pointer, empty buffers, empty segment arrays. */

typedef enum {
    SYNTH_VALIDATE_OK = 1,        /* Proceed with synthesis */
    SYNTH_VALIDATE_EARLY_EXIT = 0 /* Return early (not an error, just nothing to do) */
} SynthValidateResult;

/* Validate inputs and prepare timing pointer. Call at the start of every synth function.
   Returns SYNTH_VALIDATE_OK to proceed, SYNTH_VALIDATE_EARLY_EXIT if nothing to do. */
SynthValidateResult synth_validate_inputs(void* out_buffer, size_t out_len, size_t elem_size,
                                          SegmentArray sa, double** t_synth_ptr);

/* Convenience macro for float buffers */
#define SYNTH_VALIDATE_FLOAT(buf, len, sa, t_ptr) \
    synth_validate_inputs((buf), (len), sizeof(float), (sa), (t_ptr))

/* Convenience macro for native sample buffers */
#define SYNTH_VALIDATE_NATIVE(buf, len, sa, t_ptr) \
    synth_validate_inputs((buf), (len), sizeof(spectral_sample_t), (sa), (t_ptr))

/* GPU backends only support timbres 0-5. Timbres 6-7 need the width param, so CPU only. */

/* Check if timbre is GPU-supported; if not, calls synth_cpu() and returns 0. */
int gpu_check_timbre_or_fallback(const char* backend_name,
                                  SegmentArray sa, float* out_buffer, size_t out_len,
                                  float stretch, float pitch, SpectralTimbre timbre, 
                                  double* t_synth);

/* Returns 1 if timbre is GPU-compatible, 0 otherwise. */
static inline int gpu_timbre_supported(SpectralTimbre timbre) {
    return (int)timbre <= OSC_GPU_MAX_TIMBRE;
}

#endif /* SPECTRAL_SYNTH_INTERNAL_H */
