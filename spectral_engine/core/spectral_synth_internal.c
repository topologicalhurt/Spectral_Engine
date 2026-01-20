/* spectral_synth_internal.c - Shared Synthesis Helpers
 * 
 * Common validation and utilities shared across synthesis backends
 * (CPU, Metal, CUDA, emulator).
 */

#include "spectral_synth_internal.h"
#include "spectral_synth.h"
#include "spectral_utils.h"

/* Thread-safe dummy for null t_synth pointers */
static double g_synth_timing_dummy = 0;

/*
 * Validate synthesis inputs and prepare timing pointer.
 * 
 * This is the single source of truth for synth entry validation.
 * All synth backends should call this first.
 */
SynthValidateResult synth_validate_inputs(void* out_buffer, size_t out_len, size_t elem_size,
                                          SegmentArray sa, double** t_synth_ptr) {
    /* Handle null timing pointer */
    if (!*t_synth_ptr) {
        *t_synth_ptr = &g_synth_timing_dummy;
    }
    
    /* Empty output buffer - nothing to do */
    if (!out_buffer || out_len == 0) {
        **t_synth_ptr = 0;
        return SYNTH_VALIDATE_EARLY_EXIT;
    }
    
    /* Empty segment array - zero the buffer */
    if (sa.count == 0 || !sa.segs) {
        memset(out_buffer, 0, out_len * elem_size);
        **t_synth_ptr = 0;
        return SYNTH_VALIDATE_EARLY_EXIT;
    }
    
    return SYNTH_VALIDATE_OK;
}

/*
 * GPU timbre support check with automatic CPU fallback.
 */
int gpu_check_timbre_or_fallback(const char* backend_name,
                                  SegmentArray sa, float* out_buffer, size_t out_len,
                                  float stretch, float pitch, SpectralTimbre timbre, 
                                  double* t_synth) {
    (void)backend_name;  /* Used only in debug builds */
    
    if (!gpu_timbre_supported(timbre)) {
        SPECTRAL_DBG("%s: Timbre %d not supported on GPU (max %d), using CPU synthesis",
                     backend_name, (int)timbre, OSC_GPU_MAX_TIMBRE);
        synth_cpu(sa, out_buffer, out_len, stretch, pitch, timbre, 1, t_synth);
        return 0;  /* Used CPU fallback */
    }
    return 1;  /* GPU can handle this timbre */
}

SegmentLoopParams segment_loop_params_init(const Segment* s, const SynthParams* p, size_t out_len) {
    SegmentLoopParams lp;
    
    lp.start_idx = (size_t)(s->start * p->stretch);
    lp.length = (size_t)(s->length * p->stretch);
    
    if (lp.start_idx >= out_len) {
        lp.valid = 0;
        return lp;
    }
    if (lp.start_idx + lp.length > out_len) {
        lp.length = out_len - lp.start_idx;
    }
    
    lp.alpha = s->omega * p->pitch_factor * p->inv_stretch;
    lp.beta = s->df * p->pitch_factor * p->inv_stretch_sq;
    lp.d_amp = s->da * p->inv_stretch;
    lp.phase = s->phase;
    lp.amp = s->amp;
    lp.width = s->width;
    lp.valid = 1;
    
    return lp;
}
