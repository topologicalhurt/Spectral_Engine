/* spectral_synth_internal.c - Shared Synthesis Helpers
 * 
 * Common validation and utilities shared across all synthesis backends
 * (CPU, Metal, CUDA, emulator).
 * 
 */

#include "spectral_synth_internal.h"
#include <string.h>

/*
 * Synthesis input validation result
 */
typedef enum {
    SYNTH_VALID_OK = 0,
    SYNTH_VALID_EARLY_EXIT,
    SYNTH_VALID_ZERO_FILL
} SynthValidResult;

/*
 * Validate common synthesis inputs and prepare t_synth pointer.
 * 
 * @param out_buffer  Output buffer pointer
 * @param out_len     Output length in samples
 * @param sa          Segment array to synthesize
 * @param t_synth     Timing output pointer (will be redirected if null)
 * @param dummy_t     Caller-provided dummy for null t_synth redirection
 * @return            Validation result indicating how to proceed
 */
SynthValidResult synth_validate_inputs(const void* out_buffer, size_t out_len,
                                       const SegmentArray* sa,
                                       double** t_synth, double* dummy_t) {
    /* Handle null timing pointer */
    if (!*t_synth) *t_synth = dummy_t;
    
    /* Check output buffer */
    if (!out_buffer || out_len == 0) {
        **t_synth = 0;
        return SYNTH_VALID_EARLY_EXIT;
    }
    
    /* Check segment array */
    if (!sa || sa->count == 0 || !sa->segs) {
        **t_synth = 0;
        return SYNTH_VALID_ZERO_FILL;
    }
    
    return SYNTH_VALID_OK;
}

/*
 * Macro for common synth entry validation (float output)
 * 
 * Usage at top of synth function:
 *   SYNTH_VALIDATE_FLOAT(out_buffer, out_len, sa, t_synth);
 * 
 * Will return early with proper handling if validation fails.
 */
#define SYNTH_VALIDATE_FLOAT(out_buffer, out_len, sa, t_synth) \
    do { \
        double _dummy_t = 0; \
        double* _t_ptr = (t_synth); \
        SynthValidResult _r = synth_validate_inputs((out_buffer), (out_len), &(sa), &_t_ptr, &_dummy_t); \
        if (_r == SYNTH_VALID_EARLY_EXIT) return; \
        if (_r == SYNTH_VALID_ZERO_FILL) { \
            memset((out_buffer), 0, (out_len) * sizeof(float)); \
            return; \
        } \
        (t_synth) = _t_ptr; \
    } while(0)

/*
 * Inline validation helper for backends that need explicit control.
 * Returns 1 if synthesis should proceed, 0 if early return needed.
 * 
 * On return 0:
 *   - t_synth is set to 0
 *   - If out_buffer is valid and sa is empty, buffer is zeroed
 */
int synth_validate_and_prepare(float* out_buffer, size_t out_len,
                               SegmentArray sa, double** t_synth_ptr) {
    static double dummy = 0;
    
    if (!*t_synth_ptr) *t_synth_ptr = &dummy;
    
    if (!out_buffer || out_len == 0) {
        **t_synth_ptr = 0;
        return 0;
    }
    
    if (sa.count == 0 || !sa.segs) {
        memset(out_buffer, 0, out_len * sizeof(float));
        **t_synth_ptr = 0;
        return 0;
    }
    
    return 1;
}

