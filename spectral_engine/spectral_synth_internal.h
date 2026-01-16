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
#include "oscillator.h"

/*
 * Shared synthesis input validation
 * 
 * Validates common inputs and prepares timing pointer.
 * Returns 1 if synthesis should proceed, 0 if early return needed.
 * 
 * On return 0:
 *   - *t_synth_ptr is set to point to valid storage (dummy if was null)
 *   - **t_synth_ptr is set to 0
 *   - If out_buffer is valid and sa is empty, buffer is zeroed
 */
int synth_validate_and_prepare(float* out_buffer, size_t out_len,
                               SegmentArray sa, double** t_synth_ptr);

#endif /* SPECTRAL_SYNTH_INTERNAL_H */
