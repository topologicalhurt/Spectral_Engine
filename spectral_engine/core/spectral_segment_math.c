/* spectral_segment_math.c - Out-of-line wrappers for canonical segment math */
#include "spectral_segment_math.h"

float spectral_segment_phase_at_index_f32(float phase0, float alpha, float beta, size_t sample_index) {
    return spectral_segment_phase_at_f32(phase0, alpha, beta, (float)sample_index);
}

float spectral_segment_amp_at_index_f32(float amp0, float d_amp, size_t sample_index) {
    return spectral_segment_amp_at_f32(amp0, d_amp, (float)sample_index);
}
