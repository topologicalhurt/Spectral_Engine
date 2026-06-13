/* spectral_segment_convert.c - runtime float Segment -> Q15 conversion.
 * Contract and policy notes in the header. */
#include "spectral_segment_convert.h"

#include <stdint.h>

#include "spectral_contracts.h"
#include "spectral_utils.h"
#include "spectral_segment_math.h"

int spectral_segment_to_q15_runtime(const Segment* src, SpectralSegmentQ15* dst,
                                    float amp_scale, const SynthParams* params,
                                    size_t out_len) {
    double start_d = 0.0;
    double length_d = 0.0;
    float amp_scaled = 0.0f;
    float da_scaled = 0.0f;

    if (!src || !dst || !params || !spectral_segment_valid_for_synth(src) ||
        !spectral_is_finite_positive_f32(amp_scale) || out_len == 0u) {
        return 0;
    }
    *dst = (SpectralSegmentQ15){0};

    start_d = (double)src->start * (double)params->stretch;
    length_d = (double)src->length * (double)params->stretch;
    if (!spectral_is_finite_f64(start_d) || !spectral_is_finite_f64(length_d) ||
        start_d < 0.0 || start_d >= (double)out_len || start_d > (double)UINT32_MAX ||
        length_d <= 0.0) {
        return 0;
    }
    if (length_d > 65535.0) length_d = 65535.0;

    dst->start = (uint32_t)start_d;
    dst->length = (uint16_t)length_d;
    if (dst->length == 0u) return 0;

    dst->freq_q88 = OMEGA_TO_Q88(
        spectral_segment_alpha_f32(src->omega, params->pitch_factor, params->inv_stretch));
    dst->phase_q15 = PHASE_RAD_TO_Q15(src->phase);

    amp_scaled = spectral_clamp_f32(src->amp * amp_scale, 0.0f, 1.0f);
    dst->amp_q15 = FLOAT_TO_Q15(amp_scaled);

    da_scaled = spectral_segment_d_amp_f32(src->da, params->inv_stretch) * amp_scale;
    dst->da_q15 = FLOAT_TO_Q15(spectral_clamp_f32(da_scaled, -1.0f, 1.0f));
    /* df_q15 stays 0 (zero-initialized above): chirp is intentionally dropped. */
    return 1;
}
