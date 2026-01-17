#ifndef SPECTRAL_ENVELOPE_H
#define SPECTRAL_ENVELOPE_H

#include <stddef.h>

#define FADE_SAMPLES_DEFAULT 64

/* Fade envelope parameters for segment synthesis */
typedef struct {
    size_t fade_len;
    size_t fade_out_start;
    float inv_fade;
} FadeParams;

FadeParams fade_params_init(size_t segment_len, size_t max_fade);

float fade_envelope_in(size_t j, float inv_fade);
float fade_envelope_out(size_t j, size_t len, float inv_fade);
float fade_envelope(size_t j, const FadeParams* fp, size_t len);

#endif
