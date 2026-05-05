#include "spectral_envelope.h"
#include "spectral_consts.h"
#include "spectral_fast_math.h"
#include <assert.h>

FadeParams fade_params_init(size_t segment_len, size_t max_fade) {
    FadeParams fp;
    
    if (segment_len < 2) {
        fp.fade_len = 1;
        fp.fade_out_start = 0;
        fp.inv_fade = 1.0f;
        return fp;
    }
    
    size_t fade_len = (segment_len >> 2) < max_fade ? (segment_len >> 2) : max_fade;
    if (fade_len < 1) fade_len = 1;
    
    fp.fade_len = fade_len;
    fp.fade_out_start = segment_len - fade_len;
    fp.inv_fade = 1.0f / (float)fade_len;
    
    return fp;
}

float fade_envelope_in(size_t j, float inv_fade) {
    return 0.5f * (1.0f - fast_sin(((float)j * inv_fade - 0.5f) * SPECTRAL_PI));
}

float fade_envelope_out(size_t j, size_t len, float inv_fade) {
    assert(j < len);
    return 0.5f * (1.0f - fast_sin(((float)(len - 1 - j) * inv_fade - 0.5f) * SPECTRAL_PI));
}

float fade_envelope(size_t j, const FadeParams* fp, size_t len) {
    if (j < fp->fade_len) {
        return fade_envelope_in(j, fp->inv_fade);
    } else if (j >= fp->fade_out_start) {
        return fade_envelope_out(j, len, fp->inv_fade);
    }
    return 1.0f;
}
