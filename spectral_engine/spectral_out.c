/* spectral_out.c - Audio Output Operations
 * 
 * Handles stereo interleaving, normalization, and audio file writing.
 * Platform-optimized implementations for both desktop and embedded.
 */

#include "spectral_io.h"
#include "spectral_q15.h"
#include <math.h>

#if SPECTRAL_HAS_FILE_IO
#include <sndfile.h>
#include <stdlib.h>
#endif

#if SPECTRAL_USE_VDSP
#include <Accelerate/Accelerate.h>
#endif

/*
 * Normalization
 */

float spectral_normalize_float(float* buffer, size_t len, float headroom) {
    if (!buffer || len == 0) return 0.0f;
    
    float max_amp = 0.0f;
    
#if SPECTRAL_USE_VDSP
    vDSP_maxmgv(buffer, 1, &max_amp, len);
    if (max_amp > 0.0f) {
        float scale = headroom / max_amp;
        vDSP_vsmul(buffer, 1, &scale, buffer, 1, len);
    }
#else
    for (size_t i = 0; i < len; i++) {
        float a = fabsf(buffer[i]);
        if (a > max_amp) max_amp = a;
    }
    if (max_amp > 0.0f) {
        float scale = headroom / max_amp;
        SPECTRAL_UNROLL_4
        for (size_t i = 0; i < len; i++) {
            buffer[i] *= scale;
        }
    }
#endif
    
    return max_amp;
}

q15_t spectral_normalize_q15(q15_t* buffer, size_t len, int* shift) {
    if (!buffer || len == 0) {
        if (shift) *shift = 0;
        return 0;
    }
    
    /* Find maximum absolute value */
    q15_t max_val = 0;
    for (size_t i = 0; i < len; i++) {
        q15_t abs_val = (buffer[i] < 0) ? -buffer[i] : buffer[i];
        if (abs_val > max_val) max_val = abs_val;
    }
    
    /* Determine if normalization needed (prevent clipping) */
    int shift_amt = 0;
    if (max_val > Q15_MAX / 2) {
        /* Need to scale down - find shift amount */
        q15_t test = max_val;
        while (test > Q15_MAX / 2) {
            test >>= 1;
            shift_amt++;
        }
        
        /* Apply shift */
        for (size_t i = 0; i < len; i++) {
            buffer[i] >>= shift_amt;
        }
    }
    
    if (shift) *shift = shift_amt;
    return max_val;
}

/*
 * Stereo Interleaving
 */

void spectral_mono_to_stereo_float(const float* mono, float* stereo, size_t num_frames) {
    if (!mono || !stereo || num_frames == 0) return;
    
#if SPECTRAL_USE_VDSP
    vDSP_vclr(stereo, 1, num_frames * 2);
    vDSP_vsadd(mono, 1, (float[]){0.0f}, stereo, 2, num_frames);
    vDSP_vsadd(mono, 1, (float[]){0.0f}, stereo + 1, 2, num_frames);
#else
    SPECTRAL_UNROLL_4
    for (size_t i = 0; i < num_frames; i++) {
        stereo[i * 2]     = mono[i];
        stereo[i * 2 + 1] = mono[i];
    }
#endif
}

void spectral_mono_to_stereo_q15(const q15_t* mono, q15_t* stereo, size_t num_frames) {
    if (!mono || !stereo || num_frames == 0) return;
    
#if SPECTRAL_ARM_M7 && defined(__ARM_FEATURE_DSP)
    size_t pairs = num_frames & ~1U;
    size_t i = 0;
    for (; i < pairs; i += 2) {
        stereo[i * 2]     = mono[i];
        stereo[i * 2 + 1] = mono[i];
        stereo[i * 2 + 2] = mono[i + 1];
        stereo[i * 2 + 3] = mono[i + 1];
    }
    if (i < num_frames) {
        stereo[i * 2]     = mono[i];
        stereo[i * 2 + 1] = mono[i];
    }
#else
    SPECTRAL_UNROLL_4
    for (size_t i = 0; i < num_frames; i++) {
        stereo[i * 2]     = mono[i];
        stereo[i * 2 + 1] = mono[i];
    }
#endif
}

/*
 * File Output (Desktop Only)
 */

#if SPECTRAL_HAS_FILE_IO

SpectralError spectral_audio_write(const char* path, const float* buffer, 
                         size_t num_frames, int sample_rate, int channels) {
    if (!path || !buffer || num_frames == 0 || sample_rate <= 0 || channels <= 0) {
        return SPECTRAL_ERR_PARAM;
    }
    
    SF_INFO info = {0};
    info.samplerate = sample_rate;
    info.frames = (sf_count_t)num_frames;
    info.channels = channels;
    info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    
    SNDFILE* file = sf_open(path, SFM_WRITE, &info);
    if (!file) return SPECTRAL_ERR_FILE;
    
    sf_count_t written = sf_writef_float(file, buffer, (sf_count_t)num_frames);
    sf_close(file);
    
    return (written == (sf_count_t)num_frames) ? SPECTRAL_OK : SPECTRAL_ERR_IO;
}

SpectralError spectral_audio_write_stereo(const char* path, const float* mono,
                                size_t num_frames, int sample_rate) {
    if (!path || !mono || num_frames == 0 || sample_rate <= 0) {
        return SPECTRAL_ERR_PARAM;
    }
    
    /* Allocate stereo buffer */
    float* stereo = malloc(num_frames * 2 * sizeof(float));
    if (!stereo) return SPECTRAL_ERR_MEMORY;
    
    /* Convert mono to stereo */
    spectral_mono_to_stereo_float(mono, stereo, num_frames);
    
    /* Write stereo file */
    SpectralError result = spectral_audio_write(path, stereo, num_frames, sample_rate, 2);
    
    free(stereo);
    return result;
}

#endif /* SPECTRAL_HAS_FILE_IO */
