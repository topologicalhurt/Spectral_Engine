/* spectral_out.c - Audio Output Operations
 * 
 * Handles stereo interleaving, normalization, and audio file writing.
 * Platform-optimized implementations for both desktop and embedded.
 */

#include "spectral_io.h"
#include "spectral_q15.h"
#include "spectral_utils.h"
#include "spectral_vector_ops.h"
#include <math.h>

#if SPECTRAL_HAS_FILE_IO
#include <sndfile.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#endif

#if SPECTRAL_USE_CMSIS
#include "arm_math.h"
#endif

/*
 * Normalization
 */

float spectral_normalize_float(float* buffer, size_t len, float headroom) {
    if (!buffer || len == 0) return 0.0f;

    float max_amp = 0.0f;

#if (!SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM) && !SPECTRAL_USE_CMSIS
    spectral_vmaxmgv(buffer, &max_amp, len);
    if (max_amp > 0.0f) {
        float scale = headroom / max_amp;
        spectral_vsmul(buffer, scale, buffer, len);
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
    
#if SPECTRAL_USE_CMSIS
    uint32_t max_idx;
    arm_absmax_q15(buffer, (uint32_t)len, &max_val, &max_idx);
#else
    for (size_t i = 0; i < len; i++) {
        q15_t abs_val;
        if (buffer[i] == Q15_MIN) {
            abs_val = Q15_MAX;
        } else {
            abs_val = (buffer[i] < 0) ? (q15_t)-buffer[i] : buffer[i];
        }
        if (abs_val > max_val) max_val = abs_val;
    }
#endif
    
    /* Determine if normalization needed (prevent clipping) */
    int shift_amt = 0;
    if (max_val > Q15_MAX / 2) {
        /* Need to scale down - find shift amount */
        q15_t test = max_val;
        while (test > Q15_MAX / 2) {
            test >>= 1;
            shift_amt++;
        }
        
        /* Apply shift - CMSIS arm_shift_q15 uses SIMD on M7 */
#if SPECTRAL_USE_CMSIS
        arm_shift_q15(buffer, -shift_amt, buffer, (uint32_t)len);
#else
        for (size_t i = 0; i < len; i++) {
            buffer[i] >>= shift_amt;
        }
#endif
    }
    
    if (shift) *shift = shift_amt;
    return max_val;
}

/*
 * Stereo Interleaving
 */

void spectral_mono_to_stereo_float(const float* mono, float* stereo, size_t num_frames) {
    if (!mono || !stereo || num_frames == 0) return;

    SPECTRAL_UNROLL_4
    for (size_t i = 0; i < num_frames; i++) {
        stereo[i * 2]     = mono[i];
        stereo[i * 2 + 1] = mono[i];
    }
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

/* libsndfile writes a PEAK chunk with a run-time timestamp for float WAV.
 * Scrub the timestamp field so rendered artifacts are byte-for-byte reproducible. */
static void spectral_wav_scrub_peak_timestamp(const char* path) {
    FILE* f;
    unsigned char hdr[12];
    if (spectral_is_empty_string(path)) return;

    f = fopen(path, "r+b");
    if (!f) return;

    if (fread(hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        fclose(f);
        return;
    }
    if (memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) {
        fclose(f);
        return;
    }

    for (;;) {
        unsigned char chunk[8];
        long data_pos;
        unsigned int size;
        if (fread(chunk, 1, sizeof(chunk), f) != sizeof(chunk)) break;

        data_pos = ftell(f);
        size = (unsigned int)chunk[4]
             | ((unsigned int)chunk[5] << 8)
             | ((unsigned int)chunk[6] << 16)
             | ((unsigned int)chunk[7] << 24);

        if (memcmp(chunk, "PEAK", 4) == 0 && size >= 8) {
            unsigned char zero4[4] = {0, 0, 0, 0};
            if (fseek(f, data_pos + 4, SEEK_SET) == 0) {
                (void)fwrite(zero4, 1, sizeof(zero4), f);
            }
            break;
        }

        if (fseek(f, data_pos + (long)size + (long)(size & 1u), SEEK_SET) != 0) {
            break;
        }
    }

    fclose(f);
}

SpectralError spectral_audio_write(const char* path, const float* buffer,
                         size_t num_frames, int sample_rate, int channels) {
    if (spectral_is_empty_string(path) || !buffer || num_frames == 0 || sample_rate <= 0 || channels <= 0) {
        return SPECTRAL_ERR_PARAM;
    }
    
    SF_INFO info = {0};
    info.samplerate = sample_rate;
    info.frames = (sf_count_t)num_frames;
    info.channels = channels;
    info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

#ifdef SFC_SET_ADD_PEAK_CHUNK
    /* Configure libsndfile globally before open to avoid non-deterministic PEAK timestamps. */
    int add_peak_chunk = SF_FALSE;
    (void)sf_command(NULL, SFC_SET_ADD_PEAK_CHUNK, &add_peak_chunk, (int)sizeof(add_peak_chunk));
#endif
    
    SNDFILE* file = sf_open(path, SFM_WRITE, &info);
    if (!file) return SPECTRAL_ERR_FILE_OPEN;
    
    sf_count_t written = sf_writef_float(file, buffer, (sf_count_t)num_frames);
    sf_close(file);

    if (written == (sf_count_t)num_frames) {
        spectral_wav_scrub_peak_timestamp(path);
    }
    
    return (written == (sf_count_t)num_frames) ? SPECTRAL_OK : SPECTRAL_ERR_FILE_WRITE;
}

SpectralError spectral_audio_write_stereo(const char* path, const float* mono,
                                size_t num_frames, int sample_rate) {
    if (spectral_is_empty_string(path) || !mono || num_frames == 0 || sample_rate <= 0) {
        return SPECTRAL_ERR_PARAM;
    }

    size_t stereo_samples = 0;
    size_t stereo_bytes = 0;
    if (!spectral_size_mul(num_frames, 2u, &stereo_samples) ||
        !spectral_size_mul(stereo_samples, sizeof(float), &stereo_bytes)) {
        return SPECTRAL_ERR_OVERFLOW;
    }
    
    float* stereo = malloc(stereo_bytes);
    if (!stereo) return SPECTRAL_ERR_MEMORY;

    spectral_mono_to_stereo_float(mono, stereo, num_frames);

    SpectralError result = spectral_audio_write(path, stereo, num_frames, sample_rate, 2);
    
    free(stereo);
    return result;
}

#endif /* SPECTRAL_HAS_FILE_IO */
