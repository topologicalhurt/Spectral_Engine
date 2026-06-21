/* spectral_in.c - Audio Input Operations
 * 
 * Handles audio file reading and format conversion for desktop builds.
 * Uses libsndfile for multi-format support (WAV, FLAC, etc.)
 */

#include "spectral_io.h"
#include "spectral_utils.h"
#include "spectral_contracts.h"

#if SPECTRAL_HAS_FILE_IO

#include <sndfile.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

static int spectral_sf_count_to_size(sf_count_t value, size_t* out)
{
    size_t narrowed = 0;

    if (!out || value < 0) return 0;

    narrowed = (size_t)value;
    if ((sf_count_t)narrowed != value) {
        return 0;
    }

    *out = narrowed;
    return 1;
}

SpectralError spectral_audio_read(const char* path, SpectralAudioInfo* info, float** out_mono) {
    SF_INFO sfinfo = {0};
    SNDFILE* file = NULL;
    size_t frames = 0;
    size_t channels = 0;
    size_t total_samples = 0;
    size_t mono_bytes = 0;
    float* audio = NULL;
    float* mono = NULL;
    sf_count_t read = 0;

    if (spectral_is_empty_string(path) || !info || !out_mono) return SPECTRAL_ERR_PARAM;
    *info = (SpectralAudioInfo){0};
    *out_mono = NULL;

    file = sf_open(path, SFM_READ, &sfinfo);
    if (!file) return SPECTRAL_ERR_FILE_OPEN;

    if (sfinfo.frames <= 0 || sfinfo.channels <= 0 ||
        sfinfo.samplerate < SPECTRAL_MIN_SAMPLE_RATE ||
        sfinfo.samplerate > SPECTRAL_MAX_SAMPLE_RATE) {
        sf_close(file);
        return SPECTRAL_ERR_FILE_FORMAT;
    }

    /* libsndfile reports frame counts as sf_count_t, but allocation/indexing
     * uses size_t.  Prove representability before allocating a buffer that
     * sf_readf_float() will fill using the original sf_count_t frame count. */
    if (!spectral_sf_count_to_size(sfinfo.frames, &frames)) {
        sf_close(file);
        return SPECTRAL_ERR_OVERFLOW;
    }
    channels = (size_t)sfinfo.channels;

    /* Read all audio data. */
    if (!spectral_size_mul(frames, channels, &total_samples) ||
        !spectral_array_bytes(frames, sizeof(float), &mono_bytes)) {
        sf_close(file);
        return SPECTRAL_ERR_OVERFLOW;
    }

    audio = (float*)spectral_malloc_array(total_samples, sizeof(float));
    if (!audio) {
        sf_close(file);
        return SPECTRAL_ERR_MEMORY;
    }

    read = sf_readf_float(file, audio, sfinfo.frames);
    sf_close(file);

    if (read != sfinfo.frames) {
        free(audio);
        return SPECTRAL_ERR_FILE_READ;
    }
    if (!spectral_f32_span_finite(audio, total_samples)) {
        free(audio);
        return SPECTRAL_ERR_FILE_FORMAT;
    }

    mono = (float*)spectral_malloc_array(frames, sizeof(float));
    if (!mono) {
        free(audio);
        return SPECTRAL_ERR_MEMORY;
    }

    if (sfinfo.channels == 1) {
        memcpy(mono, audio, mono_bytes);
    } else {
        const double inv_ch = 1.0 / (double)channels;
        for (size_t i = 0; i < frames; i++) {
            double sum = 0.0;
            double mono_d = 0.0;
            size_t base = i * channels;

            for (size_t ch = 0; ch < channels; ch++) {
                sum += (double)audio[base + ch];
            }

            mono_d = sum * inv_ch;
            if (!spectral_is_finite_f64(mono_d) ||
                mono_d < -(double)FLT_MAX ||
                mono_d > (double)FLT_MAX) {
                free(mono);
                free(audio);
                return SPECTRAL_ERR_FILE_FORMAT;
            }
            mono[i] = (float)mono_d;
        }
    }

    free(audio);

    info->sample_rate = sfinfo.samplerate;
    info->channels = sfinfo.channels;
    info->frames = frames;
    *out_mono = mono;
    return SPECTRAL_OK;
}



SpectralError spectral_audio_window(float* audio, size_t total_frames,
                          float start_sec, float end_sec, int sample_rate,
                          float** out_start, size_t* out_frames) {
    double start_d = 0.0;
    double end_d = 0.0;
    size_t start_frame = 0;
    size_t end_frame = 0;

    if (!audio || !out_start || !out_frames) return SPECTRAL_ERR_PARAM;
    *out_start = NULL;
    *out_frames = 0;

    if (total_frames == 0 ||
        sample_rate < SPECTRAL_MIN_SAMPLE_RATE ||
        sample_rate > SPECTRAL_MAX_SAMPLE_RATE) {
        return SPECTRAL_ERR_PARAM;
    }
    if (!spectral_is_finite_f32(start_sec)) {
        return SPECTRAL_ERR_PARAM;
    }
    if (end_sec >= 0.0f && !spectral_is_finite_f32(end_sec)) {
        return SPECTRAL_ERR_PARAM;
    }

    start_d = (start_sec > 0.0f) ? (double)start_sec * (double)sample_rate : 0.0;
    end_d = (end_sec < 0.0f) ? (double)total_frames : (double)end_sec * (double)sample_rate;

    if (!spectral_is_finite_f64(start_d) || !spectral_is_finite_f64(end_d)) return SPECTRAL_ERR_PARAM;
    if (start_d < 0.0) start_d = 0.0;
    if (end_d < 0.0) end_d = 0.0;
    if (start_d > (double)total_frames) start_d = (double)total_frames;
    if (end_d > (double)total_frames) end_d = (double)total_frames;

    start_frame = (size_t)start_d;
    end_frame = (size_t)end_d;

    /* Clamp to valid range */
    if (start_frame > total_frames) start_frame = total_frames;
    if (end_frame > total_frames) end_frame = total_frames;
    if (end_frame <= start_frame) return SPECTRAL_ERR_PARAM;

    *out_start = audio + start_frame;
    *out_frames = end_frame - start_frame;
    return SPECTRAL_OK;
}


#endif /* SPECTRAL_HAS_FILE_IO */
