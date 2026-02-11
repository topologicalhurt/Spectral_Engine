/* spectral_in.c - Audio Input Operations
 * 
 * Handles audio file reading and format conversion for desktop builds.
 * Uses libsndfile for multi-format support (WAV, FLAC, etc.)
 */

#include "spectral_io.h"

#if SPECTRAL_HAS_FILE_IO

#include <sndfile.h>
#include <stdlib.h>
#include <string.h>

SpectralError spectral_audio_read(const char* path, SpectralAudioInfo* info, float** out_mono) {
    if (!path || !info || !out_mono) return SPECTRAL_ERR_PARAM;
    
    SF_INFO sfinfo = {0};
    SNDFILE* file = sf_open(path, SFM_READ, &sfinfo);
    if (!file) return SPECTRAL_ERR_FILE_OPEN;

    if (sfinfo.frames <= 0 || sfinfo.channels <= 0) {
        sf_close(file);
        return SPECTRAL_ERR_FILE_FORMAT;
    }
    
    info->sample_rate = sfinfo.samplerate;
    info->channels = sfinfo.channels;
    info->frames = (size_t)sfinfo.frames;
    
    /* Read all audio data */
    if ((size_t)sfinfo.frames > SIZE_MAX / (size_t)sfinfo.channels) {
        sf_close(file);
        return SPECTRAL_ERR_OVERFLOW;
    }
    size_t total_samples = (size_t)sfinfo.frames * (size_t)sfinfo.channels;
    float* audio = malloc(total_samples * sizeof(float));
    if (!audio) {
        sf_close(file);
        return SPECTRAL_ERR_MEMORY;
    }
    
    sf_count_t read = sf_readf_float(file, audio, sfinfo.frames);
    sf_close(file);
    
    if (read != sfinfo.frames) {
        free(audio);
        return SPECTRAL_ERR_FILE_READ;
    }
    
    float* mono = malloc((size_t)sfinfo.frames * sizeof(float));
    if (!mono) {
        free(audio);
        return SPECTRAL_ERR_MEMORY;
    }

    if (sfinfo.channels == 1) {
        memcpy(mono, audio, (size_t)sfinfo.frames * sizeof(float));
    } else {
        float inv_ch = 1.0f / (float)sfinfo.channels;
        for (sf_count_t i = 0; i < sfinfo.frames; i++) {
            float sum = 0.0f;
            for (int ch = 0; ch < sfinfo.channels; ch++)
                sum += audio[i * sfinfo.channels + ch];
            mono[i] = sum * inv_ch;
        }
    }
    
    free(audio);
    *out_mono = mono;
    return SPECTRAL_OK;
}

SpectralError spectral_audio_window(float* audio, size_t total_frames,
                          float start_sec, float end_sec, int sample_rate,
                          float** out_start, size_t* out_frames) {
    if (!audio || !out_start || !out_frames || sample_rate <= 0) return SPECTRAL_ERR_PARAM;
    
    size_t start_frame = (start_sec > 0) ? (size_t)(start_sec * sample_rate) : 0;
    size_t end_frame = (end_sec < 0) ? total_frames : (size_t)(end_sec * sample_rate);
    
    /* Clamp to valid range */
    if (start_frame > total_frames) start_frame = total_frames;
    if (end_frame > total_frames) end_frame = total_frames;
    if (end_frame <= start_frame) return SPECTRAL_ERR_PARAM;
    
    *out_start = audio + start_frame;
    *out_frames = end_frame - start_frame;
    return SPECTRAL_OK;
}

#endif /* SPECTRAL_HAS_FILE_IO */
