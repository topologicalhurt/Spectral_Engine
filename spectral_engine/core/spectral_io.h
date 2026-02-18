/* spectral_io.h - Audio I/O Operations
 * 
 * Separates audio input/output concerns from synthesis:
 *   - spectral_in.c:  Audio file reading, format conversion, windowing
 *   - spectral_out.c: Stereo interleaving, normalization, audio file writing
 * 
 * Desktop builds use libsndfile for file I/O.
 * Embedded builds use direct buffer operations only.
 * 
 * SPECTRAL_HAS_FILE_IO: Defined when libsndfile is available
 *   - Desktop builds: always
 *   - Simulation builds: yes (runs on desktop)
 *   - Cross-compile ARM: no
 */
#ifndef SPECTRAL_IO_H
#define SPECTRAL_IO_H

#include "spectral_config.h"
#include "spectral_error.h"
#include "spectral_q15.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Determine if file I/O is available (desktop and simulation only) */
#if !SPECTRAL_EMBEDDED || defined(SPECTRAL_EMBEDDED_SIMULATION)
#define SPECTRAL_HAS_FILE_IO 1
#else
#define SPECTRAL_HAS_FILE_IO 0
#endif

/*
 * Audio Input (spectral_in.c)
 */

#if SPECTRAL_HAS_FILE_IO

typedef struct {
    int sample_rate;
    int channels;
    size_t frames;
} SpectralAudioInfo;

/* Read audio file to mono float. Caller must free *out_mono. */
SpectralError spectral_audio_read(const char* path, SpectralAudioInfo* info, float** out_mono);

/* Extract time window from audio buffer */
SpectralError spectral_audio_window(float* audio, size_t total_frames,
                          float start_sec, float end_sec, int sample_rate,
                          float** out_start, size_t* out_frames);

#endif /* SPECTRAL_HAS_FILE_IO */


/*
 * Audio Output (spectral_out.c)
 */

/* Normalize float buffer in-place. Returns original peak. */
float spectral_normalize_float(float* buffer, size_t len, float headroom);

void spectral_mono_to_stereo_float(const float* mono, float* stereo, size_t num_frames);
void spectral_mono_to_stereo_q15(const q15_t* mono, q15_t* stereo, size_t num_frames);

/* Normalize Q15 buffer. Returns max absolute value, sets *shift. */
q15_t spectral_normalize_q15(q15_t* buffer, size_t len, int* shift);

#if SPECTRAL_HAS_FILE_IO

SpectralError spectral_audio_write(const char* path, const float* buffer, 
                         size_t num_frames, int sample_rate, int channels);

/* Write mono as stereo (duplicates to L+R) */
SpectralError spectral_audio_write_stereo(const char* path, const float* mono,
                                size_t num_frames, int sample_rate);

#endif /* SPECTRAL_HAS_FILE_IO */

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_IO_H */
