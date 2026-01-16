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
 *   - Emulator builds: yes (runs on desktop)
 *   - Cross-compile ARM: no
 */
#ifndef SPECTRAL_IO_H
#define SPECTRAL_IO_H

#include "spectral_config.h"
#include "spectral_q15.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Determine if we have file I/O available
 * Available on desktop and emulator builds, not on cross-compiled ARM */
#if !SPECTRAL_EMBEDDED || defined(SPECTRAL_EMBEDDED_EMULATION)
#define SPECTRAL_HAS_FILE_IO 1
#else
#define SPECTRAL_HAS_FILE_IO 0
#endif

/*
 * Audio Input (spectral_in.c)
 */

#if SPECTRAL_HAS_FILE_IO

/* Audio file info returned from spectral_audio_read */
typedef struct {
    int sample_rate;
    int channels;
    size_t frames;
} SpectralAudioInfo;

/* Read audio file and convert to mono float
 * 
 * Parameters:
 *   path      - Path to audio file (WAV, FLAC, etc.)
 *   info      - Output: audio file metadata
 *   out_mono  - Output: pointer to allocated mono buffer (caller must free)
 * 
 * Returns: SPECTRAL_OK on success, negative SpectralError on failure
 */
SpectralError spectral_audio_read(const char* path, SpectralAudioInfo* info, float** out_mono);

/* Apply time window to audio buffer
 * 
 * Parameters:
 *   audio       - Input audio buffer
 *   total_frames - Total frames in buffer
 *   start_sec   - Start time in seconds (0 = beginning)
 *   end_sec     - End time in seconds (negative = end of file)
 *   sample_rate - Audio sample rate
 *   out_start   - Output: pointer to windowed start
 *   out_frames  - Output: number of frames in window
 * 
 * Returns: SPECTRAL_OK on success, SPECTRAL_ERR_PARAM if window invalid
 */
SpectralError spectral_audio_window(float* audio, size_t total_frames,
                          float start_sec, float end_sec, int sample_rate,
                          float** out_start, size_t* out_frames);

#endif /* SPECTRAL_HAS_FILE_IO */


/*
 * Audio Output (spectral_out.c)
 */

/* Normalize float buffer to specified headroom
 * 
 * Parameters:
 *   buffer   - Float buffer to normalize in-place
 *   len      - Number of samples
 *   headroom - Target peak level (e.g., 0.95 for -0.5dB headroom)
 * 
 * Returns: Original peak amplitude before normalization
 */
float spectral_normalize_float(float* buffer, size_t len, float headroom);

/* Convert mono float to stereo interleaved float
 * 
 * Parameters:
 *   mono       - Input mono buffer
 *   stereo     - Output stereo buffer (must be 2x mono size)
 *   num_frames - Number of mono frames
 */
void spectral_mono_to_stereo_float(const float* mono, float* stereo, size_t num_frames);

/* Convert mono Q15 to stereo interleaved Q15
 * 
 * Parameters:
 *   mono       - Input mono buffer
 *   stereo     - Output stereo buffer (must be 2x mono size)
 *   num_frames - Number of mono frames
 * 
 * Note: Optimized for ARM Cortex-M7 when available
 */
void spectral_mono_to_stereo_q15(const q15_t* mono, q15_t* stereo, size_t num_frames);

/* Normalize Q15 buffer to prevent clipping
 * 
 * Parameters:
 *   buffer   - Q15 buffer to analyze/normalize in-place
 *   len      - Number of samples
 *   shift    - Output: right-shift applied (0 if no normalization needed)
 * 
 * Returns: Maximum absolute sample value found
 */
q15_t spectral_normalize_q15(q15_t* buffer, size_t len, int* shift);

#if SPECTRAL_HAS_FILE_IO

/* Write float buffer to audio file (mono)
 * 
 * Parameters:
 *   path        - Output file path
 *   buffer      - Float audio buffer
 *   num_frames  - Number of frames
 *   sample_rate - Output sample rate
 *   channels    - Number of channels (1 = mono, 2 = stereo)
 * 
 * Returns: SPECTRAL_OK on success, negative SpectralError on failure
 */
SpectralError spectral_audio_write(const char* path, const float* buffer, 
                         size_t num_frames, int sample_rate, int channels);

/* Write float buffer to audio file as stereo (duplicates mono to L+R)
 * 
 * Parameters:
 *   path        - Output file path
 *   mono        - Mono float audio buffer
 *   num_frames  - Number of mono frames
 *   sample_rate - Output sample rate
 * 
 * Returns: SPECTRAL_OK on success, negative SpectralError on failure
 */
SpectralError spectral_audio_write_stereo(const char* path, const float* mono,
                                size_t num_frames, int sample_rate);

#endif /* SPECTRAL_HAS_FILE_IO */

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_IO_H */
