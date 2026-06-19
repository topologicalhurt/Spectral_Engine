/* spectral_wav.h - minimal 16-bit PCM (Q15) WAV writer.
 *
 * Q15 audio IS 16-bit signed little-endian PCM, so writing a q15 buffer to a .wav is just a
 * canonical 44-byte header followed by the raw samples -- no float round-trip, no resampling:
 * exactly the bytes the embedded/Q15 path produced. Two layers:
 *   - spectral_wav_pcm16_header(): builds the 44-byte header into a caller buffer. Pure byte
 *     stores, no libc / no stdio, so it works in a freestanding/embedded build -- e.g. the
 *     QEMU audio-dump harness writes [header + samples] to a host file over semihosting.
 *   - spectral_write_wav_q15(): host convenience that fopen/fwrites header + samples; compiled
 *     only where the host file API exists (SPECTRAL_HASH_HAS_HOST_FILE_API).
 *
 * This is distinct from spectral_io.h's spectral_audio_write(), which writes a FLOAT buffer via
 * libsndfile (host-only): that path is for the desktop float engine; this one is the libc-free
 * Q15/PCM16 path shared by host and embedded.
 *
 * Endianness: WAV is little-endian; the header is emitted LE explicitly. The sample data is the
 * raw int16 in native order, which is correct on every target this engine runs (x86, arm64,
 * Cortex-M -- all little-endian). A big-endian host would need per-sample byte swaps and is not
 * a supported target.
 */
#ifndef SPECTRAL_WAV_H
#define SPECTRAL_WAV_H

#include <stdint.h>
#include <stddef.h>
#include "spectral_config.h"   /* SPECTRAL_HASH_HAS_HOST_FILE_API */

#define SPECTRAL_WAV_HEADER_BYTES 44u

static inline void spectral_wav_put_u16le(uint8_t* p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
}
static inline void spectral_wav_put_u32le(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)((v >> 24) & 0xFFu);
}

/* Fill out[0..43] with a canonical 16-bit-PCM WAV header for num_frames frames at sample_rate,
 * `channels` interleaved. The audio data that follows is num_frames*channels int16 samples. */
static inline void spectral_wav_pcm16_header(uint8_t out[SPECTRAL_WAV_HEADER_BYTES],
                                             uint32_t num_frames, uint32_t sample_rate,
                                             uint16_t channels) {
    uint32_t data_bytes  = num_frames * (uint32_t)channels * 2u;
    uint32_t byte_rate   = sample_rate * (uint32_t)channels * 2u;
    uint16_t block_align = (uint16_t)(channels * 2u);

    out[0] = 'R'; out[1] = 'I'; out[2] = 'F'; out[3] = 'F';
    spectral_wav_put_u32le(out + 4, 36u + data_bytes);  /* RIFF chunk size = 36 + data */
    out[8] = 'W'; out[9] = 'A'; out[10] = 'V'; out[11] = 'E';

    out[12] = 'f'; out[13] = 'm'; out[14] = 't'; out[15] = ' ';
    spectral_wav_put_u32le(out + 16, 16u);   /* fmt chunk size (PCM) */
    spectral_wav_put_u16le(out + 20, 1u);    /* audio format: 1 = PCM */
    spectral_wav_put_u16le(out + 22, channels);
    spectral_wav_put_u32le(out + 24, sample_rate);
    spectral_wav_put_u32le(out + 28, byte_rate);
    spectral_wav_put_u16le(out + 32, block_align);
    spectral_wav_put_u16le(out + 34, 16u);   /* bits per sample */

    out[36] = 'd'; out[37] = 'a'; out[38] = 't'; out[39] = 'a';
    spectral_wav_put_u32le(out + 40, data_bytes);
}

#if SPECTRAL_HASH_HAS_HOST_FILE_API
#include <stdio.h>
/* Host: write a Q15 (int16) buffer to a 16-bit-PCM WAV. num_frames is per-channel frame count;
 * `samples` is num_frames*channels interleaved int16. Returns 1 on success, 0 on any I/O error. */
static inline int spectral_write_wav_q15(const char* path, const int16_t* samples,
                                         uint32_t num_frames, uint32_t sample_rate,
                                         uint16_t channels) {
    if (!path || (num_frames && !samples) || channels == 0u) return 0;
    uint8_t hdr[SPECTRAL_WAV_HEADER_BYTES];
    spectral_wav_pcm16_header(hdr, num_frames, sample_rate, channels);
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    size_t n = (size_t)num_frames * channels;
    int ok = (fwrite(hdr, 1u, SPECTRAL_WAV_HEADER_BYTES, f) == SPECTRAL_WAV_HEADER_BYTES) &&
             (n == 0u || fwrite(samples, sizeof(int16_t), n, f) == n);
    return (fclose(f) == 0) && ok;
}
#endif /* SPECTRAL_HASH_HAS_HOST_FILE_API */

#endif /* SPECTRAL_WAV_H */
