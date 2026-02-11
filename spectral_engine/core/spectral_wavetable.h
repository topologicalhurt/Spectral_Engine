/* spectral_wavetable.h - Wavetable Synthesis Support
 * 
 * Provides LUT-based waveform storage and lookup for custom timbres.
 * Supports both float (desktop) and Q15 (embedded) sample formats.
 * 
 * File Formats:
 *   .spwt - Native format with header (auto-converts float<->Q15)
 *   .bin  - Raw samples in runtime format
 *   .hex  - Intel HEX text format
 * 
 * Lookup:
 *   Float phase (0.0-1.0) for desktop synthesis
 *   Fixed phase (0-65535) for embedded synthesis
 *   Linear interpolation between adjacent samples
 * 
 * The sample type is determined at compile time by SPECTRAL_EMBEDDED.
 */
#ifndef SPECTRAL_WAVETABLE_H
#define SPECTRAL_WAVETABLE_H

#include "spectral_config.h"
#include "spectral_error.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * .spwt file format (32-byte header):
 *   magic[4]="SPWT", version, format (0=float,1=Q15), size, timbre_id, reserved[15]
 * Followed by samples[size] in specified format. Conversion at load time.
 */

typedef enum {
    WAVETABLE_FORMAT_FLOAT = 0,
    WAVETABLE_FORMAT_Q15   = 1
} SpectralWavetableFormat;

typedef struct {
    char     magic[4];
    uint32_t version;
    uint32_t format;
    uint32_t size;
    uint8_t  timbre_id;
    uint8_t  reserved[15];
} SpectralWavetableHeader;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(SpectralWavetableHeader) == 32, "SpectralWavetableHeader must be 32 bytes");
#endif

#define SPECTRAL_WAVETABLE_MAGIC "SPWT"
#define SPECTRAL_WAVETABLE_VERSION 1

/* Wavetable structure using abstracted sample type */
typedef struct {
    spectral_sample_t samples[SPECTRAL_WAVETABLE_SIZE + 1];
    uint8_t valid;
    uint8_t timbre_id;
} SpectralWavetable;

typedef struct {
    SpectralWavetable tables[SPECTRAL_MAX_WAVETABLES];
    uint8_t num_loaded;
    uint8_t default_timbre;
} SpectralWavetableBank;

/* Initialization and builtin generation */
void spectral_wavetable_init(SpectralWavetableBank* bank);
void spectral_wavetable_generate_builtins(SpectralWavetableBank* bank);

/* Primary file loading - .spwt format with automatic conversion at load time */
WavetableError spectral_wavetable_load(SpectralWavetableBank* bank,
                                        const char* filename,
                                        uint8_t timbre_id);

/* Save wavetable to .spwt file in current runtime format */
WavetableError spectral_wavetable_save(const SpectralWavetableBank* bank,
                                        const char* filename,
                                        uint8_t timbre_id);

/* Legacy file loading (deprecated - prefer .spwt format) */
WavetableError spectral_wavetable_load_raw(SpectralWavetableBank* bank,
                                            const char* filename,
                                            uint8_t timbre_id);
WavetableError spectral_wavetable_load_hex(SpectralWavetableBank* bank,
                                            const char* filename,
                                            uint8_t timbre_id);

/* Buffer loading - accepts spectral_sample_t array */
WavetableError spectral_wavetable_load_buffer(SpectralWavetableBank* bank,
                                               const spectral_sample_t* data,
                                               size_t size,
                                               uint8_t timbre_id);

/* Accessors */
const SpectralWavetable* spectral_wavetable_get(const SpectralWavetableBank* bank,
                                                uint8_t timbre_id);
int spectral_wavetable_has_timbre(const SpectralWavetableBank* bank,
                                  uint8_t timbre_id);

/* Wavetable lookup - float phase (0.0-1.0) or fixed phase (0-65535) */
spectral_sample_t spectral_wavetable_lookup_f(const SpectralWavetable* table,
                                              float phase_norm);
spectral_sample_t spectral_wavetable_lookup_q(const SpectralWavetable* table,
                                              uint16_t phase_u16);
spectral_sample_t spectral_wavetable_lookup_timbre_f(const SpectralWavetableBank* bank,
                                                     uint8_t timbre_id,
                                                     float phase_norm);
spectral_sample_t spectral_wavetable_lookup_timbre_q(const SpectralWavetableBank* bank,
                                                     uint8_t timbre_id,
                                                     uint16_t phase_u16);

/* Backward-compatible aliases based on expected usage pattern */
#if SPECTRAL_EMBEDDED
#define spectral_wavetable_lookup(table, phase) spectral_wavetable_lookup_q(table, phase)
#define spectral_wavetable_lookup_timbre(bank, id, phase) spectral_wavetable_lookup_timbre_q(bank, id, phase)
#else
#define spectral_wavetable_lookup(table, phase) spectral_wavetable_lookup_f(table, phase)
#define spectral_wavetable_lookup_timbre(bank, id, phase) spectral_wavetable_lookup_timbre_f(bank, id, phase)
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_WAVETABLE_H */
