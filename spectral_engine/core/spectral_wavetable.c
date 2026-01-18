/* spectral_wavetable.c - Wavetable Implementation
 * 
 * Wavetable Management:
 *   - Bank holds up to SPECTRAL_MAX_WAVETABLES (default 8)
 *   - Built-in generation for sine, saw, square, triangle
 *   - File loading with automatic format conversion
 * 
 * .spwt File Format:
 *   Header (32 bytes): magic "SPWT", version, format, size, timbre_id
 *   Body: samples in specified format (float or Q15)
 *   On load, converts to runtime format if different
 * 
 * Lookup Implementation:
 *   - Table size is SPECTRAL_WAVETABLE_SIZE + 1 (for wraparound)
 *   - Linear interpolation between samples
 *   - Phase normalized to [0, 1) for float, [0, 65535] for fixed
 */
#include "spectral_wavetable.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if !SPECTRAL_EMBEDDED
#include <stdio.h>
#endif

static void wavetable_mark_loaded(SpectralWavetableBank* bank,
                                  SpectralWavetable* table,
                                  uint8_t timbre_id) {
    int was_valid = table->valid;
    table->valid = 1;
    table->timbre_id = timbre_id;
    if (!was_valid) {
        bank->num_loaded++;
    }
}

void spectral_wavetable_init(SpectralWavetableBank* bank) {
    if (!bank) return;
    memset(bank, 0, sizeof(SpectralWavetableBank));
    bank->default_timbre = TIMBRE_SINE;
}

void spectral_wavetable_generate_builtins(SpectralWavetableBank* bank) {
    if (!bank) return;
    
    /* Sine */
    SpectralWavetable* sine = &bank->tables[TIMBRE_SINE];
    for (size_t i = 0; i < SPECTRAL_WAVETABLE_SIZE; i++) {
        double phase = 2.0 * SPECTRAL_PI * (double)i / (double)SPECTRAL_WAVETABLE_SIZE;
        double val = sin(phase);
        sine->samples[i] = FLOAT_TO_SPECTRAL_SAMPLE((float)val);
    }
    sine->samples[SPECTRAL_WAVETABLE_SIZE] = sine->samples[0];
    wavetable_mark_loaded(bank, sine, TIMBRE_SINE);
    
    /* Sawtooth */
    SpectralWavetable* saw = &bank->tables[TIMBRE_SAW];
    for (size_t i = 0; i < SPECTRAL_WAVETABLE_SIZE; i++) {
        double t = (double)i / (double)SPECTRAL_WAVETABLE_SIZE;
        double val = 2.0 * t - 1.0;
        saw->samples[i] = FLOAT_TO_SPECTRAL_SAMPLE((float)val);
    }
    saw->samples[SPECTRAL_WAVETABLE_SIZE] = saw->samples[0];
    wavetable_mark_loaded(bank, saw, TIMBRE_SAW);
    
    /* Square */
    SpectralWavetable* square = &bank->tables[TIMBRE_SQUARE];
    for (size_t i = 0; i < SPECTRAL_WAVETABLE_SIZE; i++) {
        float val = (i < SPECTRAL_WAVETABLE_SIZE / 2) ? 1.0f : -1.0f;
        square->samples[i] = FLOAT_TO_SPECTRAL_SAMPLE(val);
    }
    square->samples[SPECTRAL_WAVETABLE_SIZE] = square->samples[0];
    wavetable_mark_loaded(bank, square, TIMBRE_SQUARE);
    
    /* Triangle */
    SpectralWavetable* tri = &bank->tables[TIMBRE_TRIANGLE];
    for (size_t i = 0; i < SPECTRAL_WAVETABLE_SIZE; i++) {
        double t = (double)i / (double)SPECTRAL_WAVETABLE_SIZE;
        double val;
        if (t < 0.25) val = 4.0 * t;
        else if (t < 0.75) val = 2.0 - 4.0 * t;
        else val = 4.0 * t - 4.0;
        tri->samples[i] = FLOAT_TO_SPECTRAL_SAMPLE((float)val);
    }
    tri->samples[SPECTRAL_WAVETABLE_SIZE] = tri->samples[0];
    wavetable_mark_loaded(bank, tri, TIMBRE_TRIANGLE);
}

#if !SPECTRAL_EMBEDDED

WavetableResult spectral_wavetable_load_raw(SpectralWavetableBank* bank,
                                            const char* filename,
                                            uint8_t timbre_id) {
    if (!bank || !filename) return WAVETABLE_ERR_PARAM;
    if (timbre_id >= SPECTRAL_MAX_WAVETABLES) return WAVETABLE_ERR_PARAM;
    
    FILE* f = fopen(filename, "rb");
    if (!f) return WAVETABLE_ERR_FILE;
    
    SpectralWavetable* table = &bank->tables[timbre_id];

    spectral_sample_t* temp = (spectral_sample_t*)malloc(SPECTRAL_WAVETABLE_SIZE * sizeof(spectral_sample_t));
    if (!temp) {
        fclose(f);
        return WAVETABLE_ERR_MEMORY;
    }

    /* Read samples in native format (float on desktop) */
    size_t read = fread(temp, sizeof(spectral_sample_t), SPECTRAL_WAVETABLE_SIZE, f);
    fclose(f);

    if (read != SPECTRAL_WAVETABLE_SIZE) {
        free(temp);
        return WAVETABLE_ERR_SIZE;
    }

    memcpy(table->samples, temp, SPECTRAL_WAVETABLE_SIZE * sizeof(spectral_sample_t));
    free(temp);
    
    table->samples[SPECTRAL_WAVETABLE_SIZE] = table->samples[0];
    wavetable_mark_loaded(bank, table, timbre_id);
    
    return WAVETABLE_OK;
}

/* Parse HEX format - reads bytes and interprets as spectral_sample_t */
static WavetableResult parse_hex_line(const char* line, uint8_t* data, size_t data_capacity,
                                      size_t* data_len, uint16_t* address, uint8_t* record_type) {
    if (!line || line[0] != ':') return WAVETABLE_ERR_FORMAT;
    
    uint8_t byte_count;
    uint8_t checksum_calc = 0;
    
    if (sscanf(line + 1, "%2hhx", &byte_count) != 1) return WAVETABLE_ERR_FORMAT;
    if (sscanf(line + 3, "%4hx", address) != 1) return WAVETABLE_ERR_FORMAT;
    if (sscanf(line + 7, "%2hhx", record_type) != 1) return WAVETABLE_ERR_FORMAT;
    if (byte_count > data_capacity) return WAVETABLE_ERR_SIZE;
    
    checksum_calc += byte_count;
    checksum_calc += (*address >> 8) & 0xFF;
    checksum_calc += *address & 0xFF;
    checksum_calc += *record_type;
    
    *data_len = byte_count;
    for (size_t i = 0; i < byte_count; i++) {
        if (sscanf(line + 9 + i * 2, "%2hhx", &data[i]) != 1) return WAVETABLE_ERR_FORMAT;
        checksum_calc += data[i];
    }
    
    uint8_t checksum_read;
    if (sscanf(line + 9 + byte_count * 2, "%2hhx", &checksum_read) != 1) return WAVETABLE_ERR_FORMAT;
    if ((uint8_t)(checksum_calc + checksum_read) != 0) return WAVETABLE_ERR_FORMAT;
    
    return WAVETABLE_OK;
}

WavetableResult spectral_wavetable_load_hex(SpectralWavetableBank* bank,
                                            const char* filename,
                                            uint8_t timbre_id) {
    if (!bank || !filename) return WAVETABLE_ERR_PARAM;
    if (timbre_id >= SPECTRAL_MAX_WAVETABLES) return WAVETABLE_ERR_PARAM;
    
    FILE* f = fopen(filename, "r");
    if (!f) return WAVETABLE_ERR_FILE;
    
    SpectralWavetable* table = &bank->tables[timbre_id];
    spectral_sample_t* temp_table = (spectral_sample_t*)calloc(SPECTRAL_WAVETABLE_SIZE + 1, sizeof(spectral_sample_t));
    if (!temp_table) {
        fclose(f);
        return WAVETABLE_ERR_MEMORY;
    }
    
    char line[256];
    uint8_t data[32];
    size_t data_len;
    uint16_t address;
    uint8_t record_type;
    size_t total_bytes = 0;
    const size_t expected_bytes = SPECTRAL_WAVETABLE_SIZE * sizeof(spectral_sample_t);
    
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != ':') continue;
        
        WavetableResult parse_result = parse_hex_line(line, data, sizeof(data), &data_len, &address, &record_type);
        if (parse_result != WAVETABLE_OK) {
            fclose(f);
            free(temp_table);
            return parse_result;
        }
        
        if (record_type == 0x00) {
            if (address + data_len > expected_bytes) {
                fclose(f);
                free(temp_table);
                return WAVETABLE_ERR_SIZE;
            }
            memcpy((uint8_t*)temp_table + address, data, data_len);
            total_bytes += data_len;
        } else if (record_type == 0x01) {
            break;
        } else {
            fclose(f);
            free(temp_table);
            return WAVETABLE_ERR_FORMAT;
        }
    }
    
    fclose(f);
    
    if (total_bytes < expected_bytes) {
        free(temp_table);
        return WAVETABLE_ERR_SIZE;
    }

    memcpy(table->samples, temp_table, SPECTRAL_WAVETABLE_SIZE * sizeof(spectral_sample_t));
    free(temp_table);
    
    table->samples[SPECTRAL_WAVETABLE_SIZE] = table->samples[0];
    wavetable_mark_loaded(bank, table, timbre_id);
    
    return WAVETABLE_OK;
}

/* .spwt format - reads header and converts to runtime spectral_sample_t */

WavetableResult spectral_wavetable_load(SpectralWavetableBank* bank,
                                        const char* filename,
                                        uint8_t timbre_id) {
    if (!bank || !filename) return WAVETABLE_ERR_PARAM;
    if (timbre_id >= SPECTRAL_MAX_WAVETABLES) return WAVETABLE_ERR_PARAM;
    
    FILE* f = fopen(filename, "rb");
    if (!f) return WAVETABLE_ERR_FILE;
    
    /* Read header */
    SpectralWavetableHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f);
        return WAVETABLE_ERR_FILE;
    }
    
    /* Validate header */
    if (memcmp(hdr.magic, SPECTRAL_WAVETABLE_MAGIC, 4) != 0) {
        fclose(f);
        return WAVETABLE_ERR_FORMAT;
    }
    if (hdr.version != SPECTRAL_WAVETABLE_VERSION) {
        fclose(f);
        return WAVETABLE_ERR_VERSION;
    }
    if (hdr.size != SPECTRAL_WAVETABLE_SIZE) {
        fclose(f);
        return WAVETABLE_ERR_SIZE;
    }
    if (hdr.format != WAVETABLE_FORMAT_FLOAT && hdr.format != WAVETABLE_FORMAT_Q15) {
        fclose(f);
        return WAVETABLE_ERR_FORMAT;
    }
    
    SpectralWavetable* table = &bank->tables[timbre_id];
    
    /* Determine if conversion is needed */
#if SPECTRAL_EMBEDDED
    const SpectralWavetableFormat runtime_format = WAVETABLE_FORMAT_Q15;
#else
    const SpectralWavetableFormat runtime_format = WAVETABLE_FORMAT_FLOAT;
#endif
    
    if (hdr.format == runtime_format) {
        /* No conversion needed - direct read */
        spectral_sample_t* temp = (spectral_sample_t*)malloc(hdr.size * sizeof(spectral_sample_t));
        if (!temp) {
            fclose(f);
            return WAVETABLE_ERR_MEMORY;
        }
        size_t read = fread(temp, sizeof(spectral_sample_t), hdr.size, f);
        if (read != hdr.size) {
            free(temp);
            fclose(f);
            return WAVETABLE_ERR_SIZE;
        }
        memcpy(table->samples, temp, hdr.size * sizeof(spectral_sample_t));
        free(temp);
    } else if (hdr.format == WAVETABLE_FORMAT_FLOAT && runtime_format == WAVETABLE_FORMAT_Q15) {
        /* File is float, runtime is Q15 - convert float->Q15 */
        float* temp = (float*)malloc(hdr.size * sizeof(float));
        if (!temp) {
            fclose(f);
            return WAVETABLE_ERR_MEMORY;
        }
        if (fread(temp, sizeof(float), hdr.size, f) != hdr.size) {
            free(temp);
            fclose(f);
            return WAVETABLE_ERR_SIZE;
        }
        for (size_t i = 0; i < hdr.size; i++) {
            table->samples[i] = FLOAT_TO_SPECTRAL_SAMPLE(temp[i]);
        }
        free(temp);
    } else {
        /* File is Q15, runtime is float - convert Q15->float */
        int16_t* temp = (int16_t*)malloc(hdr.size * sizeof(int16_t));
        if (!temp) {
            fclose(f);
            return WAVETABLE_ERR_MEMORY;
        }
        if (fread(temp, sizeof(int16_t), hdr.size, f) != hdr.size) {
            free(temp);
            fclose(f);
            return WAVETABLE_ERR_SIZE;
        }
        for (size_t i = 0; i < hdr.size; i++) {
            float sample_f = (float)temp[i] * SPECTRAL_INV_Q15_SCALE;
            table->samples[i] = FLOAT_TO_SPECTRAL_SAMPLE(sample_f);
        }
        free(temp);
    }
    
    fclose(f);
    
    table->samples[SPECTRAL_WAVETABLE_SIZE] = table->samples[0];
    wavetable_mark_loaded(bank, table, timbre_id);
    
    return WAVETABLE_OK;
}

WavetableResult spectral_wavetable_save(const SpectralWavetableBank* bank,
                                        const char* filename,
                                        uint8_t timbre_id) {
    if (!bank || !filename) return WAVETABLE_ERR_PARAM;
    if (timbre_id >= SPECTRAL_MAX_WAVETABLES) return WAVETABLE_ERR_PARAM;
    
    const SpectralWavetable* table = &bank->tables[timbre_id];
    if (!table->valid) return WAVETABLE_ERR_NOT_FOUND;
    
    FILE* f = fopen(filename, "wb");
    if (!f) return WAVETABLE_ERR_FILE;
    
    /* Prepare header */
    SpectralWavetableHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, SPECTRAL_WAVETABLE_MAGIC, 4);
    hdr.version = SPECTRAL_WAVETABLE_VERSION;
#if SPECTRAL_EMBEDDED
    hdr.format = WAVETABLE_FORMAT_Q15;
#else
    hdr.format = WAVETABLE_FORMAT_FLOAT;
#endif
    hdr.size = SPECTRAL_WAVETABLE_SIZE;
    hdr.timbre_id = timbre_id;
    
    /* Write header */
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f);
        return WAVETABLE_ERR_FILE;
    }
    
    /* Write samples (only SPECTRAL_WAVETABLE_SIZE, not the wrap sample) */
    if (fwrite(table->samples, sizeof(spectral_sample_t), SPECTRAL_WAVETABLE_SIZE, f) != SPECTRAL_WAVETABLE_SIZE) {
        fclose(f);
        return WAVETABLE_ERR_FILE;
    }
    
    fclose(f);
    return WAVETABLE_OK;
}

#else /* SPECTRAL_EMBEDDED - no file I/O */

WavetableResult spectral_wavetable_load(SpectralWavetableBank* bank,
                                        const char* filename,
                                        uint8_t timbre_id) {
    (void)bank; (void)filename; (void)timbre_id;
    return WAVETABLE_ERR_FILE;
}

WavetableResult spectral_wavetable_save(const SpectralWavetableBank* bank,
                                        const char* filename,
                                        uint8_t timbre_id) {
    (void)bank; (void)filename; (void)timbre_id;
    return WAVETABLE_ERR_FILE;
}

WavetableResult spectral_wavetable_load_raw(SpectralWavetableBank* bank,
                                            const char* filename,
                                            uint8_t timbre_id) {
    (void)bank; (void)filename; (void)timbre_id;
    return WAVETABLE_ERR_FILE;
}

WavetableResult spectral_wavetable_load_hex(SpectralWavetableBank* bank,
                                            const char* filename,
                                            uint8_t timbre_id) {
    (void)bank; (void)filename; (void)timbre_id;
    return WAVETABLE_ERR_FILE;
}

#endif /* SPECTRAL_EMBEDDED */

WavetableResult spectral_wavetable_load_buffer(SpectralWavetableBank* bank,
                                               const spectral_sample_t* data,
                                               size_t size,
                                               uint8_t timbre_id) {
    if (!bank || !data) return WAVETABLE_ERR_PARAM;
    if (timbre_id >= SPECTRAL_MAX_WAVETABLES) return WAVETABLE_ERR_PARAM;
    if (size < SPECTRAL_WAVETABLE_SIZE) return WAVETABLE_ERR_SIZE;
    
    SpectralWavetable* table = &bank->tables[timbre_id];
    memcpy(table->samples, data, SPECTRAL_WAVETABLE_SIZE * sizeof(spectral_sample_t));
    
    table->samples[SPECTRAL_WAVETABLE_SIZE] = table->samples[0];
    wavetable_mark_loaded(bank, table, timbre_id);
    
    return WAVETABLE_OK;
}

const SpectralWavetable* spectral_wavetable_get(const SpectralWavetableBank* bank,
                                                uint8_t timbre_id) {
    if (!bank || timbre_id >= SPECTRAL_MAX_WAVETABLES) return NULL;
    if (!bank->tables[timbre_id].valid) return NULL;
    return &bank->tables[timbre_id];
}

int spectral_wavetable_has_timbre(const SpectralWavetableBank* bank,
                                  uint8_t timbre_id) {
    if (!bank || timbre_id >= SPECTRAL_MAX_WAVETABLES) return 0;
    return bank->tables[timbre_id].valid;
}

/* Wavetable lookup - hot path, force inlined */

SPECTRAL_FORCEINLINE
spectral_sample_t spectral_wavetable_lookup_f(const SpectralWavetable* table,
                                              float phase_norm) {
    phase_norm = phase_norm - (float)(int)phase_norm;
    if (phase_norm < 0.0f) phase_norm += 1.0f;
    
    float idx_f = phase_norm * (float)SPECTRAL_WAVETABLE_SIZE;
    uint32_t idx = (uint32_t)idx_f;
    float frac = idx_f - (float)idx;
    
    if (idx >= SPECTRAL_WAVETABLE_SIZE) idx = 0;
    
    spectral_sample_t s0 = table->samples[idx];
    spectral_sample_t s1 = table->samples[idx + 1];
    
#if SPECTRAL_EMBEDDED
    int32_t frac_q8 = (int32_t)(frac * 256.0f);
    return (spectral_sample_t)(s0 + ((((int32_t)s1 - (int32_t)s0) * frac_q8) >> 8));
#else
    return s0 + (s1 - s0) * frac;
#endif
}

SPECTRAL_FORCEINLINE
spectral_sample_t spectral_wavetable_lookup_q(const SpectralWavetable* table,
                                              uint16_t phase_u16) {
    uint32_t idx = phase_u16 >> (16 - SPECTRAL_WAVETABLE_BITS);
    uint32_t frac_bits = 16 - SPECTRAL_WAVETABLE_BITS;
    uint32_t frac = (phase_u16 & ((1u << frac_bits) - 1)) >> (frac_bits > 8 ? frac_bits - 8 : 0);
    if (frac_bits < 8) frac <<= (8 - frac_bits);
    
    spectral_sample_t s0 = table->samples[idx];
    spectral_sample_t s1 = table->samples[idx + 1];
    
#if SPECTRAL_EMBEDDED
    return (spectral_sample_t)(s0 + ((((int32_t)s1 - (int32_t)s0) * (int32_t)frac) >> 8));
#else
    float frac_f = (float)frac / 256.0f;
    return s0 + (s1 - s0) * frac_f;
#endif
}

SPECTRAL_FORCEINLINE
spectral_sample_t spectral_wavetable_lookup_timbre_f(const SpectralWavetableBank* bank,
                                                     uint8_t timbre_id,
                                                     float phase_norm) {
    const SpectralWavetable* table = (timbre_id < SPECTRAL_MAX_WAVETABLES && 
                                      bank->tables[timbre_id].valid)
        ? &bank->tables[timbre_id]
        : &bank->tables[bank->default_timbre];
    return spectral_wavetable_lookup_f(table, phase_norm);
}

SPECTRAL_FORCEINLINE
spectral_sample_t spectral_wavetable_lookup_timbre_q(const SpectralWavetableBank* bank,
                                                     uint8_t timbre_id,
                                                     uint16_t phase_u16) {
    const SpectralWavetable* table = (timbre_id < SPECTRAL_MAX_WAVETABLES && 
                                      bank->tables[timbre_id].valid)
        ? &bank->tables[timbre_id]
        : &bank->tables[bank->default_timbre];
    return spectral_wavetable_lookup_q(table, phase_u16);
}
