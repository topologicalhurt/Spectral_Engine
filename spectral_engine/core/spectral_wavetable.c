/* spectral_wavetable.c - wavetable bank: built-in waveform generation + .spwt load/store
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
#include "spectral_utils.h"
#include "spectral_contracts.h"
#include "spectral_q.h"               /* Q15_TO_FLOAT boundary macro */
#include "spectral_osc_formulas.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM
#include "spectral_fs.h"   /* host file I/O; absent on real firmware (arch-05) */
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

/* .spwt format - reads header and converts to runtime spectral_sample_t */

static size_t wavetable_file_sample_size(SpectralWavetableFormat format)
{
    switch (format) {
        case WAVETABLE_FORMAT_FLOAT: return sizeof(float);
        case WAVETABLE_FORMAT_Q15: return sizeof(int16_t);
        default: return 0u;
    }
}

static int wavetable_file_payload_bytes(SpectralWavetableFormat format,
                                        uint32_t size,
                                        size_t* out_bytes)
{
    size_t sample_size = 0;

    if (!out_bytes) return 0;
    *out_bytes = 0;

    sample_size = wavetable_file_sample_size(format);
    if (sample_size == 0u) return 0;

    return spectral_array_bytes((size_t)size, sample_size, out_bytes);
}

static int wavetable_file_expected_bytes(SpectralWavetableFormat format,
                                         uint32_t size,
                                         size_t* out_bytes)
{
    size_t payload_bytes = 0;

    if (!out_bytes) return 0;
    *out_bytes = 0;

    if (!wavetable_file_payload_bytes(format, size, &payload_bytes)) {
        return 0;
    }

    return spectral_size_add(sizeof(SpectralWavetableHeader), payload_bytes, out_bytes);
}

static int wavetable_runtime_samples_valid(const spectral_sample_t* samples, uint32_t count)
{
    if (count > 0u && !samples) return 0;
    return SPECTRAL_SAMPLE_SPAN_FINITE(samples, count);
}


/* Host file loaders (load/save/load_raw/load_hex): pull stdio (FILE*, sscanf), the
 * spectral_fs_* shim (NOT in the firmware link set), and the heap. Compiled on every HOST build
 * (desktop + the embedded host-sims, which run on an OS) but EXCLUDED from real firmware
 * (SPECTRAL_EMBEDDED && !SPECTRAL_IS_EMBEDDED_SIM), which uses spectral_wavetable_load_buffer
 * instead (arch-05; enforced by test_firmware_purity). */
#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM
WavetableError spectral_wavetable_load(SpectralWavetableBank* bank,
                                        const char* filename,
                                        uint8_t timbre_id) {
    FILE* f = NULL;
    SpectralWavetableHeader hdr;
    SpectralWavetable* table = NULL;
    size_t payload_bytes = 0;
    size_t expected_file_bytes = 0;
    uint64_t file_size = 0;
    SpectralError fs_err = SPECTRAL_OK;

    const SpectralWavetableFormat runtime_format =
        SPECTRAL_SAMPLE_IS_FIXED ? WAVETABLE_FORMAT_Q15 : WAVETABLE_FORMAT_FLOAT;

    if (!bank || spectral_is_empty_string(filename)) return WAVETABLE_ERR_PARAM;
    if (timbre_id >= SPECTRAL_MAX_WAVETABLES) return WAVETABLE_ERR_PARAM;

    fs_err = spectral_fs_open(&f, filename, "rb");
    if (fs_err != SPECTRAL_OK || !f) return WAVETABLE_ERR_FILE;

    /* Read header exactly. */
    fs_err = spectral_fs_read_exact(f, &hdr, sizeof(hdr), SPECTRAL_ERR_FILE_READ);
    if (fs_err != SPECTRAL_OK) {
        spectral_fs_close(&f, SPECTRAL_OK);
        return WAVETABLE_ERR_FILE;
    }

    /* Validate header. */
    if (memcmp(hdr.magic, SPECTRAL_WAVETABLE_MAGIC, 4) != 0) {
        spectral_fs_close(&f, SPECTRAL_OK);
        return WAVETABLE_ERR_FORMAT;
    }
    if (hdr.version != SPECTRAL_WAVETABLE_VERSION) {
        spectral_fs_close(&f, SPECTRAL_OK);
        return WAVETABLE_ERR_VERSION;
    }
    if (hdr.size != SPECTRAL_WAVETABLE_SIZE) {
        spectral_fs_close(&f, SPECTRAL_OK);
        return WAVETABLE_ERR_SIZE;
    }
    if (hdr.format != WAVETABLE_FORMAT_FLOAT && hdr.format != WAVETABLE_FORMAT_Q15) {
        spectral_fs_close(&f, SPECTRAL_OK);
        return WAVETABLE_ERR_FORMAT;
    }
    if (hdr.timbre_id >= SPECTRAL_MAX_WAVETABLES) {
        spectral_fs_close(&f, SPECTRAL_OK);
        return WAVETABLE_ERR_FORMAT;
    }

    if (!wavetable_file_payload_bytes((SpectralWavetableFormat)hdr.format, hdr.size, &payload_bytes) ||
        !wavetable_file_expected_bytes((SpectralWavetableFormat)hdr.format, hdr.size, &expected_file_bytes)) {
        spectral_fs_close(&f, SPECTRAL_OK);
        return WAVETABLE_ERR_SIZE;
    }

    fs_err = spectral_fs_file_size(f, &file_size);
    if (fs_err != SPECTRAL_OK) {
        spectral_fs_close(&f, SPECTRAL_OK);
        return WAVETABLE_ERR_FILE;
    }
    if (file_size != (uint64_t)expected_file_bytes) {
        spectral_fs_close(&f, SPECTRAL_OK);
        return WAVETABLE_ERR_SIZE;
    }

    table = &bank->tables[timbre_id];

    if (hdr.format == runtime_format) {
        /* No conversion needed - direct read into checked temporary. */
        spectral_sample_t* temp = (spectral_sample_t*)spectral_malloc_array(
            (size_t)hdr.size, sizeof(spectral_sample_t));
        if (!temp) {
            spectral_fs_close(&f, SPECTRAL_OK);
            return WAVETABLE_ERR_MEMORY;
        }
        fs_err = spectral_fs_read_exact(f, temp, payload_bytes, SPECTRAL_ERR_FILE_READ);
        if (fs_err != SPECTRAL_OK) {
            free(temp);
            spectral_fs_close(&f, SPECTRAL_OK);
            return WAVETABLE_ERR_SIZE;
        }
        if (!wavetable_runtime_samples_valid(temp, hdr.size)) {
            free(temp);
            spectral_fs_close(&f, SPECTRAL_OK);
            return WAVETABLE_ERR_FORMAT;
        }
        memcpy(table->samples, temp, payload_bytes);
        free(temp);
    } else if (hdr.format == WAVETABLE_FORMAT_FLOAT && runtime_format == WAVETABLE_FORMAT_Q15) {
        /* File is float, runtime is Q15 - convert float->Q15. */
        float* temp = (float*)spectral_malloc_array((size_t)hdr.size, sizeof(float));
        if (!temp) {
            spectral_fs_close(&f, SPECTRAL_OK);
            return WAVETABLE_ERR_MEMORY;
        }
        fs_err = spectral_fs_read_exact(f, temp, payload_bytes, SPECTRAL_ERR_FILE_READ);
        if (fs_err != SPECTRAL_OK) {
            free(temp);
            spectral_fs_close(&f, SPECTRAL_OK);
            return WAVETABLE_ERR_SIZE;
        }
        if (!spectral_f32_span_finite(temp, hdr.size)) {
            free(temp);
            spectral_fs_close(&f, SPECTRAL_OK);
            return WAVETABLE_ERR_FORMAT;
        }
        for (size_t i = 0; i < hdr.size; i++) {
            table->samples[i] = float_to_spectral_sample(temp[i]);
        }
        free(temp);
    } else {
        /* File is Q15, runtime is float - convert Q15->float. */
        int16_t* temp = (int16_t*)spectral_malloc_array((size_t)hdr.size, sizeof(int16_t));
        if (!temp) {
            spectral_fs_close(&f, SPECTRAL_OK);
            return WAVETABLE_ERR_MEMORY;
        }
        fs_err = spectral_fs_read_exact(f, temp, payload_bytes, SPECTRAL_ERR_FILE_READ);
        if (fs_err != SPECTRAL_OK) {
            free(temp);
            spectral_fs_close(&f, SPECTRAL_OK);
            return WAVETABLE_ERR_SIZE;
        }
        for (size_t i = 0; i < hdr.size; i++) {
            /* temp[i] is a raw stored Q15 int16; de-quantize via the boundary macro (the
             * mirror of the float->Q15 branch's FLOAT_TO_Q15). This branch is float-runtime,
             * so float_to_spectral_sample is the identity — spectral_sample_to_float(int16)
             * would only widen the int16 to float with no /32768 scale. */
            float sample_f = Q15_TO_FLOAT(temp[i]);
            table->samples[i] = float_to_spectral_sample(sample_f);
        }
        free(temp);
    }

    spectral_fs_close(&f, SPECTRAL_OK);

    table->samples[SPECTRAL_WAVETABLE_SIZE] = table->samples[0];
    wavetable_mark_loaded(bank, table, timbre_id);
    return WAVETABLE_OK;
}







WavetableError spectral_wavetable_load_raw(SpectralWavetableBank* bank,
                                            const char* filename,
                                            uint8_t timbre_id) {
    size_t sample_bytes = 0;
    uint64_t file_size = 0;
    FILE* f = NULL;
    SpectralWavetable* table = NULL;
    spectral_sample_t* temp = NULL;
    SpectralError fs_err = SPECTRAL_OK;

    if (!bank || spectral_is_empty_string(filename)) return WAVETABLE_ERR_PARAM;
    if (timbre_id >= SPECTRAL_MAX_WAVETABLES) return WAVETABLE_ERR_PARAM;

    if (!spectral_array_bytes(SPECTRAL_WAVETABLE_SIZE, sizeof(spectral_sample_t), &sample_bytes)) {
        return WAVETABLE_ERR_SIZE;
    }

    fs_err = spectral_fs_open(&f, filename, "rb");
    if (fs_err != SPECTRAL_OK || !f) return WAVETABLE_ERR_FILE;

    fs_err = spectral_fs_file_size(f, &file_size);
    if (fs_err != SPECTRAL_OK) {
        spectral_fs_close(&f, SPECTRAL_OK);
        return WAVETABLE_ERR_FILE;
    }
    if (file_size != (uint64_t)sample_bytes) {
        spectral_fs_close(&f, SPECTRAL_OK);
        return WAVETABLE_ERR_SIZE;
    }

    temp = (spectral_sample_t*)spectral_malloc_array(SPECTRAL_WAVETABLE_SIZE, sizeof(spectral_sample_t));
    if (!temp) {
        spectral_fs_close(&f, SPECTRAL_OK);
        return WAVETABLE_ERR_MEMORY;
    }

    fs_err = spectral_fs_read_exact(f, temp, sample_bytes, SPECTRAL_ERR_FILE_READ);
    {
        SpectralError close_err = spectral_fs_close(&f, SPECTRAL_ERR_FILE_READ);
        if (fs_err == SPECTRAL_OK && close_err != SPECTRAL_OK) fs_err = close_err;
    }
    if (fs_err != SPECTRAL_OK) {
        free(temp);
        return WAVETABLE_ERR_SIZE;
    }

    if (!wavetable_runtime_samples_valid(temp, SPECTRAL_WAVETABLE_SIZE)) {
        free(temp);
        return WAVETABLE_ERR_FORMAT;
    }

    table = &bank->tables[timbre_id];
    memcpy(table->samples, temp, sample_bytes);
    free(temp);

    table->samples[SPECTRAL_WAVETABLE_SIZE] = table->samples[0];
    wavetable_mark_loaded(bank, table, timbre_id);

    return WAVETABLE_OK;
}


static WavetableError parse_hex_line(const char* line, uint8_t* data, size_t data_capacity,
                                     size_t* data_len, uint16_t* address, uint8_t* record_type) {
    uint8_t byte_count = 0;
    uint8_t checksum_calc = 0;
    uint8_t checksum_read = 0;
    size_t line_len = 0;
    size_t required_len = 0;

    if (!line || !data || !data_len || !address || !record_type) return WAVETABLE_ERR_PARAM;
    if (line[0] != ':') return WAVETABLE_ERR_FORMAT;

    line_len = strcspn(line, "\r\n");
    if (line_len < 11u) return WAVETABLE_ERR_FORMAT;

    if (sscanf(line + 1, "%2hhx", &byte_count) != 1) return WAVETABLE_ERR_FORMAT;
    if (byte_count > data_capacity) return WAVETABLE_ERR_SIZE;

    required_len = 11u + ((size_t)byte_count * 2u);
    if (line_len != required_len) return WAVETABLE_ERR_FORMAT;

    if (sscanf(line + 3, "%4hx", address) != 1) return WAVETABLE_ERR_FORMAT;
    if (sscanf(line + 7, "%2hhx", record_type) != 1) return WAVETABLE_ERR_FORMAT;

    checksum_calc += byte_count;
    checksum_calc += (uint8_t)((*address >> 8) & 0xFFu);
    checksum_calc += (uint8_t)(*address & 0xFFu);
    checksum_calc += *record_type;

    *data_len = (size_t)byte_count;
    for (size_t i = 0; i < (size_t)byte_count; i++) {
        if (sscanf(line + 9 + (i * 2), "%2hhx", &data[i]) != 1) return WAVETABLE_ERR_FORMAT;
        checksum_calc += data[i];
    }

    if (sscanf(line + 9 + ((size_t)byte_count * 2), "%2hhx", &checksum_read) != 1) return WAVETABLE_ERR_FORMAT;
    if ((uint8_t)(checksum_calc + checksum_read) != 0) return WAVETABLE_ERR_FORMAT;

    return WAVETABLE_OK;
}


WavetableError spectral_wavetable_load_hex(SpectralWavetableBank* bank,
                                            const char* filename,
                                            uint8_t timbre_id) {
    size_t expected_bytes = 0;
    size_t temp_table_count = 0;
    size_t covered_bytes = 0;
    FILE* f = NULL;
    SpectralWavetable* table = NULL;
    spectral_sample_t* temp_table = NULL;
    uint8_t* written = NULL;
    char line[256];
    uint8_t data[32];
    size_t data_len = 0;
    uint16_t address = 0;
    uint8_t record_type = 0;
    int saw_eof = 0;

    if (!bank || spectral_is_empty_string(filename)) return WAVETABLE_ERR_PARAM;
    if (timbre_id >= SPECTRAL_MAX_WAVETABLES) return WAVETABLE_ERR_PARAM;
    if (!spectral_array_bytes(SPECTRAL_WAVETABLE_SIZE, sizeof(spectral_sample_t), &expected_bytes)) {
        return WAVETABLE_ERR_SIZE;
    }
    if (!spectral_size_add(SPECTRAL_WAVETABLE_SIZE, 1u, &temp_table_count)) {
        return WAVETABLE_ERR_SIZE;
    }

    if (spectral_fs_open(&f, filename, "r") != SPECTRAL_OK) return WAVETABLE_ERR_FILE;
    if (!f) return WAVETABLE_ERR_FILE;

    temp_table = (spectral_sample_t*)spectral_calloc_array(temp_table_count, sizeof(spectral_sample_t));
    if (!temp_table) {
        spectral_fs_close(&f, SPECTRAL_OK);
        return WAVETABLE_ERR_MEMORY;
    }
    written = (uint8_t*)spectral_calloc_array(expected_bytes, sizeof(uint8_t));
    if (!written) {
        spectral_fs_close(&f, SPECTRAL_OK);
        free(temp_table);
        return WAVETABLE_ERR_MEMORY;
    }

    while (spectral_fs_gets(line, sizeof(line), f)) {
        WavetableError parse_result;
        if (line[0] != ':') continue;

        parse_result = parse_hex_line(line, data, sizeof(data), &data_len, &address, &record_type);
        if (parse_result != WAVETABLE_OK) {
            spectral_fs_close(&f, SPECTRAL_OK);
            free(temp_table);
            free(written);
            return parse_result;
        }

        if (record_type == 0x00) {
            size_t offset = (size_t)address;
            if (offset > expected_bytes || data_len > expected_bytes - offset) {
                spectral_fs_close(&f, SPECTRAL_OK);
                free(temp_table);
                free(written);
                return WAVETABLE_ERR_SIZE;
            }
            for (size_t i = 0; i < data_len; i++) {
                size_t pos = offset + i;
                if (!written[pos]) {
                    written[pos] = 1;
                    covered_bytes++;
                }
            }
            memcpy((uint8_t*)temp_table + offset, data, data_len);
        } else if (record_type == 0x01) {
            if (data_len != 0u || address != 0u) {
                spectral_fs_close(&f, SPECTRAL_OK);
                free(temp_table);
                free(written);
                return WAVETABLE_ERR_FORMAT;
            }
            saw_eof = 1;
            break;
        } else {
            spectral_fs_close(&f, SPECTRAL_OK);
            free(temp_table);
            free(written);
            return WAVETABLE_ERR_FORMAT;
        }
    }

    {
        SpectralError close_err = spectral_fs_close(&f, SPECTRAL_ERR_FILE_READ);
        if (close_err != SPECTRAL_OK) {
            free(temp_table);
            free(written);
            return WAVETABLE_ERR_FILE;
        }
    }

    if (!saw_eof) {
        free(temp_table);
        free(written);
        return WAVETABLE_ERR_FORMAT;
    }
    if (covered_bytes != expected_bytes) {
        free(temp_table);
        free(written);
        return WAVETABLE_ERR_SIZE;
    }
    if (!wavetable_runtime_samples_valid(temp_table, SPECTRAL_WAVETABLE_SIZE)) {
        free(temp_table);
        free(written);
        return WAVETABLE_ERR_FORMAT;
    }

    table = &bank->tables[timbre_id];
    memcpy(table->samples, temp_table, expected_bytes);
    free(temp_table);
    free(written);

    table->samples[SPECTRAL_WAVETABLE_SIZE] = table->samples[0];
    wavetable_mark_loaded(bank, table, timbre_id);

    return WAVETABLE_OK;
}
#endif /* host file loaders: !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM (arch-05) */


WavetableError spectral_wavetable_load_buffer(SpectralWavetableBank* bank,
                                               const spectral_sample_t* data,
                                               size_t size,
                                               uint8_t timbre_id) {
    SpectralWavetable* table = NULL;
    size_t copy_bytes = 0;

    if (!bank || !data) return WAVETABLE_ERR_PARAM;
    if (timbre_id >= SPECTRAL_MAX_WAVETABLES) return WAVETABLE_ERR_PARAM;
    if (size < SPECTRAL_WAVETABLE_SIZE) return WAVETABLE_ERR_SIZE;
    if (!wavetable_runtime_samples_valid(data, SPECTRAL_WAVETABLE_SIZE)) {
        return WAVETABLE_ERR_FORMAT;
    }
    if (!spectral_array_bytes(SPECTRAL_WAVETABLE_SIZE, sizeof(spectral_sample_t), &copy_bytes)) {
        return WAVETABLE_ERR_SIZE;
    }

    table = &bank->tables[timbre_id];
    memcpy(table->samples, data, copy_bytes);

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

spectral_sample_t spectral_wavetable_lookup_f(const SpectralWavetable* table,
                                              float phase_norm) {
    if (!table || !spectral_is_finite_f32(phase_norm)) return SPECTRAL_SAMPLE_ZERO;
    phase_norm = phase_norm - floorf(phase_norm);
    if (phase_norm < 0.0f) phase_norm += 1.0f;
    float idx_f = phase_norm * (float)SPECTRAL_WAVETABLE_SIZE;
    uint32_t idx = (uint32_t)idx_f;
    float frac = idx_f - (float)idx;
    if (idx >= SPECTRAL_WAVETABLE_SIZE) idx = 0;
    spectral_sample_t s0 = table->samples[idx];
    spectral_sample_t s1 = table->samples[idx + 1];
    return spectral_sample_lerp_f(s0, s1, frac);
}
