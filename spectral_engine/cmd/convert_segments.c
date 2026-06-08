/* convert_segments.c - Desktop SPEC to embedded SPQ (Q15) converter
 *
 * Converts floating-point segment files (.bin) to Q15 fixed-point format (.spq)
 * for embedded targets. Uses shared type definitions from spectral_engine headers.
 *
 * Build: make convert_segments
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spectral_config.h"
#include "spectral_q15.h"
#include "spectral_segment_parser.h"
#include "spectral_endian.h"
#include "spectral_common.h"
#include "spectral_log.h"
#include "spectral_utils.h"

typedef struct ConvertOptions {
    const char* input_path;
    const char* output_path;
    uint32_t max_segments; /* 0 = unlimited */
    size_t pool_mb;
    size_t pool_bytes;
    size_t pool_max_segments;
} ConvertOptions;

typedef struct ConvertStats {
    uint32_t output_count;
    uint32_t output_length;
    uint32_t truncated_lengths;
    uint32_t invalid_bounds;
    uint32_t invalid_fields;
    uint32_t end_saturated;
    uint32_t high_freq;
} ConvertStats;

static size_t default_pool_max_segments(void) {
    size_t pool_bytes = 0;
    if (!spectral_size_mul((size_t)SPECTRAL_CONVERT_DEFAULT_POOL_MB, SPECTRAL_BYTES_PER_MIB, &pool_bytes)) {
        return 0;
    }
    return pool_bytes / sizeof(SpectralSegmentQ15);
}

static void print_usage(const char* prog) {
    SPECTRAL_LOG_INFO("Usage: %s input.bin output.spq [max_segments] [pool_mb]", prog);
    SPECTRAL_LOG_INFO("Converts desktop SPEC (float) to embedded SPQ (Q15) format.");
    SPECTRAL_LOG_INFO("Segment size: %zu bytes (%s)",
                      sizeof(SpectralSegmentQ15),
                      SPECTRAL_HAS_CHIRP ? "with chirp" : "compact, no chirp");
    SPECTRAL_LOG_INFO("Default pool: %u MB, max segments: %zu",
                      (unsigned)SPECTRAL_CONVERT_DEFAULT_POOL_MB, default_pool_max_segments());
}

static int parse_options(int argc, char** argv, ConvertOptions* opts) {
    if (!opts || argc < 3) return 0;
    if (argc > 5) {
        SPECTRAL_LOG_ERROR_STDERR("Error: too many arguments");
        return 0;
    }

    opts->input_path = argv[1];
    opts->output_path = argv[2];
    opts->max_segments = 0;
    opts->pool_mb = SPECTRAL_CONVERT_DEFAULT_POOL_MB;

    if (argc > 3 && !parse_u32_arg(argv[3], &opts->max_segments)) {
        SPECTRAL_LOG_ERROR_STDERR("Error: invalid max_segments '%s' (expected non-negative integer)", argv[3]);
        return 0;
    }
    if (argc > 4 && !parse_size_arg(argv[4], &opts->pool_mb)) {
        SPECTRAL_LOG_ERROR_STDERR("Error: invalid pool_mb '%s' (expected non-negative integer)", argv[4]);
        return 0;
    }
    if (!spectral_size_mul(opts->pool_mb, SPECTRAL_BYTES_PER_MIB, &opts->pool_bytes)) {
        SPECTRAL_LOG_ERROR_STDERR("Error: pool_mb '%zu' is too large for this platform", opts->pool_mb);
        return 0;
    }
    opts->pool_max_segments = opts->pool_bytes / sizeof(SpectralSegmentQ15);
    return 1;
}

static int load_and_validate_header(FILE* fin, SegmentFileHeader* header) {
    if (!fin || !header) return 0;
    if (fread(header, sizeof(*header), 1, fin) != 1) {
        SPECTRAL_LOG_ERROR_STDERR("Error: cannot read segment header");
        return 0;
    }
    if (memcmp(header->magic, SEGMENT_FILE_MAGIC, 4) != 0) {
        SPECTRAL_LOG_ERROR_STDERR("Error: invalid format (expected %s)", SEGMENT_FILE_MAGIC);
        return 0;
    }
    /* The .bin (SPEC) format is stored little-endian; convert the header to
     * native order before validating its fields (magic is endian-independent).
     * No-op on little-endian hosts. */
    spectral_segment_file_header_swap_le(header);
    if (header->version != SEGMENT_FILE_VERSION) {
        SPECTRAL_LOG_ERROR_STDERR("Error: unsupported segment version %u (expected %u)",
                                  header->version, (unsigned)SEGMENT_FILE_VERSION);
        return 0;
    }
    if (header->sr == 0) {
        SPECTRAL_LOG_ERROR_STDERR("Error: invalid sample rate 0 in input header");
        return 0;
    }
    if (!spectral_is_finite_f32(header->stretch) || header->stretch < 0.0f) {
        SPECTRAL_LOG_ERROR_STDERR("Error: invalid stretch %.6g in input header", header->stretch);
        return 0;
    }
    if (!spectral_is_finite_f32(header->pitch)) {
        SPECTRAL_LOG_ERROR_STDERR("Error: invalid pitch %.6g in input header", header->pitch);
        return 0;
    }
    return 1;
}

static uint32_t clamp_start_sample(float v, ConvertStats* stats) {
    if (!spectral_is_finite_f32(v) || v < 0.0f) {
        if (stats) stats->invalid_bounds++;
        return 0u;
    }
    if (v >= (float)UINT32_MAX) {
        if (stats) stats->invalid_bounds++;
        return UINT32_MAX;
    }
    return (uint32_t)v;
}

static uint16_t clamp_length_sample(float v, ConvertStats* stats) {
    if (!spectral_is_finite_f32(v) || v <= 0.0f) {
        if (stats && v != 0.0f) stats->invalid_bounds++;
        return 0u;
    }
    if (v > (float)UINT16_MAX) {
        if (stats) stats->truncated_lengths++;
        return UINT16_MAX;
    }
    return (uint16_t)v;
}

static float sanitize_clamped_field(float v, float lo, float hi, float fallback, ConvertStats* stats) {
    int used_fallback = 0;
    float sanitized = spectral_sanitize_finite_clamp_f32(v, lo, hi, fallback, &used_fallback);
    if (used_fallback && stats) {
        stats->invalid_fields++;
    }
    return sanitized;
}

static int compute_output_length(uint32_t base_length, float stretch, uint32_t* out_length) {
    double scaled;
    if (!out_length) return 0;
    if (!spectral_is_finite_f32(stretch) || stretch < 0.0f) return 0;
    scaled = (double)base_length * (double)stretch;
    if (!(scaled >= 0.0) || scaled > (double)UINT32_MAX) return 0;
    *out_length = (uint32_t)scaled;
    return 1;
}

static int write_spq_file(const char* output_path,
                          const SegmentFileHeader* in_header,
                          const SpectralSegmentQ15* q15_segs,
                          uint32_t output_count,
                          uint32_t output_length) {
    FILE* fout = NULL;
    SpqFileHeader out_header;
    int ok = 0;

    if (!output_path || !in_header) return 0;

    fout = fopen(output_path, "wb");
    if (!fout) {
        SPECTRAL_LOG_ERROR_STDERR("Error: cannot create %s", output_path);
        return 0;
    }

    out_header.magic = SPQ_FILE_MAGIC;
    out_header.version = SPQ_FILE_VERSION;
    out_header.sample_rate = in_header->sr;
    out_header.num_segments = output_count;
    out_header.output_length = output_length;
    out_header.flags = SPECTRAL_HAS_CHIRP ? 1u : 0u;
    out_header.reserved[0] = 0;
    out_header.reserved[1] = 0;

    if (fwrite(&out_header, sizeof(out_header), 1, fout) != 1) {
        SPECTRAL_LOG_ERROR_STDERR("Error: failed writing SPQ header to %s", output_path);
        goto cleanup;
    }
    if (output_count > 0 &&
        fwrite(q15_segs, sizeof(SpectralSegmentQ15), output_count, fout) != output_count) {
        SPECTRAL_LOG_ERROR_STDERR("Error: failed writing segment payload to %s", output_path);
        goto cleanup;
    }

    ok = 1;

cleanup:
    if (fout) fclose(fout);
    return ok;
}

static void print_input_summary(const ConvertOptions* opts, const SegmentFileHeader* header) {
    size_t input_bytes = 0;
    if (!opts || !header) return;
    (void)spectral_size_mul((size_t)header->count, sizeof(Segment), &input_bytes);

    SPECTRAL_LOG_INFO("Input: %s", opts->input_path);
    SPECTRAL_LOG_INFO("  SR: %u Hz  Stretch: %.3f  Pitch: %.1f  Segs: %u  Size: %.2f MB",
                      header->sr, header->stretch, header->pitch, header->count,
                      BYTES_TO_MB(input_bytes));
}

static void print_output_summary(const ConvertOptions* opts, const ConvertStats* stats, uint32_t sample_rate) {
    size_t out_size = 0;
    size_t seg_bytes = 0;
    if (!opts || !stats || sample_rate == 0) return;

    (void)spectral_size_mul((size_t)stats->output_count, sizeof(SpectralSegmentQ15), &seg_bytes);
    (void)spectral_size_add(sizeof(SpqFileHeader), seg_bytes, &out_size);

    SPECTRAL_LOG_INFO("\nOutput: %s", opts->output_path);
    SPECTRAL_LOG_INFO("  Segments: %u  Length: %u samples (%.2fs)  Size: %.2f MB",
                      stats->output_count, stats->output_length,
                      (double)stats->output_length / (double)sample_rate,
                      BYTES_TO_MB(out_size));

    if (opts->pool_bytes > 0) {
        SPECTRAL_LOG_INFO("  Pool: %.2f / %zu MB (%.1f%%)",
                          BYTES_TO_MB(seg_bytes), opts->pool_mb,
                          100.0 * (double)seg_bytes / (double)opts->pool_bytes);
    } else {
        SPECTRAL_LOG_INFO("  Pool: %.2f / %zu MB (n/a)", BYTES_TO_MB(seg_bytes), opts->pool_mb);
    }

    if (seg_bytes > opts->pool_bytes) {
        SPECTRAL_LOG_INFO("*** ERROR: Exceeds target memory! ***");
    }
}

int main(int argc, char** argv) {
    ConvertOptions opts;
    SegmentFileHeader header;
    Segment* float_segs = NULL;
    SpectralSegmentQ15* q15_segs = NULL;
    ConvertStats stats;
    FILE* fin = NULL;
    uint32_t requested_count;
    int rc = 1;

    memset(&opts, 0, sizeof(opts));
    memset(&header, 0, sizeof(header));
    memset(&stats, 0, sizeof(stats));

    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    if (!parse_options(argc, argv, &opts)) return 1;

    fin = fopen(opts.input_path, "rb");
    if (!fin) {
        SPECTRAL_LOG_ERROR_STDERR("Error: cannot open %s", opts.input_path);
        goto cleanup;
    }
    if (!load_and_validate_header(fin, &header)) goto cleanup;

    print_input_summary(&opts, &header);

    requested_count = header.count;
    if (opts.max_segments > 0 && requested_count > opts.max_segments) {
        requested_count = opts.max_segments;
    }
    if ((size_t)requested_count > opts.pool_max_segments) {
        SPECTRAL_LOG_WARN("Warning: truncating to %zu (pool limit)", opts.pool_max_segments);
        /* Safe narrowing: this branch only runs when pool_max_segments < requested_count,
         * and requested_count is uint32_t, so pool_max_segments < UINT32_MAX here. */
        requested_count = (uint32_t)opts.pool_max_segments;
    }

    if (requested_count > 0) {
        size_t float_bytes = 0;
        size_t q15_bytes = 0;
        if (!spectral_size_mul((size_t)requested_count, sizeof(Segment), &float_bytes) ||
            !spectral_size_mul((size_t)requested_count, sizeof(SpectralSegmentQ15), &q15_bytes)) {
            SPECTRAL_LOG_ERROR_STDERR("Error: requested segment allocation is too large");
            goto cleanup;
        }

        float_segs = (Segment*)malloc(float_bytes);
        q15_segs = (SpectralSegmentQ15*)malloc(q15_bytes);
        if (!float_segs || !q15_segs) {
            SPECTRAL_LOG_ERROR_STDERR("Error: out of memory");
            goto cleanup;
        }
        {
            size_t read_count = fread(float_segs, sizeof(Segment), requested_count, fin);
            if (read_count != requested_count) {
                SPECTRAL_LOG_WARN("Warning: read %zu of %u segments", read_count, requested_count);
            }
            stats.output_count = (uint32_t)read_count;

            /* Segments are stored little-endian on disk; convert to native order
             * before reading their float fields. No-op on little-endian hosts. */
            if (spectral_is_big_endian()) {
                for (size_t i = 0; i < read_count; i++) {
                    spectral_segment_swap_endian(&float_segs[i]);
                }
            }
        }
    }

    SPECTRAL_LOG_INFO("Converting %u segments to Q15 (%s mode)...",
                      stats.output_count, SPECTRAL_HAS_CHIRP ? "full" : "compact");

    for (uint32_t i = 0; i < stats.output_count; i++) {
        const Segment* src = &float_segs[i];
        SpectralSegmentQ15* dst = &q15_segs[i];
        uint32_t seg_end;
        float omega, phase, amp, da, df;

        dst->start = clamp_start_sample(src->start, &stats);
        dst->length = clamp_length_sample(src->length, &stats);

        if (dst->start > UINT32_MAX - (uint32_t)dst->length) {
            seg_end = UINT32_MAX;
            stats.end_saturated++;
        } else {
            seg_end = dst->start + (uint32_t)dst->length;
        }
        if (seg_end > stats.output_length) stats.output_length = seg_end;

        omega = sanitize_clamped_field(src->omega, -INFINITY, INFINITY, 0.0f, &stats);
        phase = sanitize_clamped_field(src->phase, -INFINITY, INFINITY, 0.0f, &stats);
        amp = sanitize_clamped_field(src->amp, -1.0f, 1.0f, 0.0f, &stats);
        da = sanitize_clamped_field(src->da / SPECTRAL_MILLIS_PER_SECOND_F, -1.0f, 1.0f, 0.0f, &stats);

        if (omega > 255.0f) stats.high_freq++;
        dst->freq_q88 = OMEGA_TO_Q88(omega);
        dst->phase_q15 = PHASE_RAD_TO_Q15(phase);
        dst->amp_q15 = FLOAT_TO_Q15(amp);
        dst->da_q15 = FLOAT_TO_Q15(da);

#if SPECTRAL_HAS_CHIRP
        df = sanitize_clamped_field(src->df / SPECTRAL_MILLIS_PER_SECOND_F, -1.0f, 1.0f, 0.0f, &stats);
        dst->df_q15 = FLOAT_TO_Q15(df);
#endif
    }

    if (stats.truncated_lengths) {
        SPECTRAL_LOG_INFO("  %u segments length-truncated", stats.truncated_lengths);
    }
    if (stats.invalid_bounds) {
        SPECTRAL_LOG_INFO("  %u segments had invalid start/length fields (clamped)", stats.invalid_bounds);
    }
    if (stats.invalid_fields) {
        SPECTRAL_LOG_INFO("  %u segments had invalid phase/frequency/amplitude fields (sanitized)",
                          stats.invalid_fields);
    }
    if (stats.end_saturated) {
        SPECTRAL_LOG_INFO("  %u segments had end overflow (saturated)", stats.end_saturated);
    }
    if (stats.high_freq) {
        SPECTRAL_LOG_INFO("  %u segments freq > 255 Hz (encoded /4)", stats.high_freq);
    }
#if !SPECTRAL_HAS_CHIRP
    SPECTRAL_LOG_INFO("  Note: df (chirp) values ignored in compact mode");
#endif

    {
        uint32_t stretched = 0;
        if (!compute_output_length(stats.output_length, header.stretch, &stretched)) {
            SPECTRAL_LOG_ERROR_STDERR("Error: output length overflow after stretch (%.3f)", header.stretch);
            goto cleanup;
        }
        stats.output_length = stretched;
    }

    if (!write_spq_file(opts.output_path, &header, q15_segs, stats.output_count, stats.output_length)) {
        goto cleanup;
    }

    print_output_summary(&opts, &stats, header.sr);
    SPECTRAL_LOG_INFO("Done.");
    rc = 0;

cleanup:
    if (fin) fclose(fin);
    free(float_segs);
    free(q15_segs);
    return rc;
}
