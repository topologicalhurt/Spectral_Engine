/* convert_segments.c - Desktop SPEC to embedded SPQ (Q15) converter
 * 
 * Converts floating-point segment files (.bin) to Q15 fixed-point format (.spq)
 * for embedded targets. Uses shared type definitions from spectral_engine headers.
 * 
 * Build: clang -O2 -I../spectral_engine convert_segments.c -o convert_segments -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Use shared definitions from spectral_engine */
#include "spectral_config.h"
#include "spectral_q15.h"
#include "spectral_segment_parser.h"

/* Desktop segment (matches Segment in spectral_common.h) */
typedef struct {
    float start, length, phase, freq_hz, df, amp, da, width;
    float _pad[8];
} SegmentFloat;

/* Embedded memory pool defaults */
#ifndef EMBEDDED_SEGMENT_POOL_MB
#define EMBEDDED_SEGMENT_POOL_MB  48
#endif
#define EMBEDDED_SEGMENT_POOL_SIZE  ((size_t)EMBEDDED_SEGMENT_POOL_MB * 1024 * 1024)
#define EMBEDDED_MAX_SEGMENTS       (EMBEDDED_SEGMENT_POOL_SIZE / sizeof(SpectralSegmentQ15))
#define EMBEDDED_MAX_LENGTH         65535

/* Frequency to Q8.8 conversion */
static inline uint16_t freq_to_q88(float freq_hz) {
    /* Q8.8 format: values > 255 Hz encoded as /4 */
    if (freq_hz > 255.0f) freq_hz /= 4.0f;
    if (freq_hz > 255.0f) freq_hz = 255.0f;
    if (freq_hz < 0.0f) freq_hz = 0.0f;
    return (uint16_t)(freq_hz * 256.0f);
}

/* Phase to Q15 conversion: [0, 2pi) -> [-32768, 32767] */
static inline int16_t phase_to_q15(float phase_rad) {
    /* Normalize to [0, 1), subtract 0.5 to center, scale to Q15 */
    float norm = fmodf(phase_rad, SPECTRAL_TWO_PI) / SPECTRAL_TWO_PI;
    if (norm < 0.0f) norm += 1.0f;
    return (int16_t)((norm - 0.5f) * 65536.0f);
}

static void print_usage(const char* prog) {
    printf("Usage: %s input.bin output.spq [max_segments] [pool_mb]\n", prog);
    printf("Converts desktop SPEC (float) to embedded SPQ (Q15) format.\n");
    printf("Segment size: %zu bytes (%s)\n", 
           sizeof(SpectralSegmentQ15),
           SPECTRAL_HAS_CHIRP ? "with chirp" : "compact, no chirp");
    printf("Default pool: %d MB, max segments: %zu\n", 
           EMBEDDED_SEGMENT_POOL_MB, (size_t)EMBEDDED_MAX_SEGMENTS);
}

int main(int argc, char** argv) {
    if (argc < 3) { print_usage(argv[0]); return 1; }
    
    const char* input_path = argv[1];
    const char* output_path = argv[2];
    uint32_t max_segments = (argc > 3) ? (uint32_t)atoi(argv[3]) : 0;
    size_t pool_mb = (argc > 4) ? (size_t)atoi(argv[4]) : EMBEDDED_SEGMENT_POOL_MB;
    size_t pool_size = pool_mb * 1024 * 1024;
    size_t pool_max_segs = pool_size / sizeof(SpectralSegmentQ15);
    
    FILE* fin = fopen(input_path, "rb");
    if (!fin) { fprintf(stderr, "Error: Cannot open %s\n", input_path); return 1; }
    
    SegmentFileHeader header;
    if (fread(&header, sizeof(header), 1, fin) != 1) {
        fprintf(stderr, "Error: Cannot read header\n");
        fclose(fin); return 1;
    }
    
    if (memcmp(header.magic, SEGMENT_FILE_MAGIC, 4) != 0) {
        fprintf(stderr, "Error: Invalid format (expected %s)\n", SEGMENT_FILE_MAGIC);
        fclose(fin); return 1;
    }
    
    printf("Input: %s\n", input_path);
    printf("  SR: %u Hz  Stretch: %.3f  Pitch: %.1f  Segs: %u  Size: %.2f MB\n",
           header.sr, header.stretch, header.pitch, header.count,
           (header.count * sizeof(SegmentFloat)) / (1024.0 * 1024.0));
    
    uint32_t output_count = header.count;
    if (max_segments > 0 && output_count > max_segments) output_count = max_segments;
    if (output_count > pool_max_segs) {
        printf("Warning: Truncating to %zu (pool limit)\n", pool_max_segs);
        output_count = (uint32_t)pool_max_segs;
    }
    
    SegmentFloat* float_segs = malloc(output_count * sizeof(SegmentFloat));
    SpectralSegmentQ15* q15_segs = malloc(output_count * sizeof(SpectralSegmentQ15));
    if (!float_segs || !q15_segs) { 
        fprintf(stderr, "Error: Out of memory\n"); 
        free(float_segs); free(q15_segs);
        fclose(fin); 
        return 1; 
    }
    
    size_t read_count = fread(float_segs, sizeof(SegmentFloat), output_count, fin);
    fclose(fin);
    if (read_count != output_count) {
        printf("Warning: Read %zu of %u segments\n", read_count, output_count);
        output_count = (uint32_t)read_count;
    }
    
    printf("Converting %u segments to Q15 (%s mode)...\n", 
           output_count, SPECTRAL_HAS_CHIRP ? "full" : "compact");
    
    uint32_t output_length = 0, truncated = 0, high_freq = 0;
    
    for (uint32_t i = 0; i < output_count; i++) {
        SegmentFloat* src = &float_segs[i];
        SpectralSegmentQ15* dst = &q15_segs[i];
        
        dst->start = (uint32_t)src->start;
        dst->length = (src->length > EMBEDDED_MAX_LENGTH) 
            ? (truncated++, EMBEDDED_MAX_LENGTH) 
            : (uint16_t)src->length;
        
        uint32_t seg_end = dst->start + dst->length;
        if (seg_end > output_length) output_length = seg_end;
        
        if (src->freq_hz > 255.0f) high_freq++;
        dst->freq_q88 = freq_to_q88(src->freq_hz);
        dst->phase_q15 = phase_to_q15(src->phase);
        dst->amp_q15 = FLOAT_TO_Q15(CLAMP(src->amp, -1.0f, 1.0f));
        dst->da_q15 = FLOAT_TO_Q15(CLAMP(src->da / 1000.0f, -1.0f, 1.0f));
        
#if SPECTRAL_HAS_CHIRP
        /* Convert df (Hz/sample) to Q15 frequency delta */
        dst->df_q15 = FLOAT_TO_Q15(CLAMP(src->df / 1000.0f, -1.0f, 1.0f));
#endif
    }
    
    if (truncated) printf("  %u segments length-truncated\n", truncated);
    if (high_freq) printf("  %u segments freq > 255 Hz (encoded /4)\n", high_freq);
#if !SPECTRAL_HAS_CHIRP
    printf("  Note: df (chirp) values ignored in compact mode\n");
#endif
    
    output_length = (uint32_t)(output_length * header.stretch);
    
    FILE* fout = fopen(output_path, "wb");
    if (!fout) { 
        fprintf(stderr, "Error: Cannot create %s\n", output_path); 
        free(float_segs); free(q15_segs); 
        return 1; 
    }
    
    SpqFileHeader out_header = {
        .magic = SPQ_FILE_MAGIC,
        .version = SPQ_FILE_VERSION,
        .sample_rate = header.sr,
        .num_segments = output_count,
        .output_length = output_length,
        .flags = SPECTRAL_HAS_CHIRP ? 1 : 0,  /* Flag 0x01 = has chirp */
        .reserved = {0, 0}
    };
    
    fwrite(&out_header, sizeof(out_header), 1, fout);
    fwrite(q15_segs, sizeof(SpectralSegmentQ15), output_count, fout);
    fclose(fout);
    
    size_t out_size = sizeof(SpqFileHeader) + output_count * sizeof(SpectralSegmentQ15);
    uint32_t seg_mem = output_count * sizeof(SpectralSegmentQ15);
    
    printf("\nOutput: %s\n", output_path);
    printf("  Segments: %u  Length: %u samples (%.2fs)  Size: %.2f MB\n",
           output_count, output_length, output_length / (float)header.sr, 
           out_size / (1024.0 * 1024.0));
    printf("  Pool: %.2f / %zu MB (%.1f%%)\n", 
           seg_mem / (1024.0 * 1024.0), pool_mb, 100.0 * seg_mem / pool_size);
    
    if (seg_mem > pool_size) printf("*** ERROR: Exceeds target memory! ***\n");
    
    free(float_segs);
    free(q15_segs);
    printf("Done.\n");
    return 0;
}
