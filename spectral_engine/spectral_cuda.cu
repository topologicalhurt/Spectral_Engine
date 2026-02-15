/* spectral_cuda.cu - Standalone CUDA synthesis CLI tool
 *
 * Thin wrapper around the spectral_synth_cuda library backend.
 * Provides device info, demo mode, and file-based synthesis.
 */

#include <cuda_runtime.h>
#include <sndfile.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "spectral_common.h"
#include "spectral_segment_parser.h"
#include "spectral_synth.h"

void print_device_info(void) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("CUDA Device: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Memory: %.1f GB, SMs: %d\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), prop.multiProcessorCount);
}

void run_demo(void) {
    printf("\n--- CUDA Synthesis Demo ---\n");

    size_t num_segments = 10000000;
    size_t out_len = 44100 * 10;

    printf("Generating %zu random segments...\n", num_segments);

    Segment* segments = (Segment*)malloc(num_segments * sizeof(Segment));
    float* output = (float*)calloc(out_len, sizeof(float));
    if (!segments || !output) {
        fprintf(stderr, "Error: Failed to allocate memory for demo\n");
        free(segments);
        free(output);
        return;
    }

    srand(42);
    for (size_t i = 0; i < num_segments; i++) {
        segments[i].start = (float)(rand() % (int)(out_len * 0.9f));
        segments[i].length = 32.0f + (float)(rand() % 128);
        segments[i].phase = (float)(rand() % 1000) / 1000.0f * TWO_PI;
        segments[i].omega = 100.0f + (float)(rand() % 2000);
        segments[i].df = 0.0f;
        segments[i].amp = 0.001f + (float)(rand() % 100) / 10000.0f;
        segments[i].da = 0.0f;
        segments[i].width = 1.0f;
    }

    printf("Test: %zu segments, %zu output samples (density=%.1f)\n",
           num_segments, out_len, (float)num_segments / out_len);

    SegmentArray sa = { segments, (uint32_t)num_segments, (uint32_t)num_segments };
    double t_synth;

    cuda_init();
    if (!cuda_available()) {
        fprintf(stderr, "Error: No CUDA device available\n");
        free(segments);
        free(output);
        return;
    }

    SpectralError err = synth_cuda(sa, output, out_len, 1.0f, 0.0f, TIMBRE_SINE, &t_synth);
    if (err != SPECTRAL_OK) {
        fprintf(stderr, "Error: synth_cuda returned %d\n", err);
    } else {
        printf("\nResults:\n");
        printf("  Synthesis time: %.1f ms\n", t_synth * 1000.0);
        printf("  Throughput: %.1f M segments/sec\n", num_segments / t_synth / 1e6);
        printf("  Effective rate: %.1f GB/s\n",
               (num_segments * sizeof(Segment) + out_len * sizeof(float)) / t_synth / 1e9);
    }

    cuda_cleanup();
    free(segments);
    free(output);
}

int main(int argc, char** argv) {
    print_device_info();

    if (argc < 2 || strcmp(argv[1], "--demo") == 0) {
        run_demo();
        return 0;
    }

    const char* seg_path = argv[1];
    const char* out_path = (argc > 2) ? argv[2] : "out_cuda.wav";

    SegmentArray sa;
    int sr;
    float stretch, pitch;

    printf("Loading segments from %s...\n", seg_path);
    int load_err = segments_load(seg_path, &sa, &sr, &stretch, &pitch);
    if (load_err) {
        fprintf(stderr, "Error loading segments: %d\n", load_err);
        return 1;
    }

    printf("Loaded %u segments (sr=%d, stretch=%.2f, pitch=%.1f)\n",
           sa.count, sr, stretch, pitch);

    size_t out_len = (size_t)(sa.count > 0 ?
                              (sa.segs[sa.count-1].start + sa.segs[sa.count-1].length) * stretch * 1.1
                              : sr * 10);
    float* output = (float*)calloc(out_len, sizeof(float));
    if (!output) {
        fprintf(stderr, "Error: Failed to allocate output buffer\n");
        free(sa.segs);
        return 1;
    }

    cuda_init();
    if (!cuda_available()) {
        fprintf(stderr, "Error: No CUDA device available\n");
        free(sa.segs);
        free(output);
        return 1;
    }

    double t_synth;
    SpectralError err = synth_cuda(sa, output, out_len, stretch, pitch, TIMBRE_SINE, &t_synth);
    if (err != SPECTRAL_OK) {
        fprintf(stderr, "Error: synth_cuda returned %d\n", err);
        cuda_cleanup();
        free(sa.segs);
        free(output);
        return 1;
    }

    printf("Synthesis time: %.1f ms\n", t_synth * 1000.0);

    /* Normalize */
    float max_amp = 0.0f;
    for (size_t i = 0; i < out_len; i++) {
        float a = fabsf(output[i]);
        if (a > max_amp) max_amp = a;
    }
    if (max_amp > 0) {
        float scale = 0.95f / max_amp;
        for (size_t i = 0; i < out_len; i++) output[i] *= scale;
    }

    /* Write output */
    SF_INFO info = {0};
    info.samplerate = sr;
    info.channels = 1;
    info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    SNDFILE* outfile = sf_open(out_path, SFM_WRITE, &info);
    if (!outfile) {
        fprintf(stderr, "Error: Cannot create output file %s\n", out_path);
        cuda_cleanup();
        free(sa.segs);
        free(output);
        return 1;
    }
    sf_writef_float(outfile, output, out_len);
    sf_close(outfile);

    printf("Wrote %s (%.1f sec)\n", out_path, (float)out_len / sr);

    cuda_cleanup();
    free(sa.segs);
    free(output);
    return 0;
}
