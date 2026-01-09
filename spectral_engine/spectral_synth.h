/* spectral_synth.h - Synthesis backend interface */

#ifndef SPECTRAL_SYNTH_H
#define SPECTRAL_SYNTH_H

#include "spectral_common.h"

typedef enum {
    BACKEND_AUTO   = 0,
    BACKEND_CPU    = 1,
    BACKEND_METAL  = 2,
    BACKEND_CUDA   = 3,
    BACKEND_EXPORT = 4
} SynthBackend;

/* CPU */
void synth_cpu(SegmentArray sa, float* out_buffer, size_t out_len, 
               float stretch, float pitch, int timbre_id, int n_threads, 
               double* t_synth);

/* Metal (macOS) */
#ifdef __APPLE__
#define HAS_METAL 1
void metal_init(void);
int  metal_available(void);
void synth_metal(SegmentArray sa, float* out_buffer, size_t out_len,
                 float stretch, float pitch, double* t_synth);
#else
#define HAS_METAL 0
#endif

/* CUDA (Linux/NVIDIA) */
#ifdef USE_CUDA
#define HAS_CUDA 1
void cuda_init(void);
int  cuda_available(void);
void synth_cuda(SegmentArray sa, float* out_buffer, size_t out_len,
                float stretch, float pitch, double* t_synth);
#else
#define HAS_CUDA 0
#endif

/* Segment I/O */
int segments_export(const char* path, const SegmentArray* sa, int sr, float stretch, float pitch);
int segments_import(const char* path, SegmentArray* sa, int* sr, float* stretch, float* pitch);

#endif
