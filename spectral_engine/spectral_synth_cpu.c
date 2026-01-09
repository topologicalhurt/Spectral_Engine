/* spectral_synth_cpu.c - CPU synthesis with OpenMP */

#include "spectral_synth.h"
#include <omp.h>
#include <string.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define USE_VDSP 1
#else
#define USE_VDSP 0
#endif

void synth_cpu(SegmentArray sa, float* out_buffer, size_t out_len, 
               float stretch, float pitch, int timbre_id, int n_threads,
               double* t_synth) {
            
    float inv_stretch = 1.0f / stretch;
    float inv_stretch_sq = 1.0f / (stretch * stretch);
    float pitch_factor = powf(2.0f, pitch / 12.0f);

    double synth_start = omp_get_wtime();
    
    // Aligned per-thread buffers for SIMD
    size_t buf_size = ((out_len * sizeof(float)) + CACHE_ALIGN - 1) & ~(CACHE_ALIGN - 1);
    float** thread_bufs = malloc(n_threads * sizeof(float*));
    for (int t = 0; t < n_threads; t++) {
        thread_bufs[t] = spectral_aligned_alloc(buf_size);
        memset(thread_bufs[t], 0, buf_size);
    }
    
    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        float* __restrict__ dst_base = thread_bufs[tid];
        
        #pragma omp for schedule(static)
        for (size_t i = 0; i < sa.count; i++) {
            // Prefetch next segment
            if (i + 4 < sa.count) PREFETCH_READ(&sa.segs[i + 4]);
            Segment s = sa.segs[i];
            
            size_t start_idx = (size_t)(s.start * stretch);
            size_t length = (size_t)(s.length * stretch);
            
            if (start_idx >= out_len) continue;
            if (start_idx + length > out_len) length = out_len - start_idx;
            
            float alpha = s.freq_hz * pitch_factor * inv_stretch;
            float beta = s.df * pitch_factor * inv_stretch_sq;
            float d_a = s.da * inv_stretch;
            float phase = s.phase;
            float amp = s.amp;
            float* __restrict__ dst = dst_base + start_idx;
            
            if (timbre_id == 0) {
                #pragma omp simd
                for (size_t j = 0; j < length; j++) {
                    float p = phase + j * (alpha + beta * j);
                    dst[j] += (amp + d_a * j) * fast_sin(p);
                }
            } else {
                float width = s.width;
                for (size_t j = 0; j < length; j++) {
                    float p = phase + j * (alpha + beta * j);
                    float norm = p * INV_TWO_PI;
                    float rads = TWO_PI * (norm - (int)norm + (norm < 0.0f) - 0.5f);
                    float a = amp + d_a * j;
                    
                    switch (timbre_id) {
                        case 1: dst[j] += a * (rads * -0.318309886f); break;
                        case 2: dst[j] += a * ((rads > 0) ? 1.0f : -1.0f); break;
                        case 3: dst[j] += a * ((1.0f - fabsf(rads * 0.636619772f)) * 2.0f - 1.0f); break;
                        case 4: dst[j] += a * asinf(rads * 0.318309886f); break;
                        case 5: dst[j] += a * (1.0f - (rads * rads * 0.10132118f)); break;
                        case 6: dst[j] += a * ((int)(rads * width) / width); break;
                        case 7: dst[j] += a * (((rads + PI) * INV_TWO_PI < width) ? 1.0f : -1.0f); break;
                    }
                }
            }
        }
    }
    
#if USE_VDSP
    memcpy(out_buffer, thread_bufs[0], out_len * sizeof(float));
    for (int t = 1; t < n_threads; t++) {
        vDSP_vadd(out_buffer, 1, thread_bufs[t], 1, out_buffer, 1, out_len);
    }
#else
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < out_len; j++) {
        float sum = thread_bufs[0][j];
        for (int t = 1; t < n_threads; t++) {
            sum += thread_bufs[t][j];
        }
        out_buffer[j] = sum;
    }
#endif
    
    for (int t = 0; t < n_threads; t++) free(thread_bufs[t]);
    free(thread_bufs);
    
    *t_synth = omp_get_wtime() - synth_start;
}

int segments_export(const char* path, const SegmentArray* sa, 
                    int sr, float stretch, float pitch) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    
    uint32_t magic = 0x53504543;
    uint32_t version = 1;
    uint64_t count = sa->count;
    
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&count, sizeof(count), 1, f);
    fwrite(&sr, sizeof(sr), 1, f);
    fwrite(&stretch, sizeof(stretch), 1, f);
    fwrite(&pitch, sizeof(pitch), 1, f);
    fwrite(sa->segs, sizeof(Segment), sa->count, f);
    
    fclose(f);
    return 0;
}

int segments_import(const char* path, SegmentArray* sa,
                    int* sr, float* stretch, float* pitch) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    
    uint32_t magic, version;
    uint64_t count;
    
    fread(&magic, sizeof(magic), 1, f);
    if (magic != 0x53504543) { fclose(f); return -2; }
    
    fread(&version, sizeof(version), 1, f);
    fread(&count, sizeof(count), 1, f);
    fread(sr, sizeof(*sr), 1, f);
    fread(stretch, sizeof(*stretch), 1, f);
    fread(pitch, sizeof(*pitch), 1, f);
    
    sa->segs = (Segment*)malloc(count * sizeof(Segment));
    sa->count = sa->capacity = count;
    fread(sa->segs, sizeof(Segment), count, f);
    
    fclose(f);
    return 0;
}
