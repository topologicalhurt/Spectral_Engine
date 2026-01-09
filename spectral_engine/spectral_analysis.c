/* spectral_analysis.c - FFT and peak tracking */

#include "spectral_analysis.h"
#include <omp.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define USE_VDSP 1
#else
#include <fftw3.h>
#define USE_VDSP 0
#endif

SegmentArray analyze_audio(const float* audio, size_t n_samples, int sr, 
                           int n_fft, int hop, float db_thresh,
                           double* t_fft, double* t_track) {
    size_t n_frames = (n_samples - n_fft) / hop + 1;
    size_t n_freqs = n_fft / 2 + 1;
    
    float* out_spec = spectral_aligned_alloc(n_frames * n_freqs * 2 * sizeof(float));
    float* window_func = spectral_aligned_alloc(n_fft * sizeof(float));
#if USE_VDSP
    vDSP_hann_window(window_func, n_fft, vDSP_HANN_NORM);
#else
    for(int i=0; i<n_fft; i++) {
        window_func[i] = 0.5f * (1.0f - cosf(2.0f * PI * i / (n_fft - 1)));
    }
#endif
    
#if USE_VDSP
    vDSP_Length log2n = (vDSP_Length)log2(n_fft);
    int n_threads = omp_get_max_threads();
    FFTSetup* fft_setups = malloc(n_threads * sizeof(FFTSetup));
    float** thread_real = malloc(n_threads * sizeof(float*));
    float** thread_imag = malloc(n_threads * sizeof(float*));
    float** thread_windowed = malloc(n_threads * sizeof(float*));
    
    for (int t = 0; t < n_threads; t++) {
        fft_setups[t] = vDSP_create_fftsetup(log2n, FFT_RADIX2);
        thread_real[t] = spectral_aligned_alloc(n_fft * sizeof(float));
        thread_imag[t] = spectral_aligned_alloc(n_fft * sizeof(float));
        thread_windowed[t] = spectral_aligned_alloc(n_fft * sizeof(float));
    }
    
    double fft_start = omp_get_wtime();
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        FFTSetup setup = fft_setups[tid];
        DSPSplitComplex split = { thread_real[tid], thread_imag[tid] };
        float* windowed = thread_windowed[tid];
        
        #pragma omp for schedule(static)
        for (size_t t = 0; t < n_frames; t++) {
            const float* src = audio + t * hop;
            vDSP_vmul(src, 1, window_func, 1, windowed, 1, n_fft);
            vDSP_ctoz((DSPComplex*)windowed, 2, &split, 1, n_fft/2);
            vDSP_fft_zrip(setup, &split, 1, log2n, FFT_FORWARD);
            
            float* dest = out_spec + t * n_freqs * 2;
            dest[0] = split.realp[0];
            dest[1] = 0.0f;
            dest[n_freqs*2 - 2] = split.imagp[0];
            dest[n_freqs*2 - 1] = 0.0f;
            for (size_t k = 1; k < n_freqs - 1; k++) {
                dest[k*2] = split.realp[k];
                dest[k*2 + 1] = split.imagp[k];
            }
        }
    }
    
    *t_fft = omp_get_wtime() - fft_start;
    
    for (int t = 0; t < n_threads; t++) {
        vDSP_destroy_fftsetup(fft_setups[t]);
        free(thread_real[t]);
        free(thread_imag[t]);
        free(thread_windowed[t]);
    }
    free(fft_setups);
    free(thread_real);
    free(thread_imag);
    free(thread_windowed);
#else
    fftwf_complex* fftw_out = (fftwf_complex*)out_spec;
    float* in_window = fftwf_alloc_real(n_fft);
    fftwf_plan p = fftwf_plan_dft_r2c_1d(n_fft, in_window, fftw_out, FFTW_MEASURE);

    double fft_start = omp_get_wtime();
    
    for (size_t t = 0; t < n_frames; t++) {
        const float* src = audio + t * hop;
        for (int i = 0; i < n_fft; i++) {
            in_window[i] = src[i] * window_func[i];
        }
        fftwf_execute_dft_r2c(p, in_window, fftw_out + t * n_freqs);
    }
    
    *t_fft = omp_get_wtime() - fft_start;
    
    fftwf_destroy_plan(p);
    fftwf_free(in_window);
#endif
    free(window_func);

    double track_start = omp_get_wtime();
    
    size_t total_bins = n_frames * n_freqs;
    int n_threads_track = omp_get_max_threads();
    float* magsq = spectral_aligned_alloc(total_bins * sizeof(float));
    float max_magsq = 0.0f;
    
    #pragma omp parallel reduction(max:max_magsq)
    {
        float local_max = 0.0f;
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < total_bins; i++) {
            if ((i & 15) == 0) PREFETCH_READ(&out_spec[(i + 64) * 2]);
            float re = out_spec[i * 2];
            float im = out_spec[i * 2 + 1];
            float msq = re*re + im*im;
            magsq[i] = msq;
            if (msq > local_max) local_max = msq;
        }
        if (local_max > max_magsq) max_magsq = local_max;
    }
    
    // Squared threshold for comparison
    float thresh_linear = powf(10.0f, db_thresh / 20.0f);
    float threshsq = thresh_linear * thresh_linear * max_magsq;
    
    // Single-pass extraction with per-thread buffers (eliminates counting pass)
    size_t frames_per_thread = (n_frames + n_threads_track - 1) / n_threads_track;
    size_t max_segs_per_thread = frames_per_thread * (n_freqs / 4);  // Conservative estimate
    
    Segment** thread_segs = malloc(n_threads_track * sizeof(Segment*));
    size_t* thread_counts = calloc(n_threads_track, sizeof(size_t));
    
    for (int t = 0; t < n_threads_track; t++) {
        thread_segs[t] = malloc(max_segs_per_thread * sizeof(Segment));
    }
    
    float freq_step = (float)sr / n_fft;
    float two_pi_ts = TWO_PI / sr;
    float inv_hop = 1.0f / hop;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Segment* local_segs = thread_segs[tid];
        size_t local_count = 0;
        
        #pragma omp for schedule(static) nowait
        for (size_t t = 0; t < n_frames - 1; t++) {
            const float* __restrict__ spec_row = out_spec + t * n_freqs * 2;
            const float* __restrict__ row = magsq + t * n_freqs;
            const float* __restrict__ next_row = magsq + (t+1) * n_freqs;
            const float t_hop = (float)(t * hop);
            
            // Prefetch next frame's data
            if (t + 2 < n_frames) {
                PREFETCH_READ(magsq + (t+2) * n_freqs);
                PREFETCH_READ(out_spec + (t+1) * n_freqs * 2);
            }

            float prev = row[0];
            float curr = row[1];
            
            for (size_t f = 1; f < n_freqs - 1; f++) {
                float next = row[f+1];
                
                if (curr > threshsq && curr > prev && curr > next) {
                    float m0 = next_row[f-1], m1 = next_row[f], m2 = next_row[f+1];
                    float max_vsq = (m0 > m1) ? ((m0 > m2) ? m0 : m2) : ((m1 > m2) ? m1 : m2);
                    
                    if (max_vsq >= threshsq) {
                        int best_next = (m0 > m1) ? ((m0 > m2) ? f-1 : f+1) : ((m1 > m2) ? f : f+1);
                        float m = sqrtf(curr);
                        float max_v = sqrtf(max_vsq);
                        float f_s = f * freq_step;
                        
                        local_segs[local_count++] = (Segment){
                            .start = t_hop,
                            .length = (float)hop,
                            .phase = fast_atan2(spec_row[f*2 + 1], spec_row[f*2]),
                            .freq_hz = f_s * two_pi_ts,
                            .df = 0.5f * (best_next - (int)f) * freq_step * inv_hop * two_pi_ts,
                            .amp = m,
                            .da = (max_v - m) * inv_hop,
                            .width = 0.5f
                        };
                    }
                }
                prev = curr;
                curr = next;
            }
        }
        thread_counts[tid] = local_count;
    }
    
    size_t total_segs = 0;
    size_t* offsets = malloc(n_threads_track * sizeof(size_t));
    for (int t = 0; t < n_threads_track; t++) {
        offsets[t] = total_segs;
        total_segs += thread_counts[t];
    }
    
    Segment* segs = malloc(total_segs * sizeof(Segment));
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < n_threads_track; t++) {
        memcpy(segs + offsets[t], thread_segs[t], thread_counts[t] * sizeof(Segment));
        free(thread_segs[t]);
    }
    free(thread_segs);
    free(thread_counts);
    free(offsets);

    *t_track = omp_get_wtime() - track_start;

    free(out_spec);
    free(magsq);

    SegmentArray res;
    res.segs = segs;
    res.count = total_segs;
    res.capacity = total_segs;
    return res;
}
