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
    // Parallel FFTW processing with per-thread plans (mirrors vDSP path)
    int n_threads = omp_get_max_threads();
    fftwf_plan* fft_plans = malloc(n_threads * sizeof(fftwf_plan));
    float** thread_in = malloc(n_threads * sizeof(float*));
    fftwf_complex** thread_out = malloc(n_threads * sizeof(fftwf_complex*));
    
    for (int t = 0; t < n_threads; t++) {
        thread_in[t] = fftwf_alloc_real(n_fft);
        thread_out[t] = fftwf_alloc_complex(n_freqs);
        fft_plans[t] = fftwf_plan_dft_r2c_1d(n_fft, thread_in[t], thread_out[t], FFTW_ESTIMATE);
    }
    
    double fft_start = omp_get_wtime();
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float* in_buf = thread_in[tid];
        fftwf_complex* out_buf = thread_out[tid];
        fftwf_plan plan = fft_plans[tid];
        
        #pragma omp for schedule(static)
        for (size_t t = 0; t < n_frames; t++) {
            const float* src = audio + t * hop;
            
            // Apply window function
            for (int i = 0; i < n_fft; i++) {
                in_buf[i] = src[i] * window_func[i];
            }
            
            fftwf_execute(plan);
            
            // Copy to interleaved output format (matches vDSP path)
            float* dest = out_spec + t * n_freqs * 2;
            for (size_t k = 0; k < n_freqs; k++) {
                dest[k * 2] = out_buf[k][0];      // real
                dest[k * 2 + 1] = out_buf[k][1];  // imag
            }
        }
    }
    
    *t_fft = omp_get_wtime() - fft_start;
    
    for (int t = 0; t < n_threads; t++) {
        fftwf_destroy_plan(fft_plans[t]);
        fftwf_free(thread_in[t]);
        fftwf_free(thread_out[t]);
    }
    free(fft_plans);
    free(thread_in);
    free(thread_out);
#endif
    free(window_func);

    double track_start = omp_get_wtime();
    
    // Compute magnitudes and find max in a single pass, optimized with SIMD-friendly code
    size_t total_bins = n_frames * n_freqs;
    int n_threads_track = omp_get_max_threads();
    float* magsq = spectral_aligned_alloc(total_bins * sizeof(float));
    float max_magsq = 0.0f;
    
    #pragma omp parallel reduction(max:max_magsq)
    {
        float local_max = 0.0f;
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < total_bins; i += 4) {
            // Process 4 bins at a time for better ILP
            size_t end = (i + 4 <= total_bins) ? i + 4 : total_bins;
            for (size_t j = i; j < end; j++) {
                float re = out_spec[j * 2];
                float im = out_spec[j * 2 + 1];
                float msq = re*re + im*im;
                magsq[j] = msq;
                if (msq > local_max) local_max = msq;
            }
        }
        if (local_max > max_magsq) max_magsq = local_max;
    }
    
    // Squared threshold for comparison
    float thresh_linear = powf(10.0f, db_thresh / 20.0f);
    float threshsq = thresh_linear * thresh_linear * max_magsq;
    
    // Peak detection constants
    float freq_step = (float)sr / n_fft;
    float two_pi_ts = TWO_PI / sr;
    float inv_hop = 1.0f / hop;
    float freq_step_times_two_pi = freq_step * two_pi_ts;
    float freq_step_df_factor = 0.5f * freq_step * inv_hop * two_pi_ts;
    float hop_float = (float)hop;
    
    size_t frames_per_thread = (n_frames + n_threads_track - 1) / n_threads_track;
    size_t max_segs_per_thread = frames_per_thread * (n_freqs / 8);
    
    Segment** thread_segs = malloc(n_threads_track * sizeof(Segment*));
    size_t* thread_counts = calloc(n_threads_track, sizeof(size_t));
    size_t* thread_capacities = malloc(n_threads_track * sizeof(size_t));
    
    for (int t = 0; t < n_threads_track; t++) {
        thread_segs[t] = malloc(max_segs_per_thread * sizeof(Segment));
        thread_capacities[t] = max_segs_per_thread;
    }

    // Main peak detection loop - optimized for minimal branches and cache efficiency
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Segment* __restrict__ local_segs = thread_segs[tid];
        size_t local_count = 0;
        size_t local_capacity = thread_capacities[tid];
        
        // Use static scheduling with larger chunks for better cache locality
        #pragma omp for schedule(static) nowait
        for (size_t t = 0; t < n_frames - 1; t++) {
            const float* __restrict__ spec_row = out_spec + t * n_freqs * 2;
            const float* __restrict__ row = magsq + t * n_freqs;
            const float* __restrict__ next_row = magsq + (t+1) * n_freqs;
            const float t_hop = t * hop_float;
            
            // Process in larger blocks for better cache usage and instruction-level parallelism
            for (size_t f = 1; f < n_freqs - 1; f++) {
                float prev = row[f-1];
                float curr = row[f];
                float next_mag = row[f+1];
                
                // Fast rejection: most bins fail this test
                if (curr <= threshsq || curr <= prev || curr <= next_mag) {
                    continue;
                }
                
                // This is a peak - load next frame data
                float m0 = next_row[f-1];
                float m1 = next_row[f];
                float m2 = next_row[f+1];
                
                // Find max of next frame triplet
                float max_vsq = (m0 > m1) ? m0 : m1;
                max_vsq = (max_vsq > m2) ? max_vsq : m2;
                
                // Most peaks continue in next frame
                if (max_vsq < threshsq) {
                    continue;
                }
                
                // Grow buffer if needed
                if (__builtin_expect(local_count >= local_capacity, 0)) {
                    local_capacity *= 2;
                    local_segs = realloc(local_segs, local_capacity * sizeof(Segment));
                    thread_segs[tid] = local_segs;
                    thread_capacities[tid] = local_capacity;
                }
                
                // Find best_next index
                int best_idx = (m0 >= m1) ? 0 : 1;
                best_idx = (m2 > ((best_idx == 0) ? m0 : m1)) ? 2 : best_idx;
                int best_next = (int)f + best_idx - 1;
                
                // Compute expensive operations once we know we need them
                float m = sqrtf(curr);
                float max_v = sqrtf(max_vsq);
                
                size_t spec_idx = f * 2;
                float re = spec_row[spec_idx];
                float im = spec_row[spec_idx + 1];
                
                // Store segment with precomputed constants - use direct writes
                Segment* seg = &local_segs[local_count];
                seg->start = t_hop;
                seg->length = hop_float;
                seg->phase = fast_atan2(im, re);
                seg->freq_hz = f * freq_step_times_two_pi;
                seg->df = (best_next - (int)f) * freq_step_df_factor;
                seg->amp = m;
                seg->da = (max_v - m) * inv_hop;
                seg->width = 0.5f;
                local_count++;
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
    free(thread_capacities);
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
