/* spectral_analysis.c - analysis dispatch
 *
 * Chooses between:
 *  - full-matrix STFT path (smaller/medium inputs)
 *  - chunked fused FFT+tracking path (large inputs)
 */
#include "spectral_analysis_internal.h"
#include "spectral_log.h"

SpectralError spectral_analysis_window_context_init(SpectralAnalysisWindowContext* ctx,
                                                    size_t n_fft,
                                                    SpectralWindowType type)
{
    size_t bytes = 0;
    const SpectralWindowDescriptor* desc = NULL;

    if (!ctx || n_fft == 0u) return SPECTRAL_ERR_PARAM;
    *ctx = (SpectralAnalysisWindowContext){0};

    desc = spectral_window_descriptor(type);
    if (!desc) return SPECTRAL_ERR_PARAM;

    if (!spectral_array_bytes(n_fft, sizeof(float), &bytes)) {
        return SPECTRAL_ERR_OVERFLOW;
    }

    ctx->samples = (float*)spectral_aligned_alloc(bytes);
    if (!ctx->samples) return SPECTRAL_ERR_MEMORY;

    if (spectral_window_generate(ctx->samples, n_fft, type) != SPECTRAL_OK) {
        spectral_analysis_window_context_free(ctx);
        return SPECTRAL_ERR_PARAM;
    }

    ctx->metrics = spectral_window_metrics(ctx->samples, n_fft);
    ctx->descriptor = desc;
    ctx->bytes = bytes;
    return SPECTRAL_OK;
}

void spectral_analysis_window_context_free(SpectralAnalysisWindowContext* ctx)
{
    if (!ctx) return;
    free(ctx->samples);
    *ctx = (SpectralAnalysisWindowContext){0};
}

void spectral_analysis_window_context_apply_magsq_scales(const SpectralAnalysisWindowContext* ctx,
                                                         SpectralFftResources* res)
{
    if (!ctx || !res) return;
    spectral_fft_resources_set_magsq_scales(res,
                                            ctx->metrics.endpoint_bin_magsq_scale,
                                            ctx->metrics.positive_bin_magsq_scale);
}

SegmentArray spectral_analysis_return_empty(double* t_fft, double* t_track)
{
    if (t_fft) *t_fft = 0;
    if (t_track) *t_track = 0;
    return (SegmentArray)SEGMENT_ARRAY_EMPTY;
}

int spectral_analysis_estimate_fft_bytes(size_t frame_count,
                                         size_t n_fft,
                                         size_t n_freqs,
                                         size_t* out_bytes)
{
    size_t frame_fft_floats = 0;
    size_t read_input_floats = 0;
    size_t read_window_floats = 0;
    size_t write_windowed_floats = 0;
    size_t read_spectrum_floats = 0;
    size_t write_magsq_floats = 0;
    size_t write_phase_floats = 0;
    size_t total_floats = 0;

    if (!out_bytes) return 0;

    if (!spectral_size_mul(frame_count, n_fft, &read_input_floats) ||
        !spectral_size_mul(frame_count, n_fft, &read_window_floats) ||
        !spectral_size_mul(frame_count, n_fft, &write_windowed_floats) ||
        !spectral_size_mul(frame_count, n_freqs, &frame_fft_floats) ||
        !spectral_size_mul(frame_fft_floats, 2u, &read_spectrum_floats) ||
        !spectral_size_mul(frame_count, n_freqs, &write_magsq_floats) ||
        !spectral_size_mul(frame_count, n_freqs, &write_phase_floats)) {
        return 0;
    }

    if (!spectral_size_add(read_input_floats, read_window_floats, &total_floats) ||
        !spectral_size_add(total_floats, write_windowed_floats, &total_floats) ||
        !spectral_size_add(total_floats, read_spectrum_floats, &total_floats) ||
        !spectral_size_add(total_floats, write_magsq_floats, &total_floats) ||
        !spectral_size_add(total_floats, write_phase_floats, &total_floats)) {
        return 0;
    }

    return spectral_size_mul(total_floats, sizeof(float), out_bytes);
}

SegmentArray analyze_audio(const float* audio, size_t n_samples, int sr,
                           int n_fft, int hop, float db_thresh,
                           double* t_fft, double* t_track)
{
    size_t n_frames = 0;
    size_t n_freqs = 0;
    size_t total_bins = 0;
    size_t n_fft_size = 0;
    int use_fused_path = 0;
    SegmentArray result = (SegmentArray)SEGMENT_ARRAY_EMPTY;

    if (!audio || !t_fft || !t_track ||
        sr < SPECTRAL_MIN_SAMPLE_RATE || sr > SPECTRAL_MAX_SAMPLE_RATE ||
        n_fft < SPECTRAL_MIN_FFT_SIZE || hop <= 0 ||
        !isfinite(db_thresh) || n_samples < (size_t)n_fft) {
        return spectral_analysis_return_empty(t_fft, t_track);
    }

    n_fft_size = (size_t)n_fft;
    if ((n_fft_size & (n_fft_size - 1u)) != 0u) {
        return spectral_analysis_return_empty(t_fft, t_track);
    }

    n_frames = (n_samples - n_fft_size) / (size_t)hop + 1;
    n_freqs = (size_t)n_fft / 2u + 1u;
    if (n_freqs < 3u || !spectral_size_mul(n_frames, n_freqs, &total_bins)) {
        return spectral_analysis_return_empty(t_fft, t_track);
    }

    use_fused_path = (total_bins > SPECTRAL_STFT_CHUNK_THRESHOLD);
    SPECTRAL_LOG_INFO("Analysis crossover: bins=%zu threshold=%zu path=%s",
                      total_bins,
                      (size_t)SPECTRAL_STFT_CHUNK_THRESHOLD,
                      use_fused_path ? "spsc_pipeline" : "full_matrix");

    if (use_fused_path) {
        result = spectral_analysis_run_fused(audio, n_samples, sr, n_fft, hop, db_thresh,
                                             n_frames, n_freqs, t_fft, t_track);
    } else {
        result = spectral_analysis_run_full(audio, n_samples, sr, n_fft, hop, db_thresh,
                                            n_frames, n_freqs, t_fft, t_track);
    }

    {
        double analysis_total = (*t_fft) + (*t_track);
        double fft_share_pct = (analysis_total > 0.0) ? (100.0 * (*t_fft) / analysis_total) : 0.0;
        double track_share_pct = (analysis_total > 0.0) ? (100.0 * (*t_track) / analysis_total) : 0.0;
        SPECTRAL_LOG_INFO("Analysis summary: path=%s fft=%.3fms track=%.3fms total=%.3fms fft_share=%.1f%% track_share=%.1f%% segments=%u",
                          use_fused_path ? "spsc_pipeline" : "full_matrix",
                          (*t_fft) * SPECTRAL_MILLIS_PER_SECOND_D,
                          (*t_track) * SPECTRAL_MILLIS_PER_SECOND_D,
                          analysis_total * SPECTRAL_MILLIS_PER_SECOND_D,
                          fft_share_pct,
                          track_share_pct,
                          result.count);
    }

    return result;
}
