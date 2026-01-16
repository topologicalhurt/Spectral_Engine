/* spectral_cli_pipeline.h - Main Processing Pipeline
 * 
 * Orchestrates the spectral processing pipeline:
 *   1. Input: Load audio file or pre-analyzed segments
 *   2. Analysis: FFT analysis and peak tracking (desktop only)
 *   3. Synthesis: Render segments with selected backend
 *   4. Output: Normalize and write audio file
 */
#ifndef SPECTRAL_PIPELINE_H
#define SPECTRAL_PIPELINE_H

#include "spectral_cli.h"
#include "spectral_common.h"
#include "spectral_wavetable.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Pipeline timing results */
typedef struct {
    double t_fft;
    double t_track;
    double t_synth;
    double t_norm;
    double t_total;
    double audio_dur;
    double realtime_x;
} SpectralTimingResults;

/* Pipeline result codes */
typedef enum {
    PIPELINE_OK = 0,
    PIPELINE_ERR_INPUT = -1,
    PIPELINE_ERR_ANALYSIS = -2,
    PIPELINE_ERR_SYNTHESIS = -3,
    PIPELINE_ERR_OUTPUT = -4,
    PIPELINE_ERR_WAVETABLE = -5,
    PIPELINE_ERR_MEMORY = -6
} PipelineResult;

/* Run the full processing pipeline
 * 
 * Parameters:
 *   opts    - Parsed and validated command-line options
 *   timing  - Output: timing results (may be NULL)
 * 
 * Returns: PIPELINE_OK on success, negative error code on failure
 */
PipelineResult spectral_pipeline_run(const SpectralCliOptions* opts, 
                                     SpectralTimingResults* timing);

/* Print timing results */
void spectral_pipeline_print_timing(const SpectralTimingResults* timing, 
                                    uint32_t segment_count);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_PIPELINE_H */
