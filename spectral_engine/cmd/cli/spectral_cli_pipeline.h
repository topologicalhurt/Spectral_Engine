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

#include "spectral_error.h"
#include <stdint.h>

typedef struct SpectralCliOptions SpectralCliOptions;

#ifdef __cplusplus
extern "C" {
#endif

/* Pipeline timing results.
 *
 * t_fft..t_write are per-stage BUSY (kernel) spans. t_total is their SUM (busy total).
 * wall_total is the real monotonic wall time of the whole run (analysis + synth + backend
 * init + norm + write + inter-stage gaps) — it is the honest headline, and is >= t_total;
 * the difference (idle = wall_total - t_total) is allocation/backend-init/scheduling time
 * the per-stage kernel timers do not see. realtime_x is audio_dur / wall_total (NOT the
 * summed-kernel total, which over-reports realtime by the idle ratio). */
typedef struct {
    double t_fft;
    double t_track;
    double t_synth;
    double t_norm;
    double t_write;
    double t_total;     /* busy total = sum of the per-stage kernel spans */
    double wall_total;  /* honest total = real monotonic wall span of the whole run */
    double audio_dur;
    double realtime_x;  /* audio_dur / wall_total */
} SpectralTimingResults;

/* Run the full processing pipeline
 * 
 * Parameters:
 *   opts    - Parsed and validated command-line options
 *   timing  - Output: timing results (may be NULL)
 * 
 * Returns: PIPELINE_OK on success, negative error code on failure
 */
PipelineError spectral_pipeline_run(const SpectralCliOptions* opts, 
                                     SpectralTimingResults* timing);

/* Print timing results */
void spectral_pipeline_print_timing(const SpectralTimingResults* timing, 
                                    uint32_t segment_count);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_PIPELINE_H */
