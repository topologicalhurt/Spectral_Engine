/* main.c - Spectral Audio Processor
 * 
 * Desktop and simulation entry point for spectral audio analysis and synthesis.
 * For embedded ARM targets, the entry point is in api/.
 * 
 * Build targets:
 *   make desktop   - Full build with GPU backends (Metal/CUDA)
 *   make simulate  - Q15 fixed-point simulation for embedded testing
 */

#if defined(SPECTRAL_EMBEDDED) && !defined(SPECTRAL_EMBEDDED_SIMULATION) && \
    !defined(SPECTRAL_USE_EMBEDDED_SYNTH)
#error "main.c should not be compiled for embedded cross-compile targets. Use api/ for embedded targets."
#endif

#include <stdio.h>
#include "spectral_cli.h"
#include "spectral_cli_pipeline.h"
#include "spectral_log.h"
#include "spectral_synth.h"
#include "spectral_utils.h"

int main(int argc, char** argv) {
#if SPECTRAL_IS_EMBEDDED_SIM
    SPECTRAL_LOG_INFO("+------------------------------------------------------------+");
    SPECTRAL_LOG_INFO("|           EMBEDDED TARGET SIMULATION MODE                  |");
    SPECTRAL_LOG_INFO("|  Single-threaded Q15 synthesis (modeling Cortex-M7)        |");
    SPECTRAL_LOG_INFO("+------------------------------------------------------------+\n");
#endif

    /* Parse command-line arguments */
    SpectralCliOptions opts;
    spectral_cli_init(&opts);
    
    if (!spectral_cli_parse(&opts, argc, argv)) {
        if (opts.error_message) {
            spectral_log(SPECTRAL_LOG_LEVEL_ERROR, stderr, 0, 0,
                         "Error: %s\n\n", opts.error_message);
        }
        spectral_cli_print_usage();
        return opts.error_message ? 1 : 0;
    }

    /* Run the processing pipeline */
    SpectralTimingResults timing;
    PipelineError result = spectral_pipeline_run(&opts, &timing);
    
    spectral_backend_cleanup_all();

    if (result != PIPELINE_OK) {
        spectral_log_error_codef(SPECTRAL_ERROR_DOMAIN_PIPELINE, result,
                                 "Pipeline failed (input=%s, backend=%s/%d, timbre=%s/%d, stretch=%.6g, pitch=%.6g)",
                                 opts.input_path ? opts.input_path : "<null>",
                                 spectral_backend_name(opts.backend), (int)opts.backend,
                                 timbre_name(opts.timbre), (int)opts.timbre,
                                 (double)opts.stretch, (double)opts.pitch);
        return (int)result;
    }

    SPECTRAL_LOG_INFO("\nDone.");
    return 0;
}
