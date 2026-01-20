/* main.c - Spectral Audio Processor
 * 
 * Desktop and emulator entry point for spectral audio analysis and synthesis.
 * For embedded ARM targets, the entry point is in api/.
 * 
 * Build targets:
 *   make desktop   - Full build with GPU backends (Metal/CUDA)
 *   make emulator  - Q15 fixed-point emulation for embedded testing
 */

#if defined(SPECTRAL_EMBEDDED) && !defined(SPECTRAL_EMBEDDED_EMULATION) && \
    !defined(SPECTRAL_USE_EMBEDDED_SYNTH)
#error "main.c should not be compiled for embedded cross-compile targets. Use api/ for embedded targets."
#endif

#include <stdio.h>
#include "spectral_cli.h"
#include "spectral_cli_pipeline.h"

/* Emulator mode detection */
#if defined(SPECTRAL_EMBEDDED_EMULATION) || defined(SPECTRAL_USE_EMBEDDED_SYNTH)
#define IS_EMULATOR 1
#else
#define IS_EMULATOR 0
#endif

int main(int argc, char** argv) {
#if IS_EMULATOR
    printf("+------------------------------------------------------------+\n");
    printf("|            EMBEDDED TARGET EMULATOR MODE                   |\n");
    printf("|  Single-threaded Q15 synthesis (simulating Cortex-M7)      |\n");
    printf("+------------------------------------------------------------+\n\n");
#endif

    /* Parse command-line arguments */
    SpectralCliOptions opts;
    spectral_cli_init(&opts);
    
    if (!spectral_cli_parse(&opts, argc, argv)) {
        spectral_cli_print_usage();
        return 1;
    }
    
    /* Validate options */
    if (!spectral_cli_validate(&opts)) {
        return 1;
    }
    
    /* Run the processing pipeline */
    SpectralTimingResults timing;
    PipelineError result = spectral_pipeline_run(&opts, &timing);
    
    if (result != PIPELINE_OK) {
        printf("Error: Pipeline failed (code %d)\n", result);
        return (int)result;
    }
    
    printf("\nDone.\n");
    return 0;
}
