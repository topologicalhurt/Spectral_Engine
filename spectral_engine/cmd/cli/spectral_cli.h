/* spectral_cli.h - Command-Line Argument Parsing
 * 
 * Parses and validates command-line arguments for spectral processing.
 * Handles different argument layouts for desktop, simulation, and restricted modes.
 */
#ifndef SPECTRAL_CLI_H
#define SPECTRAL_CLI_H

#include "spectral_config.h"
#include "spectral_error.h"
#include "spectral_backend.h"
#include "spectral_processing_chain.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Parsed command-line options */
typedef struct SpectralCliOptions {
    const char*    input_path;
    const char*    wavetable_path;
    SpectralTimbre timbre;
    float          stretch;
    float          pitch;
    int            n_fft;
    int            hop;
    float          db_thresh;
    int            n_threads;
    SynthBackend   backend;
    SpectralProcessMask processing_mask;
    int            enable_cache;
    float          start_sec;
    float          end_sec;
    int            use_wavetable;
    int            valid;           /* 1 if parsing succeeded */
    const char*    error_message;   /* Error message if valid==0 */
} SpectralCliOptions;

/* Initialize options with defaults */
void spectral_cli_init(SpectralCliOptions* opts);

/* Parse command-line arguments
 * Returns 1 on success, 0 on failure (check opts->error_message) */
int spectral_cli_parse(SpectralCliOptions* opts, int argc, char** argv);

/* Print usage information for current build mode */
void spectral_cli_print_usage(void);

/* Validate options after parsing
 * Returns 1 if valid, 0 if invalid (sets opts->error_message) */
int spectral_cli_validate(SpectralCliOptions* opts);

/* Runtime config validation (for programmatic use) */
typedef struct SpectralConfigParams {
    int sample_rate;
    float stretch;
    float pitch;
    int timbre;
    size_t buffer_size;
    size_t num_segments;
    int n_threads;
} SpectralConfigParams;

SpectralError spectral_config_validate(const SpectralConfigParams* cfg,
                                       char* error_msg, size_t error_msg_size);
int spectral_config_is_valid(const SpectralConfigParams* cfg);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_CLI_H */
