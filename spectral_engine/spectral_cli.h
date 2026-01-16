/* spectral_cli.h - Command-Line Argument Parsing
 * 
 * Parses and validates command-line arguments for spectral processing.
 * Handles different argument layouts for desktop, emulator, and restricted modes.
 */
#ifndef SPECTRAL_CLI_H
#define SPECTRAL_CLI_H

#include "spectral_config.h"
#include "spectral_synth.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Default analysis parameters */
#ifndef DEFAULT_N_FFT
#define DEFAULT_N_FFT       4096
#endif
#ifndef DEFAULT_HOP
#define DEFAULT_HOP         512
#endif
#ifndef DEFAULT_DB_THRESH
#define DEFAULT_DB_THRESH   -60.0f
#endif

/* Parsed command-line options */
typedef struct {
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

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_CLI_H */
