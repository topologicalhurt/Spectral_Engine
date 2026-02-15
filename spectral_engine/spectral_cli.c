/* spectral_cli.c - Command-Line Argument Parsing Implementation
 * 
 * Handles parsing for desktop, emulator, and restricted build modes.
 */
#include "spectral_cli.h"
#include "spectral_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "spectral_omp.h"

#define CLI_MAX_ARGV  64  /* Max argv entries for stack-allocated effective argv */

void spectral_cli_init(SpectralCliOptions* opts) {
    if (!opts) return;
    
    opts->input_path = NULL;
    opts->wavetable_path = NULL;
    opts->timbre = TIMBRE_SINE;
    opts->stretch = 1.0f;
    opts->pitch = 0.0f;
    opts->n_fft = DEFAULT_N_FFT;
    opts->hop = DEFAULT_HOP;
    opts->db_thresh = DEFAULT_DB_THRESH;
    opts->n_threads = omp_get_max_threads();
    opts->backend = BACKEND_AUTO;
    opts->processing_mask = SPECTRAL_PROC_DEFAULT;
    opts->start_sec = 0.0f;
    opts->end_sec = -1.0f;
    opts->use_wavetable = 0;
    opts->valid = 0;
    opts->error_message = NULL;
    
#if SPECTRAL_IS_EMULATOR || SPECTRAL_RESTRICTED_MODE
    opts->n_threads = 1;  /* Force single-threaded for embedded modes */
    opts->backend = BACKEND_CPU;
#endif
}

int spectral_cli_parse(SpectralCliOptions* opts, int argc, char** argv) {
    if (!opts || argc < 2) {
        if (opts) {
            opts->valid = 0;
            opts->error_message = "No input file specified";
        }
        return 0;
    }
    
    spectral_cli_init(opts);
    opts->input_path = argv[1];
    
    /* Parse named options and build effective positional argv */
    int eff_argc = argc;
    char* eff_argv_buf[CLI_MAX_ARGV];
    char** eff_argv_heap = NULL;
    char** eff_argv = argv;
    int* skip = (int*)calloc((size_t)argc, sizeof(int));
    if (!skip) {
        opts->valid = 0;
        opts->error_message = "Out of memory filtering argv";
        return 0;
    }

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--wavetable") == 0) {
            if (i + 1 >= argc) {
                free(skip);
                opts->valid = 0;
                opts->error_message = "Missing wavetable path after -w/--wavetable";
                return 0;
            }
            opts->wavetable_path = argv[i + 1];
            opts->use_wavetable = 1;
            skip[i] = 1;
            skip[i + 1] = 1;
            i++;
            continue;
        }
        if (strcmp(argv[i], "-pm") == 0 || strcmp(argv[i], "--proc-mask") == 0 ||
            strcmp(argv[i], "--process-mask") == 0) {
            if (i + 1 >= argc) {
                free(skip);
                opts->valid = 0;
                opts->error_message = "Missing mask value after -pm/--proc-mask";
                return 0;
            }
            if (!spectral_process_mask_parse(argv[i + 1], &opts->processing_mask)) {
                free(skip);
                opts->valid = 0;
                opts->error_message = "Invalid processing mask. Use names (e.g. serra_smith_1990,johnston_1988), none/default/all, or numeric bitmask";
                return 0;
            }
            skip[i] = 1;
            skip[i + 1] = 1;
            i++;
            continue;
        }
    }

    /* Use stack buffer if it fits, else heap-allocate */
    char** dst;
    if (argc <= CLI_MAX_ARGV) {
        dst = eff_argv_buf;
    } else {
        eff_argv_heap = (char**)malloc(argc * sizeof(char*));
        if (!eff_argv_heap) {
            free(skip);
            opts->valid = 0;
            opts->error_message = "Out of memory filtering argv";
            return 0;
        }
        dst = eff_argv_heap;
    }

    eff_argc = 0;
    for (int i = 0; i < argc; i++) {
        if (skip[i]) continue;
        dst[eff_argc++] = argv[i];
    }

    free(skip);
    eff_argv = dst;

    argc = eff_argc;
    argv = eff_argv;
    
#if SPECTRAL_RESTRICTED_MODE
    /* Restricted mode: timbre stretch pitch threads start end */
    int timbre_arg = (argc > 2) ? atoi(argv[2]) : 0;
    opts->timbre = (timbre_arg >= TIMBRE_MIN && timbre_arg <= TIMBRE_MAX) 
                 ? (SpectralTimbre)timbre_arg : TIMBRE_SINE;
    opts->stretch = (argc > 3) ? (float)atof(argv[3]) : 1.0f;
    opts->pitch = (argc > 4) ? (float)atof(argv[4]) : 0.0f;
    opts->n_threads = 1;
    opts->backend = BACKEND_CPU;
    
#elif SPECTRAL_IS_EMULATOR
    /* Emulator mode: timbre stretch pitch n_fft hop db_thresh start end */
    int timbre_arg = (argc > 2) ? atoi(argv[2]) : 0;
    opts->timbre = (timbre_arg >= TIMBRE_MIN && timbre_arg <= TIMBRE_MAX) 
                 ? (SpectralTimbre)timbre_arg : TIMBRE_SINE;
    opts->stretch = (argc > 3) ? (float)atof(argv[3]) : 1.0f;
    opts->pitch = (argc > 4) ? (float)atof(argv[4]) : 0.0f;
    opts->n_fft = (argc > 5) ? atoi(argv[5]) : DEFAULT_N_FFT;
    opts->hop = (argc > 6) ? atoi(argv[6]) : DEFAULT_HOP;
    opts->db_thresh = (argc > 7) ? (float)atof(argv[7]) : DEFAULT_DB_THRESH;
    opts->n_threads = 1;
    opts->backend = BACKEND_CPU;
    opts->start_sec = (argc > 8) ? (float)atof(argv[8]) : 0.0f;
    opts->end_sec = (argc > 9) ? (float)atof(argv[9]) : -1.0f;
    
#else
    /* Desktop mode: timbre stretch pitch n_fft hop db_thresh threads backend start end */
    int timbre_arg = (argc > 2) ? atoi(argv[2]) : 0;
    opts->timbre = (timbre_arg >= TIMBRE_MIN && timbre_arg <= TIMBRE_MAX) 
                 ? (SpectralTimbre)timbre_arg : TIMBRE_SINE;
    opts->stretch = (argc > 3) ? (float)atof(argv[3]) : 1.0f;
    opts->pitch = (argc > 4) ? (float)atof(argv[4]) : 0.0f;
    opts->n_fft = (argc > 5) ? atoi(argv[5]) : DEFAULT_N_FFT;
    opts->hop = (argc > 6) ? atoi(argv[6]) : DEFAULT_HOP;
    opts->db_thresh = (argc > 7) ? (float)atof(argv[7]) : DEFAULT_DB_THRESH;
    opts->n_threads = (argc > 8) ? atoi(argv[8]) : omp_get_max_threads();
    opts->backend = (argc > 9) ? (SynthBackend)atoi(argv[9]) : BACKEND_AUTO;
    opts->start_sec = (argc > 10) ? (float)atof(argv[10]) : 0.0f;
    opts->end_sec = (argc > 11) ? (float)atof(argv[11]) : -1.0f;
#endif
    
    free(eff_argv_heap);  /* NULL-safe */
    opts->valid = 1;
    return spectral_cli_validate(opts);
}

int spectral_cli_validate(SpectralCliOptions* opts) {
    if (!opts) return 0;
    
#if !SPECTRAL_RESTRICTED_MODE
    /* Validate FFT size */
    if ((opts->n_fft & (opts->n_fft - 1)) != 0 || opts->n_fft < 64) {
        opts->valid = 0;
        opts->error_message = "n_fft must be power of 2 >= 64";
        return 0;
    }

    if (opts->hop <= 0 || opts->hop > opts->n_fft) {
        opts->valid = 0;
        opts->error_message = "hop must be > 0 and <= n_fft";
        return 0;
    }
#endif
    
    /* Validate timbre */
    if (opts->timbre < TIMBRE_MIN || opts->timbre > TIMBRE_MAX) {
        opts->valid = 0;
        opts->error_message = "timbre must be 0-7";
        return 0;
    }

    if (!isfinite(opts->stretch) || opts->stretch <= 0.0f) {
        opts->valid = 0;
        opts->error_message = "stretch must be > 0";
        return 0;
    }

    if (!isfinite(opts->pitch)) {
        opts->valid = 0;
        opts->error_message = "pitch must be finite";
        return 0;
    }

    if (!isfinite(opts->start_sec) || !isfinite(opts->end_sec)) {
        opts->valid = 0;
        opts->error_message = "start/end times must be finite";
        return 0;
    }
    
    /* Validate backend enum */
    if ((int)opts->backend < BACKEND_AUTO || (int)opts->backend > BACKEND_EXPORT) {
        opts->valid = 0;
        opts->error_message = "backend must be 0-4 (auto/cpu/metal/cuda/export)";
        return 0;
    }

#if SPECTRAL_RESTRICTED_MODE
    /* Restricted mode: force CPU backend */
    if (opts->backend != BACKEND_CPU && opts->backend != BACKEND_EXPORT) {
        opts->valid = 0;
        opts->error_message = "Restricted mode only supports CPU backend";
        return 0;
    }
#else
    /* Desktop: check wavetable + GPU compatibility */
    if (opts->use_wavetable) {
        if (opts->backend == BACKEND_METAL || opts->backend == BACKEND_CUDA) {
            opts->valid = 0;
            opts->error_message = "GPU backends do not support wavetable synthesis";
            return 0;
        }
        if (opts->backend == BACKEND_AUTO) {
            opts->backend = BACKEND_CPU;  /* Force CPU for wavetable */
        }
    }
#endif
    
    /* Validate thread count */
    if (opts->n_threads < 1) opts->n_threads = 1;
    int hw_threads = omp_get_max_threads();
    if (opts->n_threads > hw_threads) opts->n_threads = hw_threads;
    
    opts->valid = 1;
    return 1;
}

void spectral_cli_print_usage(void) {
#if SPECTRAL_RESTRICTED_MODE
    printf("Usage: ./spectral_embedded segments.bin [options]\n\n");
    printf("Restricted Mode Build (embedded-compatible)\n");
    printf("  - CPU synthesis only (no Metal/CUDA)\n");
    printf("  - Loads pre-analyzed segment files\n\n");
    printf("Positional args (legacy):\n");
    printf("  timbre stretch pitch threads start end\n\n");
    printf("Wavetable option:\n");
    printf("  -w <file>  Use wavetable file\n\n");
    printf("Processing mask option:\n");
    printf("  -pm, --proc-mask <spec>  Optional processing chain mask\n\n");
    printf("Defaults: timbre=0 stretch=1.0 pitch=0\n");
    printf("Timbres: 0=sine 1=saw 2=square 3=tri 4=asin 5=para 6=quant 7=pwm\n");
    
#elif SPECTRAL_IS_EMULATOR
    printf("Usage: ./spectral_emulator input.wav [options]\n\n");
    printf("Embedded Target Emulator Build\n");
    printf("  - Q15 fixed-point synthesis (same as ARM Cortex-M)\n");
    printf("  - SINGLE-THREADED (simulates embedded constraints)\n\n");
    printf("Positional args:\n");
    printf("  timbre stretch pitch n_fft hop db_thresh start end\n\n");
    printf("Processing mask option:\n");
    printf("  -pm, --proc-mask <spec>  Optional processing chain mask\n\n");
    printf("Defaults: timbre=0 stretch=1.0 pitch=0 n_fft=%d hop=%d thresh=%.1f\n",
           DEFAULT_N_FFT, DEFAULT_HOP, DEFAULT_DB_THRESH);
    printf("Timbres: 0=sine 1=saw 2=square 3=tri 4=asin 5=para 6=quant 7=pwm\n");
    
#else
    printf("Usage: ./spectral input.wav [options]\n\n");
    printf("Positional args (legacy):\n");
    printf("  timbre stretch pitch n_fft hop db_thresh threads backend start end\n\n");
    printf("Named options:\n");
    printf("  -w <file>  Use wavetable file (.spwt, .hex, .bin)\n\n");
    printf("  -pm, --proc-mask <spec>  Processing chain mask (comma-separated names or integer bitmask)\n");
    printf("      Names: serra_smith_1990, johnston_1988, adaptive_track_density, reassigned,\n");
    printf("             hybrid_render, event_bucket, higher_order_interp, qnoise_shaping\n\n");
    printf("      Special: none, default, all\n\n");
    printf("Defaults: timbre=0 stretch=1.0 pitch=0 n_fft=%d hop=%d thresh=%.1f\n",
           DEFAULT_N_FFT, DEFAULT_HOP, DEFAULT_DB_THRESH);
#if HAS_METAL
    printf("Backend: 0=auto 1=cpu 2=metal 3=cuda 4=export\n");
#elif HAS_CUDA
    printf("Backend: 0=auto 1=cpu 3=cuda 4=export\n");
#else
    printf("Backend: 0=auto 1=cpu 4=export\n");
#endif
    printf("Timbres: 0=sine 1=saw 2=square 3=tri 4=asin 5=para 6=quant 7=pwm\n");
#endif
}

 /* Runtime Config Validation (for programmatic use) */
SpectralError spectral_config_validate(const SpectralConfigParams* cfg,
                                       char* error_msg, size_t error_msg_size)
{
    if (!cfg) {
        if (error_msg && error_msg_size > 0)
            snprintf(error_msg, error_msg_size, "NULL config");
        return SPECTRAL_ERR_PARAM;
    }
    
    if (cfg->sample_rate < 8000 || cfg->sample_rate > 192000) {
        if (error_msg && error_msg_size > 0)
            snprintf(error_msg, error_msg_size, "sample_rate %d out of range", cfg->sample_rate);
        return SPECTRAL_ERR_PARAM;
    }
    
    if (!isfinite(cfg->stretch) || cfg->stretch <= 0.0f || cfg->stretch > 1000.0f) {
        if (error_msg && error_msg_size > 0)
            snprintf(error_msg, error_msg_size, "stretch %.2f invalid", cfg->stretch);
        return SPECTRAL_ERR_PARAM;
    }
    
    if (!isfinite(cfg->pitch) || cfg->pitch < -48.0f || cfg->pitch > 48.0f) {
        if (error_msg && error_msg_size > 0)
            snprintf(error_msg, error_msg_size, "pitch %.1f invalid", cfg->pitch);
        return SPECTRAL_ERR_PARAM;
    }
    
    if (cfg->timbre < TIMBRE_MIN || cfg->timbre > TIMBRE_MAX) {
        if (error_msg && error_msg_size > 0)
            snprintf(error_msg, error_msg_size, "timbre %d invalid", cfg->timbre);
        return SPECTRAL_ERR_PARAM;
    }
    
    if (cfg->buffer_size == 0) {
        if (error_msg && error_msg_size > 0)
            snprintf(error_msg, error_msg_size, "buffer_size is zero");
        return SPECTRAL_ERR_PARAM;
    }
    
#if SPECTRAL_EMBEDDED
    if (cfg->buffer_size > 4096) {
        if (error_msg && error_msg_size > 0)
            snprintf(error_msg, error_msg_size, "buffer_size %zu too large", cfg->buffer_size);
        return SPECTRAL_ERR_OVERFLOW;
    }
#endif
    
#if !SPECTRAL_EMBEDDED
    if (cfg->n_threads < 1 || cfg->n_threads > 256) {
        if (error_msg && error_msg_size > 0)
            snprintf(error_msg, error_msg_size, "n_threads %d out of range", cfg->n_threads);
        return SPECTRAL_ERR_PARAM;
    }
#endif
    
    return SPECTRAL_OK;
}

int spectral_config_is_valid(const SpectralConfigParams* cfg)
{
    return spectral_config_validate(cfg, NULL, 0) == SPECTRAL_OK;
}
