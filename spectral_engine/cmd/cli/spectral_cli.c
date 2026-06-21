/* spectral_cli.c - Command-Line Argument Parsing Implementation
 * 
 * Handles parsing for desktop, simulation, and restricted build modes.
 */
#include "spectral_cli.h"
#include "spectral_metal.h"
#include "spectral_cuda.h"
#include "spectral_synth.h"
#include "spectral_utils.h"
#include "spectral_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "spectral_omp.h"

enum {
    CLI_MAX_ARGV = 64,  /* Max argv entries for stack-allocated effective argv */
    CLI_ERROR_MESSAGE_MAX = 256,
    CLI_VALUE_MESSAGE_MAX = 96
};

static SPECTRAL_THREAD_LOCAL char g_cli_error_message[CLI_ERROR_MESSAGE_MAX];

typedef enum {
    CLI_COMMON_VALIDATE_OK = 0,
    CLI_COMMON_VALIDATE_TIMBRE,
    CLI_COMMON_VALIDATE_STRETCH,
    CLI_COMMON_VALIDATE_PITCH_FINITE,
    CLI_COMMON_VALIDATE_PITCH_RANGE
} CliCommonValidationResult;

static int cli_fail(SpectralCliOptions* opts, const char* error_message) {
    if (!opts) return 0;
    opts->valid = 0;
    opts->error_message = error_message;
    return 0;
}

static const char* cli_safe_value(const char* value) {
    return value ? value : "<null>";
}

static const char* cli_format_pitch_range(char* buf, size_t buf_size) {
    if (!buf || buf_size == 0) return NULL;
    snprintf(buf, buf_size, "%.1f..%.1f semitones",
             (double)SPECTRAL_MIN_PITCH, (double)SPECTRAL_MAX_PITCH);
    return buf;
}

static int cli_fail_expected_actual(
    SpectralCliOptions* opts,
    const char* field,
    const char* expected,
    const char* actual)
{
    if (!opts) return 0;
    spectral_format_expected_actual(g_cli_error_message, sizeof(g_cli_error_message),
                                    field, expected, cli_safe_value(actual));
    return cli_fail(opts, g_cli_error_message);
}

static SpectralError cli_config_fail(
    SpectralError error_code,
    char* error_msg,
    size_t error_msg_size,
    const char* format,
    ...)
{
    if (error_msg && error_msg_size > 0 && format) {
        va_list args;
        va_start(args, format);
        vsnprintf(error_msg, error_msg_size, format, args);
        va_end(args);
    }
    return error_code;
}

static int cli_fail_common_validation(
    SpectralCliOptions* opts,
    CliCommonValidationResult common_validation)
{
    char actual[CLI_VALUE_MESSAGE_MAX] = {0};
    char expected[CLI_VALUE_MESSAGE_MAX] = {0};

    switch (common_validation) {
    case CLI_COMMON_VALIDATE_OK:
        return 1;
    case CLI_COMMON_VALIDATE_TIMBRE:
        snprintf(actual, sizeof(actual), "%d", opts ? (int)opts->timbre : 0);
        return cli_fail_expected_actual(opts, "timbre", "integer 0..7 (sine..pwm)", actual);
    case CLI_COMMON_VALIDATE_STRETCH:
        snprintf(actual, sizeof(actual), "%.6g", opts ? (double)opts->stretch : 0.0);
        return cli_fail_expected_actual(opts, "stretch", "finite float > 0", actual);
    case CLI_COMMON_VALIDATE_PITCH_FINITE:
        snprintf(actual, sizeof(actual), "%.6g", opts ? (double)opts->pitch : 0.0);
        return cli_fail_expected_actual(opts, "pitch", "finite float", actual);
    case CLI_COMMON_VALIDATE_PITCH_RANGE:
        cli_format_pitch_range(expected, sizeof(expected));
        snprintf(actual, sizeof(actual), "%.6g", opts ? (double)opts->pitch : 0.0);
        return cli_fail_expected_actual(opts, "pitch", expected, actual);
    }
    return cli_fail(opts, "invalid synthesis parameters");
}

static SpectralError cli_config_fail_common_validation(
    const SpectralConfigParams* cfg,
    CliCommonValidationResult common_validation,
    char* error_msg,
    size_t error_msg_size)
{
    char actual[CLI_VALUE_MESSAGE_MAX] = {0};
    char expected[CLI_VALUE_MESSAGE_MAX] = {0};
    char message[CLI_ERROR_MESSAGE_MAX] = {0};

    switch (common_validation) {
    case CLI_COMMON_VALIDATE_OK:
        return SPECTRAL_OK;
    case CLI_COMMON_VALIDATE_TIMBRE:
        snprintf(actual, sizeof(actual), "%d", cfg->timbre);
        spectral_format_expected_actual(message, sizeof(message),
                                        "timbre", "integer 0..7 (sine..pwm)", actual);
        return cli_config_fail(SPECTRAL_ERR_PARAM, error_msg, error_msg_size, "%s", message);
    case CLI_COMMON_VALIDATE_STRETCH:
        snprintf(actual, sizeof(actual), "%.6g", (double)cfg->stretch);
        spectral_format_expected_actual(message, sizeof(message),
                                        "stretch", "finite float > 0", actual);
        return cli_config_fail(SPECTRAL_ERR_PARAM, error_msg, error_msg_size, "%s", message);
    case CLI_COMMON_VALIDATE_PITCH_FINITE:
        snprintf(actual, sizeof(actual), "%.6g", (double)cfg->pitch);
        spectral_format_expected_actual(message, sizeof(message),
                                        "pitch", "finite float", actual);
        return cli_config_fail(SPECTRAL_ERR_PARAM, error_msg, error_msg_size, "%s", message);
    case CLI_COMMON_VALIDATE_PITCH_RANGE:
        cli_format_pitch_range(expected, sizeof(expected));
        snprintf(actual, sizeof(actual), "%.6g", (double)cfg->pitch);
        spectral_format_expected_actual(message, sizeof(message), "pitch", expected, actual);
        return cli_config_fail(SPECTRAL_ERR_PARAM, error_msg, error_msg_size, "%s", message);
    }
    return cli_config_fail(SPECTRAL_ERR_PARAM, error_msg, error_msg_size,
                           "invalid synthesis parameters");
}

static int cli_parse_optional_int_arg(
    SpectralCliOptions* opts,
    int argc,
    char** argv,
    int arg_index,
    int* dst,
    const char* field_name)
{
    if (!opts || !dst || !field_name) return 0;
    if (arg_index >= 0 && argc > arg_index && !parse_i32_arg(argv[arg_index], dst)) {
        return cli_fail_expected_actual(opts, field_name, "integer",
                                        cli_safe_value(argv[arg_index]));
    }
    return 1;
}

static int cli_parse_optional_float_arg(
    SpectralCliOptions* opts,
    int argc,
    char** argv,
    int arg_index,
    float* dst,
    const char* field_name)
{
    if (!opts || !dst || !field_name) return 0;
    if (arg_index >= 0 && argc > arg_index && !parse_f32_arg(argv[arg_index], dst)) {
        return cli_fail_expected_actual(opts, field_name, "finite float",
                                        cli_safe_value(argv[arg_index]));
    }
    return 1;
}

#if SPECTRAL_RESTRICTED_MODE
enum {
    CLI_POS_TIMBRE = 2,
    CLI_POS_STRETCH = 3,
    CLI_POS_PITCH = 4,
    CLI_POS_N_FFT = -1,
    CLI_POS_HOP = -1,
    CLI_POS_DB_THRESH = -1,
    CLI_POS_THREADS = -1,
    CLI_POS_BACKEND = -1,
    CLI_POS_START_SEC = -1,
    CLI_POS_END_SEC = -1
};
#elif SPECTRAL_IS_EMBEDDED_SIM
enum {
    CLI_POS_TIMBRE = 2,
    CLI_POS_STRETCH = 3,
    CLI_POS_PITCH = 4,
    CLI_POS_N_FFT = 5,
    CLI_POS_HOP = 6,
    CLI_POS_DB_THRESH = 7,
    CLI_POS_THREADS = -1,
    CLI_POS_BACKEND = -1,
    CLI_POS_START_SEC = 8,
    CLI_POS_END_SEC = 9
};
#else
enum {
    CLI_POS_TIMBRE = 2,
    CLI_POS_STRETCH = 3,
    CLI_POS_PITCH = 4,
    CLI_POS_N_FFT = 5,
    CLI_POS_HOP = 6,
    CLI_POS_DB_THRESH = 7,
    CLI_POS_THREADS = 8,
    CLI_POS_BACKEND = 9,
    CLI_POS_START_SEC = 10,
    CLI_POS_END_SEC = 11
};
#endif

static int cli_max_positional_arg_index(void) {
    int max_index = 1; /* argv[0]=program, argv[1]=input path */
    const int indices[] = {
        CLI_POS_TIMBRE,
        CLI_POS_STRETCH,
        CLI_POS_PITCH,
        CLI_POS_N_FFT,
        CLI_POS_HOP,
        CLI_POS_DB_THRESH,
        CLI_POS_THREADS,
        CLI_POS_BACKEND,
        CLI_POS_START_SEC,
        CLI_POS_END_SEC
    };
    const size_t count = sizeof(indices) / sizeof(indices[0]);
    for (size_t i = 0; i < count; i++) {
        if (indices[i] > max_index) {
            max_index = indices[i];
        }
    }
    return max_index;
}

static SpectralTimbre cli_timbre_from_int_or_default(int timbre_value) {
    return (timbre_value >= TIMBRE_SINE && timbre_value <= TIMBRE_PWM)
        ? (SpectralTimbre)timbre_value
        : TIMBRE_SINE;
}

static int cli_parse_common_positional_synth_args(
    SpectralCliOptions* opts,
    int argc,
    char** argv,
    int timbre_arg_index,
    int stretch_arg_index,
    int pitch_arg_index)
{
    int timbre_arg = 0;

    if (!cli_parse_optional_int_arg(opts, argc, argv, timbre_arg_index, &timbre_arg,
                                    "timbre")) {
        return 0;
    }
    opts->timbre = cli_timbre_from_int_or_default(timbre_arg);
    if (!cli_parse_optional_float_arg(opts, argc, argv, stretch_arg_index, &opts->stretch,
                                      "stretch")) {
        return 0;
    }
    if (!cli_parse_optional_float_arg(opts, argc, argv, pitch_arg_index, &opts->pitch,
                                      "pitch")) {
        return 0;
    }
    return 1;
}

static int cli_parse_optional_time_window_args(
    SpectralCliOptions* opts,
    int argc,
    char** argv,
    int start_arg_index,
    int end_arg_index)
{
    if (!cli_parse_optional_float_arg(opts, argc, argv, start_arg_index, &opts->start_sec,
                                      "start_sec")) {
        return 0;
    }
    if (!cli_parse_optional_float_arg(opts, argc, argv, end_arg_index, &opts->end_sec,
                                      "end_sec")) {
        return 0;
    }
    return 1;
}

static int cli_parse_mode_positional_args(SpectralCliOptions* opts, int argc, char** argv) {
    int backend_arg = (int)opts->backend;

    if (!cli_parse_common_positional_synth_args(
            opts, argc, argv, CLI_POS_TIMBRE, CLI_POS_STRETCH, CLI_POS_PITCH)) {
        return 0;
    }
    if (!cli_parse_optional_int_arg(opts, argc, argv, CLI_POS_N_FFT, &opts->n_fft,
                                    "n_fft")) {
        return 0;
    }
    if (!cli_parse_optional_int_arg(opts, argc, argv, CLI_POS_HOP, &opts->hop,
                                    "hop")) {
        return 0;
    }
    if (!cli_parse_optional_float_arg(opts, argc, argv, CLI_POS_DB_THRESH, &opts->db_thresh,
                                      "db_thresh")) {
        return 0;
    }
    if (!cli_parse_optional_int_arg(opts, argc, argv, CLI_POS_THREADS, &opts->n_threads,
                                    "threads")) {
        return 0;
    }
    if (!cli_parse_optional_int_arg(opts, argc, argv, CLI_POS_BACKEND, &backend_arg,
                                    "backend")) {
        return 0;
    }
    if (!cli_parse_optional_time_window_args(
            opts, argc, argv, CLI_POS_START_SEC, CLI_POS_END_SEC)) {
        return 0;
    }

    if (CLI_POS_BACKEND >= 0) {
        opts->backend = (SynthBackend)backend_arg;
    }
#if SPECTRAL_IS_EMBEDDED_SIM || SPECTRAL_RESTRICTED_MODE
    opts->n_threads = 1;
    opts->backend = BACKEND_CPU;
#endif
    return 1;
}

static CliCommonValidationResult cli_validate_common_synth_params(
    int timbre,
    float stretch,
    float pitch,
    int enforce_pitch_range,
    float pitch_min,
    float pitch_max)
{
    if (timbre < TIMBRE_SINE || timbre > TIMBRE_PWM) {
        return CLI_COMMON_VALIDATE_TIMBRE;
    }
    if (!spectral_is_finite_positive_f32(stretch)) {
        return CLI_COMMON_VALIDATE_STRETCH;
    }
    if (!spectral_is_finite_f32(pitch)) {
        return CLI_COMMON_VALIDATE_PITCH_FINITE;
    }
    if (enforce_pitch_range && (pitch < pitch_min || pitch > pitch_max)) {
        return CLI_COMMON_VALIDATE_PITCH_RANGE;
    }
    return CLI_COMMON_VALIDATE_OK;
}

static int cli_clamp_threads_to_hw(int requested_threads) {
    int clamped = requested_threads;
    int hw_threads = omp_get_max_threads();
    if (clamped < 1) clamped = 1;
    if (clamped > hw_threads) clamped = hw_threads;
    return clamped;
}

static int SPECTRAL_MAYBE_UNUSED
cli_threads_in_range(int n_threads, int min_threads, int max_threads) {
    return n_threads >= min_threads && n_threads <= max_threads;
}

static int SPECTRAL_MAYBE_UNUSED cli_is_valid_fft_size(int n_fft) {
    return (n_fft >= SPECTRAL_MIN_FFT_SIZE) && ((n_fft & (n_fft - 1)) == 0);
}

static int SPECTRAL_MAYBE_UNUSED cli_is_valid_hop_size(int hop, int n_fft) {
    return (hop > 0) && (hop <= n_fft);
}

void spectral_cli_init(SpectralCliOptions* opts) {
    if (!opts) return;
    
    opts->input_path = NULL;
    opts->wavetable_path = NULL;
    opts->timbre = TIMBRE_SINE;
    opts->stretch = 1.0f;
    opts->pitch = 0.0f;
    opts->n_fft = SPECTRAL_DEFAULT_N_FFT;
    opts->hop = SPECTRAL_DEFAULT_HOP;
    opts->db_thresh = SPECTRAL_DEFAULT_DB_THRESH;
    opts->n_threads = omp_get_max_threads();
    opts->backend = BACKEND_AUTO;
    opts->osc_quality = SPECTRAL_OSC_QUALITY_NAIVE;
    opts->osc_force_scalar = 0;   /* SIMD is the default CPU oscillator path */
    opts->enable_q15 = 0;         /* float is the default compute domain */
    opts->processing_mask = SPECTRAL_PROC_NONE;
    opts->enable_cache = 0;
    opts->start_sec = 0.0f;
    opts->end_sec = -1.0f;
    opts->use_wavetable = 0;
    opts->valid = 0;
    opts->error_message = NULL;
    
#if SPECTRAL_IS_EMBEDDED_SIM || SPECTRAL_RESTRICTED_MODE
    opts->n_threads = 1;  /* Force single-threaded for embedded modes */
    opts->backend = BACKEND_CPU;
#endif
}

int spectral_cli_parse(SpectralCliOptions* opts, int argc, char** argv) {
    int eff_argc = argc;
    char* eff_argv_buf[CLI_MAX_ARGV];
    char** eff_argv_heap = NULL;
    char** eff_argv = argv;
    int* skip = NULL;
    size_t alloc_bytes = 0;

    if (!opts) {
        return 0;
    }
    spectral_cli_init(opts);

    if (argc >= 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0 ||
                      strcmp(argv[1], "help") == 0)) {
        return cli_fail(opts, NULL);
    }

    if (argc < 2) {
        return cli_fail(opts, "No input file specified");
    }

    opts->input_path = argv[1];

    /* Parse named options and build effective positional argv */
    if (!spectral_size_mul((size_t)argc, sizeof(*skip), &alloc_bytes)) {
        return cli_fail(opts, "Argument list too large");
    }
    skip = (int*)calloc(1, alloc_bytes);
    if (!skip) {
        return cli_fail(opts, "Out of memory filtering argv");
    }

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--wavetable") == 0) {
            if (i + 1 >= argc) {
                cli_fail(opts, "Missing wavetable path after -w/--wavetable");
                goto fail;
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
                cli_fail(opts, "Missing mask value after -pm/--proc-mask");
                goto fail;
            }
            if (!spectral_process_mask_parse(argv[i + 1], &opts->processing_mask)) {
                cli_fail(opts, "Invalid processing mask. Use names (e.g. serra_smith_1990,johnston_1988), none/default/all, or numeric bitmask");
                goto fail;
            }
            skip[i] = 1;
            skip[i + 1] = 1;
            i++;
            continue;
        }
        if (strcmp(argv[i], "--cache") == 0) {
            opts->enable_cache = 1;
            skip[i] = 1;
            continue;
        }
        if (strcmp(argv[i], "--scalar") == 0) {
            opts->osc_force_scalar = 1;   /* force the scalar reference oscillator */
            skip[i] = 1;
            continue;
        }
        if (strcmp(argv[i], "--simd") == 0 || strcmp(argv[i], "-S") == 0) {
            opts->osc_force_scalar = 0;   /* explicit affirmative of the SIMD default */
            skip[i] = 1;
            continue;
        }
        if (strcmp(argv[i], "--q15") == 0) {
            opts->enable_q15 = 1;   /* opt into the Q15 fixed-point compute domain */
            skip[i] = 1;
            continue;
        }
        if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quality") == 0) {
            if (i + 1 >= argc) {
                cli_fail(opts, "Missing mode after -q/--quality");
                goto fail;
            }
            if (!spectral_osc_quality_parse(argv[i + 1], &opts->osc_quality)) {
                cli_fail(opts, "Invalid oscillator quality. Use naive|polyblep|additive|oversample (aliases blep/add/os) or 0..3");
                goto fail;
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
        if (!spectral_size_mul((size_t)argc, sizeof(*eff_argv_heap), &alloc_bytes)) {
            cli_fail(opts, "Argument list too large");
            goto fail;
        }
        eff_argv_heap = (char**)malloc(alloc_bytes);
        if (!eff_argv_heap) {
            cli_fail(opts, "Out of memory filtering argv");
            goto fail;
        }
        dst = eff_argv_heap;
    }

    eff_argc = 0;
    for (int i = 0; i < argc; i++) {
        if (skip[i]) continue;
        dst[eff_argc++] = argv[i];
    }

    free(skip);
    skip = NULL;
    eff_argv = dst;

    argc = eff_argc;
    argv = eff_argv;

    if (argc > (cli_max_positional_arg_index() + 1)) {
        cli_fail(opts, "Too many positional arguments");
        goto fail;
    }

    if (!cli_parse_mode_positional_args(opts, argc, argv)) goto fail;

    free(eff_argv_heap);  /* NULL-safe */
    opts->valid = 1;
    return spectral_cli_validate(opts);

fail:
    free(skip);
    free(eff_argv_heap);
    return 0;
}

int spectral_cli_validate(SpectralCliOptions* opts) {
    CliCommonValidationResult common_validation = CLI_COMMON_VALIDATE_OK;
    char actual[CLI_VALUE_MESSAGE_MAX] = {0};

    if (!opts) return 0;
    
#if !SPECTRAL_RESTRICTED_MODE
    /* Validate FFT size */
    if (!cli_is_valid_fft_size(opts->n_fft)) {
        snprintf(actual, sizeof(actual), "%d", opts->n_fft);
        return cli_fail_expected_actual(opts, "n_fft", "power-of-two integer >= 64", actual);
    }

    if (!cli_is_valid_hop_size(opts->hop, opts->n_fft)) {
        snprintf(actual, sizeof(actual), "%d (n_fft=%d)", opts->hop, opts->n_fft);
        return cli_fail_expected_actual(opts, "hop", "integer > 0 and <= n_fft", actual);
    }
#endif
    
    common_validation = cli_validate_common_synth_params(
        (int)opts->timbre, opts->stretch, opts->pitch,
        1, SPECTRAL_MIN_PITCH, SPECTRAL_MAX_PITCH);
    if (!cli_fail_common_validation(opts, common_validation)) {
        return 0;
    }

    /* Mirror spectral_config_validate(): every stretch consumer (synth dispatch,
     * seg-cache, segment parser, gpu_tile) rejects stretch > SPECTRAL_MAX_STRETCH,
     * so reject it at the CLI boundary too — otherwise an over-cap stretch passes
     * validation, wastes the full analysis pass, then fails deep in synthesis. */
    if (opts->stretch > SPECTRAL_MAX_STRETCH) {
        char expected[CLI_VALUE_MESSAGE_MAX] = {0};
        snprintf(expected, sizeof(expected), "finite float in (0, %.0f]",
                 (double)SPECTRAL_MAX_STRETCH);
        snprintf(actual, sizeof(actual), "%.6g", (double)opts->stretch);
        return cli_fail_expected_actual(opts, "stretch", expected, actual);
    }

    if (!spectral_is_finite_f32(opts->start_sec)) {
        snprintf(actual, sizeof(actual), "%.6g", (double)opts->start_sec);
        return cli_fail_expected_actual(opts, "start_sec", "finite float", actual);
    }
    if (!spectral_is_finite_f32(opts->end_sec)) {
        snprintf(actual, sizeof(actual), "%.6g", (double)opts->end_sec);
        return cli_fail_expected_actual(opts, "end_sec", "finite float", actual);
    }
    
    /* Validate backend enum */
    if ((int)opts->backend < BACKEND_AUTO || (int)opts->backend > BACKEND_EXPORT) {
        snprintf(actual, sizeof(actual), "%d", (int)opts->backend);
        return cli_fail_expected_actual(opts, "backend",
                                        "0..4 (auto/cpu/metal/cuda/export)", actual);
    }

#if SPECTRAL_RESTRICTED_MODE
    /* Restricted mode: force CPU backend */
    if (opts->backend != BACKEND_CPU && opts->backend != BACKEND_EXPORT) {
        snprintf(actual, sizeof(actual), "%s (%d)",
                 spectral_backend_name(opts->backend), (int)opts->backend);
        return cli_fail_expected_actual(opts, "restricted_mode backend",
                                        "CPU (1) or Export (4)", actual);
    }
#else
    /* Desktop: check wavetable + GPU compatibility */
    if (opts->use_wavetable) {
        if (opts->backend == BACKEND_METAL || opts->backend == BACKEND_CUDA) {
            snprintf(actual, sizeof(actual), "%s (%d)",
                     spectral_backend_name(opts->backend), (int)opts->backend);
            return cli_fail_expected_actual(opts, "wavetable backend",
                                            "AUTO (0) or CPU (1)", actual);
        }
        if (opts->backend == BACKEND_AUTO) {
            opts->backend = BACKEND_CPU;  /* Force CPU for wavetable */
        }
    }
#endif
    
    /* Validate thread count */
    opts->n_threads = cli_clamp_threads_to_hw(opts->n_threads);
    
    opts->valid = 1;
    return 1;
}

void spectral_cli_print_usage(void) {
#if SPECTRAL_RESTRICTED_MODE
    SPECTRAL_LOG_INFO("Usage: ./spectral_embedded output/segments.bin [options]");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("Restricted Mode Build (embedded-compatible)");
    SPECTRAL_LOG_INFO("  - CPU synthesis only (no Metal/CUDA)");
    SPECTRAL_LOG_INFO("  - Loads pre-analyzed segment files");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("Positional args:");
    SPECTRAL_LOG_INFO("  timbre stretch pitch");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("Wavetable option:");
    SPECTRAL_LOG_INFO("  -w <file>  Use wavetable file (.bin/.raw/.hex)");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("Processing mask option:");
    SPECTRAL_LOG_INFO("  -pm, --proc-mask <spec>  Optional processing chain mask");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("Defaults: timbre=0 stretch=1.0 pitch=0");
    SPECTRAL_LOG_INFO("Timbres: 0=sine 1=saw 2=square 3=tri 4=asin 5=para 6=quant 7=pwm");
    
#elif SPECTRAL_IS_EMBEDDED_SIM
    SPECTRAL_LOG_INFO("Usage: ./<simulation-binary> input.wav [options]");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("Embedded Target Simulation Build");
    SPECTRAL_LOG_INFO("  - Q15 fixed-point synthesis (same as ARM Cortex-M)");
    SPECTRAL_LOG_INFO("  - SINGLE-THREADED (models embedded constraints)");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("Positional args:");
    SPECTRAL_LOG_INFO("  timbre stretch pitch n_fft hop db_thresh start end");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("Wavetable option:");
    SPECTRAL_LOG_INFO("  -w <file>  Use wavetable file (.bin/.raw/.hex)");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("Processing mask option:");
    SPECTRAL_LOG_INFO("  -pm, --proc-mask <spec>  Optional processing chain mask");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("Defaults: timbre=0 stretch=1.0 pitch=0 n_fft=%d hop=%d thresh=%.1f",
                      SPECTRAL_DEFAULT_N_FFT, SPECTRAL_DEFAULT_HOP, SPECTRAL_DEFAULT_DB_THRESH);
    SPECTRAL_LOG_INFO("Timbres: 0=sine 1=saw 2=square 3=tri 4=asin 5=para 6=quant 7=pwm");
    
#else
    SPECTRAL_LOG_INFO("Usage: ./spectral input.wav [options]");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("Positional args:");
    SPECTRAL_LOG_INFO("  timbre stretch pitch n_fft hop db_thresh threads backend start end");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("Named options:");
    SPECTRAL_LOG_INFO("  -w <file>  Use wavetable file (.spwt/.bin/.raw/.hex)");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("  -pm, --proc-mask <spec>  Processing chain mask (comma-separated names or integer bitmask)");
    SPECTRAL_LOG_INFO("      Experimental (active only in a SPECTRAL_PRECISE_PHASE=1 build; fails loudly otherwise): adaptive_track_density");
    SPECTRAL_LOG_INFO("      Reserved (parse but fail loudly until implemented): serra_smith_1990,");
    SPECTRAL_LOG_INFO("             johnston_1988, reassigned, hybrid_render, event_bucket,");
    SPECTRAL_LOG_INFO("             higher_order_interp, qnoise_shaping");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("      Special: none, default, all");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("  --cache                   Prewarm backend and cache analyzed segments for this input");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("  -q, --quality <mode>      Anti-alias ('musical') oscillator quality (CPU float path)");
    SPECTRAL_LOG_INFO("      Modes: naive (default), polyblep, additive, oversample (aliases: blep, add, os; 0..3)");
    SPECTRAL_LOG_INFO("      Non-naive modes force the CPU backend; default is bit-identical to the point-sampled path");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("  -S, --simd                CPU oscillator uses SIMD (default; ~1.8x on saw/square, <=1 ULP vs scalar)");
    SPECTRAL_LOG_INFO("  --scalar                  Force the scalar reference oscillator (parity/debug)");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("  --q15                     Opt into the Q15 fixed-point compute domain (CPU path; forces CPU backend)");
    SPECTRAL_LOG_INFO("      Packed 8-wide SIMD Q15 on sine + saw/square/triangle/parabola (~1.3-1.6x over float-SIMD); scalar Q15 under --scalar");
    SPECTRAL_LOG_INFO("");
    SPECTRAL_LOG_INFO("Defaults: timbre=0 stretch=1.0 pitch=0 n_fft=%d hop=%d thresh=%.1f",
                      SPECTRAL_DEFAULT_N_FFT, SPECTRAL_DEFAULT_HOP, SPECTRAL_DEFAULT_DB_THRESH);
#if HAS_METAL
    SPECTRAL_LOG_INFO("Backend: 0=auto 1=cpu 2=metal 3=cuda 4=export");
#elif HAS_CUDA
    SPECTRAL_LOG_INFO("Backend: 0=auto 1=cpu 3=cuda 4=export");
#else
    SPECTRAL_LOG_INFO("Backend: 0=auto 1=cpu 4=export");
#endif
    SPECTRAL_LOG_INFO("Timbres: 0=sine 1=saw 2=square 3=tri 4=asin 5=para 6=quant 7=pwm");
#endif
}

 /* Runtime Config Validation (for programmatic use) */
SpectralError spectral_config_validate(const SpectralConfigParams* cfg,
                                       char* error_msg, size_t error_msg_size)
{
    CliCommonValidationResult common_validation = CLI_COMMON_VALIDATE_OK;

    if (!cfg) {
        return cli_config_fail(SPECTRAL_ERR_PARAM, error_msg, error_msg_size,
                               "NULL config");
    }
    
    if (cfg->sample_rate < SPECTRAL_MIN_SAMPLE_RATE ||
        cfg->sample_rate > SPECTRAL_MAX_SAMPLE_RATE) {
        return cli_config_fail(SPECTRAL_ERR_PARAM, error_msg, error_msg_size,
                               "sample_rate %d out of range", cfg->sample_rate);
    }

    common_validation = cli_validate_common_synth_params(
        cfg->timbre, cfg->stretch, cfg->pitch,
        1, SPECTRAL_MIN_PITCH, SPECTRAL_MAX_PITCH);
    if (common_validation != CLI_COMMON_VALIDATE_OK) {
        return cli_config_fail_common_validation(
            cfg, common_validation, error_msg, error_msg_size);
    }

    if (cfg->stretch > SPECTRAL_MAX_STRETCH) {
        return cli_config_fail(SPECTRAL_ERR_PARAM, error_msg, error_msg_size,
                               "stretch %.2f invalid", cfg->stretch);
    }
    
    if (cfg->buffer_size == 0) {
        return cli_config_fail(SPECTRAL_ERR_PARAM, error_msg, error_msg_size,
                               "buffer_size is zero");
    }
    
#if SPECTRAL_EMBEDDED
    if (cfg->buffer_size > SPECTRAL_EMBEDDED_MAX_BUFFER_SIZE) {
        return cli_config_fail(SPECTRAL_ERR_OVERFLOW, error_msg, error_msg_size,
                               "buffer_size %zu too large", cfg->buffer_size);
    }
#endif
    
#if !SPECTRAL_EMBEDDED
    if (!cli_threads_in_range(cfg->n_threads, 1, SPECTRAL_MAX_THREADS)) {
        return cli_config_fail(SPECTRAL_ERR_PARAM, error_msg, error_msg_size,
                               "n_threads %d out of range", cfg->n_threads);
    }
#endif

    return SPECTRAL_OK;
}
