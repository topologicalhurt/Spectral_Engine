/* spectral_utils.h - Generic Utility Functions
 * 
 * This module provides reusable utilities for:
 *   - Formatted table output for diagnostics
 *   - Status message printing with severity levels
 *   - Human-readable formatting for bytes, durations, percentages
 *   - Progress bar display
 *   - Timbre name lookup
 * 
 * Used by: main.c, spectral_perf.c
 */
#ifndef SPECTRAL_UTILS_H
#define SPECTRAL_UTILS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * Table Printing
 */

typedef enum {
    ALIGN_LEFT,
    ALIGN_RIGHT,
    ALIGN_CENTER
} TableAlign;

typedef struct {
    const char* header;
    int width;
    TableAlign align;
} TableColumn;

typedef struct {
    TableColumn* columns;
    int num_columns;
    int border;          /* 0=none, 1=ascii */
    const char* title;   /* optional table title */
} TableConfig;

void table_print_header(const TableConfig* cfg);
void table_print_separator(const TableConfig* cfg);
void table_print_footer(const TableConfig* cfg);
void table_print_row_str(const TableConfig* cfg, const char** values);
void table_print_title(const TableConfig* cfg);
void table_print_row(const TableConfig* cfg, ...);

/* 
 * Status Messages
 */

typedef enum {
    STATUS_OK,
    STATUS_WARN,
    STATUS_ERROR,
    STATUS_INFO
} StatusLevel;

void status_print(StatusLevel level, const char* fmt, ...);

/* 
 * String Formatting
 */

const char* format_bytes(size_t bytes, char* buf, size_t buf_size);
const char* format_duration_ms(double ms, char* buf, size_t buf_size);
const char* format_percent(double pct, char* buf, size_t buf_size);

/* 
 * Numeric Output
 */

void print_padded_int(int value, int width, TableAlign align);
void print_padded_float(double value, int width, int precision, TableAlign align);
void print_padded_str(const char* str, int width, TableAlign align);

/* 
 * Progress Display
 */

void progress_bar_print(double fraction, int width, const char* label);

/* 
 * Timbre Names
 */

#include "spectral_config.h"

const char* timbre_name(SpectralTimbre timbre);

/*
 * Debug Printing
 * 
 * - spectral_debug(): Prints when SPECTRAL_DEBUG=1 and NDEBUG not defined
 * - spectral_warn(): Always prints to stderr (errors/warnings)
 * - spectral_debug_once(): Prints once per unique id
 * 
 * Build flags:
 *   -DSPECTRAL_DEBUG=1  Enable debug output
 *   -DNDEBUG            Disable all debug output (release builds)
 */

#ifndef SPECTRAL_DEBUG
#define SPECTRAL_DEBUG 0
#endif

/* Debug levels for filtering output */
typedef enum {
    DEBUG_LEVEL_ERROR = 0,
    DEBUG_LEVEL_WARN  = 1,
    DEBUG_LEVEL_INFO  = 2,
    DEBUG_LEVEL_TRACE = 3
} SpectralDebugLevel;

void spectral_debug(SpectralDebugLevel level, const char* fmt, ...);
void spectral_warn(const char* fmt, ...);
int  spectral_debug_once(int id, SpectralDebugLevel level, const char* fmt, ...);

/* Convenience macros with automatic level */
#if SPECTRAL_DEBUG && !defined(NDEBUG)
#define SPECTRAL_DBG(...)       spectral_debug(DEBUG_LEVEL_INFO, __VA_ARGS__)
#define SPECTRAL_DBG_TRACE(...) spectral_debug(DEBUG_LEVEL_TRACE, __VA_ARGS__)
#else
#define SPECTRAL_DBG(...)       ((void)0)
#define SPECTRAL_DBG_TRACE(...) ((void)0)
#endif

#define SPECTRAL_WARN(...)      spectral_warn(__VA_ARGS__)
#define SPECTRAL_WARN_ONCE(id, ...) spectral_debug_once((id), DEBUG_LEVEL_WARN, __VA_ARGS__)

/* 
 * Box Drawing Characters (ASCII set)
 */

typedef struct {
    char top_left;
    char top_right;
    char bottom_left;
    char bottom_right;
    char horizontal;
    char vertical;
    char cross;
    char t_down;
    char t_up;
    char t_left;
    char t_right;
} BoxChars;

extern const BoxChars BOX_ASCII;
extern const BoxChars BOX_UNICODE;

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_UTILS_H */
