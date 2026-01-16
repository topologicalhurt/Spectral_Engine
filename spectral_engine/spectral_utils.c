/* spectral_utils.c - Generic Utility Functions
 * 
 * Implementation of reusable utilities for formatted output,
 * status messages, and human-readable value formatting.
 */
#include "spectral_utils.h"
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

/* 
 * Box Drawing Character Sets
 */

const BoxChars BOX_ASCII = {
    .top_left = '+', .top_right = '+',
    .bottom_left = '+', .bottom_right = '+',
    .horizontal = '-', .vertical = '|',
    .cross = '+', .t_down = '+', .t_up = '+',
    .t_left = '+', .t_right = '+'
};

const BoxChars BOX_UNICODE = {
    .top_left = '+', .top_right = '+',
    .bottom_left = '+', .bottom_right = '+',
    .horizontal = '=', .vertical = '|',
    .cross = '+', .t_down = '+', .t_up = '+',
    .t_left = '+', .t_right = '+'
};

/* 
 * String Formatting
 */

const char* format_bytes(size_t bytes, char* buf, size_t buf_size) {
    if (bytes >= 1024 * 1024 * 1024) {
        snprintf(buf, buf_size, "%.1f GB", bytes / (1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= 1024 * 1024) {
        snprintf(buf, buf_size, "%.1f MB", bytes / (1024.0 * 1024.0));
    } else if (bytes >= 1024) {
        snprintf(buf, buf_size, "%.1f KB", bytes / 1024.0);
    } else {
        snprintf(buf, buf_size, "%zu B", bytes);
    }
    return buf;
}

const char* format_duration_ms(double ms, char* buf, size_t buf_size) {
    if (ms >= 1000.0) {
        snprintf(buf, buf_size, "%.2f s", ms / 1000.0);
    } else if (ms >= 1.0) {
        snprintf(buf, buf_size, "%.1f ms", ms);
    } else {
        snprintf(buf, buf_size, "%.0f us", ms * 1000.0);
    }
    return buf;
}

const char* format_percent(double pct, char* buf, size_t buf_size) {
    snprintf(buf, buf_size, "%.1f%%", pct);
    return buf;
}

/* 
 * Padded Printing
 */

void print_padded_str(const char* str, int width, TableAlign align) {
    int len = (int)strlen(str);
    int pad = width - len;
    if (pad < 0) pad = 0;
    
    switch (align) {
        case ALIGN_RIGHT:
            printf("%*s%s", pad, "", str);
            break;
        case ALIGN_CENTER: {
            int left = pad / 2;
            int right = pad - left;
            printf("%*s%s%*s", left, "", str, right, "");
            break;
        }
        case ALIGN_LEFT:
        default:
            printf("%s%*s", str, pad, "");
            break;
    }
}

void print_padded_int(int value, int width, TableAlign align) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%d", value);
    print_padded_str(buf, width, align);
}

void print_padded_float(double value, int width, int precision, TableAlign align) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.*f", precision, value);
    print_padded_str(buf, width, align);
}

/* 
 * Table Printing
 */

static int table_total_width(const TableConfig* cfg) {
    int width = 0;
    for (int i = 0; i < cfg->num_columns; i++) {
        width += cfg->columns[i].width;
    }
    if (cfg->border) {
        width += cfg->num_columns + 1;
    } else {
        width += cfg->num_columns - 1;
    }
    return width;
}

void table_print_separator(const TableConfig* cfg) {
    if (cfg->border == 0) return;
    
    printf("+");
    for (int i = 0; i < cfg->num_columns; i++) {
        for (int j = 0; j < cfg->columns[i].width; j++) {
            printf("-");
        }
        printf("+");
    }
    printf("\n");
}

void table_print_title(const TableConfig* cfg) {
    if (!cfg->title) return;
    
    int total = table_total_width(cfg);
    
    if (cfg->border) {
        printf("+");
        for (int i = 0; i < total - 2; i++) printf("=");
        printf("+\n");
        
        printf("|");
        print_padded_str(cfg->title, total - 2, ALIGN_CENTER);
        printf("|\n");
    } else {
        printf("%s\n", cfg->title);
    }
}

void table_print_header(const TableConfig* cfg) {
    table_print_title(cfg);
    table_print_separator(cfg);
    
    if (cfg->border) printf("|");
    for (int i = 0; i < cfg->num_columns; i++) {
        print_padded_str(cfg->columns[i].header, cfg->columns[i].width, ALIGN_CENTER);
        if (cfg->border) printf("|");
        else if (i < cfg->num_columns - 1) printf(" ");
    }
    printf("\n");
    
    table_print_separator(cfg);
}

void table_print_footer(const TableConfig* cfg) {
    table_print_separator(cfg);
}

void table_print_row_str(const TableConfig* cfg, const char** values) {
    if (cfg->border) printf("|");
    for (int i = 0; i < cfg->num_columns; i++) {
        print_padded_str(values[i], cfg->columns[i].width, cfg->columns[i].align);
        if (cfg->border) printf("|");
        else if (i < cfg->num_columns - 1) printf(" ");
    }
    printf("\n");
}

void table_print_row(const TableConfig* cfg, ...) {
    va_list args;
    va_start(args, cfg);
    
    if (cfg->border) printf("|");
    for (int i = 0; i < cfg->num_columns; i++) {
        const char* val = va_arg(args, const char*);
        print_padded_str(val ? val : "", cfg->columns[i].width, cfg->columns[i].align);
        if (cfg->border) printf("|");
        else if (i < cfg->num_columns - 1) printf(" ");
    }
    printf("\n");
    
    va_end(args);
}

/* 
 * Status Messages
 */

void status_print(StatusLevel level, const char* fmt, ...) {
    const char* prefix;
    switch (level) {
        case STATUS_OK:    prefix = "OK:      "; break;
        case STATUS_WARN:  prefix = "WARNING: "; break;
        case STATUS_ERROR: prefix = "ERROR:   "; break;
        case STATUS_INFO:  prefix = "INFO:    "; break;
        default:           prefix = "";          break;
    }
    
    printf("%s", prefix);
    
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    
    printf("\n");
}

/* 
 * Progress Bar
 */

void progress_bar_print(double fraction, int width, const char* label) {
    if (fraction < 0.0) fraction = 0.0;
    if (fraction > 1.0) fraction = 1.0;
    
    int filled = (int)(fraction * width);
    
    printf("[");
    for (int i = 0; i < width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }
    printf("] %5.1f%%", fraction * 100.0);
    
    if (label) printf(" %s", label);
    printf("\r");
    fflush(stdout);
}

/* 
 * Timbre Names
 */

const char* timbre_name(SpectralTimbre timbre) {
    static const char* names[TIMBRE_COUNT] = {
        "sine", "saw", "square", "triangle",
        "asin", "parabola", "quantized", "pwm"
    };
    unsigned int idx = (unsigned int)timbre;
    return (idx < TIMBRE_COUNT) ? names[idx] : "unknown";
}

/*
 * Debug Printing Implementation
 */

/* Maximum number of unique IDs for once-only warnings */
#define SPECTRAL_DEBUG_ONCE_MAX 64

static uint8_t debug_once_flags[SPECTRAL_DEBUG_ONCE_MAX] = {0};

void spectral_debug(SpectralDebugLevel level, const char* fmt, ...) {
#if SPECTRAL_DEBUG && !defined(NDEBUG)
    /* Could add level filtering here if needed */
    (void)level;
    
    fprintf(stderr, "[DEBUG] ");
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
#else
    (void)level;
    (void)fmt;
#endif
}

void spectral_warn(const char* fmt, ...) {
    fprintf(stderr, "[WARNING] ");
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

int spectral_debug_once(int id, SpectralDebugLevel level, const char* fmt, ...) {
#ifndef NDEBUG
    if (id < 0 || id >= SPECTRAL_DEBUG_ONCE_MAX) return 0;
    if (debug_once_flags[id]) return 0;
    
    debug_once_flags[id] = 1;
    
    const char* level_str;
    switch (level) {
        case DEBUG_LEVEL_ERROR: level_str = "error"; break;
        case DEBUG_LEVEL_WARN:  level_str = "warning"; break;
        case DEBUG_LEVEL_INFO:  level_str = "info"; break;
        case DEBUG_LEVEL_TRACE: level_str = "trace"; break;
        default:                level_str = "debug"; break;
    }
    
    fprintf(stderr, "[DEBUG ] %s: ", level_str);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    return 1;
#else
    (void)id;
    (void)level;
    (void)fmt;
    return 0;
#endif
}
