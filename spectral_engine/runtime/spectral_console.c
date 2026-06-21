/* spectral_console.c - Console formatting and status output */
#include "spectral_console.h"
#include "spectral_log.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static void spectral_console_vwrite(SpectralLogLevel level, int append_newline,
                                    const char* fmt, va_list args) {
    spectral_log_v(level, stdout, 0, append_newline, fmt, args);
}

static void spectral_console_write(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    spectral_console_vwrite(SPECTRAL_LOG_LEVEL_INFO, 0, fmt, args);
    va_end(args);
}

static void spectral_console_writeln(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    spectral_console_vwrite(SPECTRAL_LOG_LEVEL_INFO, 1, fmt, args);
    va_end(args);
}

void print_padded_str(const char* str, int width, TableAlign align) {
    int len = (int)strlen(str);
    int pad = width - len;
    if (pad < 0) pad = 0;

    switch (align) {
    case ALIGN_RIGHT:
        spectral_console_write("%*s%s", pad, "", str);
        break;
    case ALIGN_CENTER: {
        int left = pad / 2;
        int right = pad - left;
        spectral_console_write("%*s%s%*s", left, "", str, right, "");
        break;
    }
    case ALIGN_LEFT:
    default:
        spectral_console_write("%s%*s", str, pad, "");
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

    spectral_console_write("+");
    for (int i = 0; i < cfg->num_columns; i++) {
        for (int j = 0; j < cfg->columns[i].width; j++) {
            spectral_console_write("-");
        }
        spectral_console_write("+");
    }
    spectral_console_writeln("");
}

void table_print_title(const TableConfig* cfg) {
    if (!cfg->title) return;

    int total = table_total_width(cfg);

    if (cfg->border) {
        spectral_console_write("+");
        for (int i = 0; i < total - 2; i++) spectral_console_write("=");
        spectral_console_writeln("+");

        spectral_console_write("|");
        print_padded_str(cfg->title, total - 2, ALIGN_CENTER);
        spectral_console_writeln("|");
    } else {
        spectral_console_writeln("%s", cfg->title);
    }
}

void table_print_header(const TableConfig* cfg) {
    table_print_title(cfg);
    table_print_separator(cfg);

    if (cfg->border) spectral_console_write("|");
    for (int i = 0; i < cfg->num_columns; i++) {
        print_padded_str(cfg->columns[i].header, cfg->columns[i].width, ALIGN_CENTER);
        if (cfg->border) spectral_console_write("|");
        else if (i < cfg->num_columns - 1) spectral_console_write(" ");
    }
    spectral_console_writeln("");

    table_print_separator(cfg);
}

void table_print_footer(const TableConfig* cfg) {
    table_print_separator(cfg);
}

void table_print_row(const TableConfig* cfg, ...) {
    va_list args;
    va_start(args, cfg);

    if (cfg->border) spectral_console_write("|");
    for (int i = 0; i < cfg->num_columns; i++) {
        const char* val = va_arg(args, const char*);
        print_padded_str(val ? val : "", cfg->columns[i].width, cfg->columns[i].align);
        if (cfg->border) spectral_console_write("|");
        else if (i < cfg->num_columns - 1) spectral_console_write(" ");
    }
    spectral_console_writeln("");

    va_end(args);
}

void status_print(StatusLevel level, const char* fmt, ...) {
    SpectralLogLevel log_level = SPECTRAL_LOG_LEVEL_INFO;
    const char* prefix;
    va_list args;

    switch (level) {
    case STATUS_OK:
        prefix = "OK:      ";
        log_level = SPECTRAL_LOG_LEVEL_INFO;
        break;
    case STATUS_WARN:
        prefix = "WARNING: ";
        log_level = SPECTRAL_LOG_LEVEL_WARN;
        break;
    case STATUS_ERROR:
        prefix = "ERROR:   ";
        log_level = SPECTRAL_LOG_LEVEL_ERROR;
        break;
    case STATUS_INFO:
        prefix = "INFO:    ";
        log_level = SPECTRAL_LOG_LEVEL_INFO;
        break;
    default:
        prefix = "";
        log_level = SPECTRAL_LOG_LEVEL_INFO;
        break;
    }

    va_start(args, fmt);
    spectral_log(log_level, stdout, 0, 0, "%s", prefix);
    spectral_console_vwrite(log_level, 1, fmt, args);
    va_end(args);
}
