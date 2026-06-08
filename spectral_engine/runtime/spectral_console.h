/* spectral_console.h - Console formatting and status output
 *
 * Thread safety: these functions write to stdout without synchronization.
 * Call only from the main thread or protect with an external lock when used
 * alongside OMP parallel regions. */
#ifndef SPECTRAL_CONSOLE_H
#define SPECTRAL_CONSOLE_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Table formatting
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
    int border;        /* 0=none, 1=ascii */
    const char* title; /* optional table title */
} TableConfig;

void table_print_header(const TableConfig* cfg);
void table_print_separator(const TableConfig* cfg);
void table_print_footer(const TableConfig* cfg);
void table_print_title(const TableConfig* cfg);
void table_print_row(const TableConfig* cfg, ...);

/*
 * Status messages
 */
typedef enum {
    STATUS_OK,
    STATUS_WARN,
    STATUS_ERROR,
    STATUS_INFO
} StatusLevel;

void status_print(StatusLevel level, const char* fmt, ...);

/*
 * Numeric output helpers for tables/console
 */
void print_padded_int(int value, int width, TableAlign align);
void print_padded_float(double value, int width, int precision, TableAlign align);
void print_padded_str(const char* str, int width, TableAlign align);

/*
 * Box drawing character sets
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

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_CONSOLE_H */
