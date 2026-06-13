/* Shared CHECK assert for the standalone contract tests.
 *
 * Record-and-continue: a failed check marks the run failed but keeps
 * executing, so one run reports every violation (these are
 * characterization tests — the full printed table is the diagnostic).
 * The header owns the per-TU failure flag; a test's main() returns g_fail.
 */
#ifndef SPECTRAL_TESTS_CHECK_H
#define SPECTRAL_TESTS_CHECK_H

#include <stdio.h>

static int g_fail = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_fail = 1; } \
} while (0)

#endif /* SPECTRAL_TESTS_CHECK_H */
