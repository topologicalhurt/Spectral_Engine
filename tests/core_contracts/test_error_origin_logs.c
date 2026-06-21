/* test_error_origin_logs.c - kernel-hardening V.4 (logging contract, AI_CANON #29).
 *
 * analyze_audio's defensive early returns are LIBRARY-BOUNDARY error origins the CLI can
 * never reach (the CLI pre-validates n_fft/sr/hop and rejects with its own message before
 * calling in). They are reachable only by a direct library caller, so they are pinned here
 * against the REAL analyze_audio entry point:
 *   - a NULL argument must log "analyze_audio: null argument"
 *   - an invalid analysis shape must log "analyze_audio: invalid analysis shape"
 *
 * Both go through spectral_log_error_codef -> stdout, so we redirect stdout to a capture
 * file, drive the two origins, and assert each cause string is present.
 *
 * Fail-on-bug: revert either origin in spectral_analysis.c to a silent return (delete the
 * spectral_log_error_codef call) and this test fails.
 */
#include "spectral_analysis.h"   /* analyze_audio */
#include <stdio.h>
#include <string.h>

#define CAP_PATH "error_origin_logs_capture.txt"

int main(void)
{
    double t_fft = 0.0;
    double t_track = 0.0;
    static float audio[8192];   /* zero-filled valid buffer for the bad-shape call */
    char captured[8192];
    size_t n = 0;
    FILE* f = NULL;
    int ok = 1;

    /* Redirect stdout (where the structured error log lands) to a capture file. The process
     * exits right after asserting, so there is no need to restore stdout; diagnostics below
     * go to stderr so they stay out of the captured buffer. */
    if (!freopen(CAP_PATH, "w", stdout)) {
        fprintf(stderr, "FAIL: could not redirect stdout to %s\n", CAP_PATH);
        return 1;
    }

    /* Origin 1: null argument (caught before any shape work). */
    (void)analyze_audio(NULL, 8192, 48000, 4096, 128, -90.0f, &t_fft, &t_track);

    /* Origin 2: invalid shape — sr below SPECTRAL_MIN_SAMPLE_RATE; the buffer/timers are
     * valid so we pass the null guard and land in the shape-init failure path. */
    (void)analyze_audio(audio, 8192, 5, 4096, 128, -90.0f, &t_fft, &t_track);

    fflush(stdout);

    f = fopen(CAP_PATH, "r");
    if (!f) {
        fprintf(stderr, "FAIL: could not reopen capture file %s\n", CAP_PATH);
        return 1;
    }
    n = fread(captured, 1, sizeof(captured) - 1u, f);
    fclose(f);
    remove(CAP_PATH);
    captured[n] = '\0';

    if (!strstr(captured, "analyze_audio: null argument")) {
        fprintf(stderr, "FAIL: missing the null-argument error-origin log\n");
        ok = 0;
    }
    if (!strstr(captured, "analyze_audio: invalid analysis shape")) {
        fprintf(stderr, "FAIL: missing the invalid-shape error-origin log\n");
        ok = 0;
    }

    if (ok) {
        fprintf(stderr, "PASS: both analyze_audio error-origin logs present\n");
    }
    return ok ? 0 : 1;
}
