/* test_filter_envelope.c - subtractive filter envelope contract (RENDERER_ABSTRACTION_PLAN.md 3b).
 *
 * Pins the per-band spectral filter: piecewise-linear gain interpolation (incl. endpoint clamping,
 * empty/single-point degenerate cases) and the per-partial apply (subtractive filtering as a
 * spectral amplitude multiply). Fail-on-bug: break the interpolation or the bin->Hz mapping and a
 * check fails.
 */
#include "spectral_filter_envelope.h"
#include <math.h>
#include <stdio.h>

static int g_fails = 0;
#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", (m)); g_fails++; } } while (0)
static int approx(double a, double b) { return fabs(a - b) <= 1e-5; }

int main(void)
{
    /* lowpass-ish: flat 1.0 to 1000 Hz, ramp to 0.0 at 2000 Hz */
    SpectralFilterEnvelope env = {0};
    SpectralFilterEnvelope empty = {0};
    SpectralFilterEnvelope one = {0};
    SpectralIfftPartial parts[3];

    env.freq_hz[0] = 0.0f;    env.gain[0] = 1.0f;
    env.freq_hz[1] = 1000.0f; env.gain[1] = 1.0f;
    env.freq_hz[2] = 2000.0f; env.gain[2] = 0.0f;
    env.n_points = 3;

    /* gain interpolation + clamping */
    CHECK(approx(spectral_filter_envelope_gain(&env, 0.0f), 1.0), "gain at 0 Hz");
    CHECK(approx(spectral_filter_envelope_gain(&env, 500.0f), 1.0), "gain in flat region");
    CHECK(approx(spectral_filter_envelope_gain(&env, 1000.0f), 1.0), "gain at breakpoint 1000");
    CHECK(approx(spectral_filter_envelope_gain(&env, 1500.0f), 0.5), "gain interpolated at 1500");
    CHECK(approx(spectral_filter_envelope_gain(&env, 2000.0f), 0.0), "gain at breakpoint 2000");
    CHECK(approx(spectral_filter_envelope_gain(&env, 3000.0f), 0.0), "gain clamped above range");
    CHECK(approx(spectral_filter_envelope_gain(&env, -100.0f), 1.0), "gain clamped below range");

    /* degenerate envelopes */
    CHECK(approx(spectral_filter_envelope_gain(&empty, 440.0f), 1.0), "empty envelope = passthrough");
    CHECK(approx(spectral_filter_envelope_gain(NULL, 440.0f), 1.0), "NULL envelope = passthrough");
    one.freq_hz[0] = 440.0f; one.gain[0] = 0.7f; one.n_points = 1;
    CHECK(approx(spectral_filter_envelope_gain(&one, 100.0f), 0.7) &&
          approx(spectral_filter_envelope_gain(&one, 8000.0f), 0.7), "single-point = constant gain");

    /* apply: sr=4000, n_fft=512 -> 7.8125 Hz/bin; bin 128->1000Hz(g=1), 192->1500Hz(g=0.5),
       256->2000Hz(g=0) */
    parts[0].bin = 128.0f; parts[0].amp = 2.0f; parts[0].phase0 = 0.0f;
    parts[1].bin = 192.0f; parts[1].amp = 2.0f; parts[1].phase0 = 0.0f;
    parts[2].bin = 256.0f; parts[2].amp = 2.0f; parts[2].phase0 = 0.0f;
    spectral_filter_envelope_apply(&env, parts, 3, 4000.0f, 512u);
    CHECK(approx(parts[0].amp, 2.0), "apply: 1000 Hz partial unattenuated");
    CHECK(approx(parts[1].amp, 1.0), "apply: 1500 Hz partial halved");
    CHECK(approx(parts[2].amp, 0.0), "apply: 2000 Hz partial cut");

    if (g_fails == 0) fprintf(stderr, "PASS: filter envelope contract\n");
    return g_fails ? 1 : 0;
}
