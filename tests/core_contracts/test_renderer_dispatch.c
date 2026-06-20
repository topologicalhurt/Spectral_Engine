/* test_renderer_dispatch.c - renderer-abstraction contract (RENDERER_ABSTRACTION_PLAN.md, Stage 1).
 *
 * Pins the renderer registry: each id resolves to the right descriptor + name, the per-renderer
 * capability flags are a contract (all spectral-native; additive = single lobe, wavetable +
 * subtractive = harmonic stack; only subtractive needs a filter), and each timbre classifies to
 * the right renderer. Fail-on-bug: flip a cap or a timbre mapping and a check fails.
 */
#include "spectral_renderer.h"
#include <stdio.h>
#include <string.h>

static int g_fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", (msg)); g_fails++; } } while (0)

int main(void)
{
    const SpectralRenderer* add;
    const SpectralRenderer* wt;
    const SpectralRenderer* sub;

    /* registry: ids resolve, out-of-range is NULL */
    CHECK(spectral_renderer_by_id(SPECTRAL_RENDERER_ADDITIVE) != NULL, "additive id resolves");
    CHECK(spectral_renderer_by_id(SPECTRAL_RENDERER_COUNT) == NULL, "out-of-range id is NULL");

    add = spectral_renderer_by_id(SPECTRAL_RENDERER_ADDITIVE);
    wt  = spectral_renderer_by_id(SPECTRAL_RENDERER_WAVETABLE);
    sub = spectral_renderer_by_id(SPECTRAL_RENDERER_SUBTRACTIVE);

    /* names */
    CHECK(strcmp(add->name, "additive") == 0,    "additive name");
    CHECK(strcmp(wt->name,  "wavetable") == 0,    "wavetable name");
    CHECK(strcmp(sub->name, "subtractive") == 0,  "subtractive name");

    /* caps contract */
    CHECK(add->caps.spectral_native && wt->caps.spectral_native && sub->caps.spectral_native,
          "all three are spectral-native");
    CHECK(!add->caps.deposits_harmonic_stack, "additive deposits a single lobe");
    CHECK(wt->caps.deposits_harmonic_stack && sub->caps.deposits_harmonic_stack,
          "wavetable + subtractive deposit harmonic stacks");
    CHECK(sub->caps.needs_filter, "subtractive needs a filter");
    CHECK(!add->caps.needs_filter && !wt->caps.needs_filter, "additive + wavetable need no filter");

    /* timbre -> renderer classification */
    CHECK(spectral_renderer_for_timbre(TIMBRE_SINE)     == SPECTRAL_RENDERER_ADDITIVE,    "sine -> additive");
    CHECK(spectral_renderer_for_timbre(TIMBRE_SAW)      == SPECTRAL_RENDERER_SUBTRACTIVE, "saw -> subtractive");
    CHECK(spectral_renderer_for_timbre(TIMBRE_SQUARE)   == SPECTRAL_RENDERER_SUBTRACTIVE, "square -> subtractive");
    CHECK(spectral_renderer_for_timbre(TIMBRE_TRIANGLE) == SPECTRAL_RENDERER_SUBTRACTIVE, "triangle -> subtractive");
    CHECK(spectral_renderer_for_timbre(TIMBRE_PWM)      == SPECTRAL_RENDERER_SUBTRACTIVE, "pwm -> subtractive");

    /* name helper */
    CHECK(strcmp(spectral_renderer_name(SPECTRAL_RENDERER_SUBTRACTIVE), "subtractive") == 0, "name helper");
    CHECK(strcmp(spectral_renderer_name(SPECTRAL_RENDERER_COUNT), "unknown") == 0, "name helper out-of-range");

    if (g_fails == 0) {
        fprintf(stderr, "PASS: renderer dispatch contract\n");
    }
    return g_fails ? 1 : 0;
}
