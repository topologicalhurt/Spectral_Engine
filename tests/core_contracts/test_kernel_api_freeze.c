/* test_kernel_api_freeze.c - the public API surface guard (S3 / REVIEWER_HANDOFF).
 *
 * Pins the CURRENT 0.0.1 (pre-release) surface declared by spectral_kernel.h:
 *   - the version macros (0.0.1) + the AT_LEAST predicate;
 *   - the scene-model struct layouts + the ABI enum values;
 *   - the signatures of the curated public entry points — each is reassigned to a typed function
 *     pointer, so if a declared signature drifts this TU FAILS TO COMPILE.
 *
 * At 0.0.1 the surface is NOT frozen — it may still change toward 1.0; the guard makes every change
 * DELIBERATE (you bump the version + update this test when you change the API). It becomes a hard
 * freeze (breaking change ⇒ MAJOR bump) at 1.0. Fail-on-bug: change a signature/layout/value or the
 * version without updating here, and this fails to compile or fails at runtime.
 */
#include "spectral_kernel.h"
#include <stdio.h>
#include <string.h>

/* ---- frozen struct layouts (the public scene model + I/O) ---- */
_Static_assert(sizeof(Segment) == 64, "0.0.1 surface: Segment is 64 bytes");
_Static_assert(sizeof(SegmentCompact) == 32, "0.0.1 surface: SegmentCompact is 32 bytes");
_Static_assert(sizeof(SpectralAudioInfo) == sizeof(int) * 2 + sizeof(size_t),
               "0.0.1 surface: SpectralAudioInfo = {int sample_rate; int channels; size_t frames;}");
_Static_assert(SPECTRAL_OK == 0, "0.0.1 surface: SPECTRAL_OK == 0");
_Static_assert(SPECTRAL_RENDERER_COUNT == 3, "0.0.1 surface: 3 renderers (additive/wavetable/subtractive)");

/* ---- frozen enum VALUES (the ABI a consumer encodes against — not just struct sizes) ---- */
_Static_assert(SPECTRAL_ERR_PARAM == -1 && SPECTRAL_ERR_MEMORY == -2, "0.0.1 surface: core error codes");
_Static_assert(TIMBRE_SINE == 0 && TIMBRE_PWM == 7 && TIMBRE_COUNT == 8, "0.0.1 surface: timbre enum");
_Static_assert(BACKEND_AUTO == 0 && BACKEND_CPU == 1 && BACKEND_METAL == 2 &&
               BACKEND_CUDA == 3 && BACKEND_EXPORT == 4, "0.0.1 surface: SynthBackend enum");
_Static_assert(SPECTRAL_RENDERER_ADDITIVE == 0 && SPECTRAL_RENDERER_WAVETABLE == 1 &&
               SPECTRAL_RENDERER_SUBTRACTIVE == 2, "0.0.1 surface: renderer ids");
_Static_assert(SPECTRAL_SYNTH_API_VERSION_MAJOR == 0 && SPECTRAL_SYNTH_API_VERSION_PATCH == 1,
               "synth API at 0.0.1, aligned to the kernel 0.0.1 surface");

/* ---- frozen signatures: a typed pointer per curated public entry point ----
 * (the initializer fails to compile if the declared signature changes) */
static SegmentArray (*const fp_analyze)(const float*, size_t, int, int, int, float, double*, double*)
    = analyze_audio;
static SpectralError (*const fp_synth)(SegmentArray, float*, size_t, float, float, SpectralTimbre, int, double*)
    = synth_cpu;
static SpectralError (*const fp_synth_wt)(SegmentArray, float*, size_t, float, float,
                                          const SpectralWavetableBank*, SpectralTimbre, int, double*)
    = synth_cpu_wavetable;
static SpectralError (*const fp_dispatch)(SegmentArray, float*, size_t, float, float, SpectralTimbre,
                                          SynthBackend, const SpectralWavetableBank*, int, double*, SynthBackend*)
    = spectral_synth_dispatch;
static SpectralError (*const fp_read)(const char*, SpectralAudioInfo*, float**)
    = spectral_audio_read;
static SpectralError (*const fp_write)(const char*, const float*, size_t, int, int)
    = spectral_audio_write;
static const SpectralRenderer* (*const fp_renderer)(SpectralRendererId)
    = spectral_renderer_by_id;
static SpectralRendererId (*const fp_for_timbre)(SpectralTimbre)
    = spectral_renderer_for_timbre;
static const SpectralWavetable* (*const fp_wt_get)(const SpectralWavetableBank*, uint8_t)
    = spectral_wavetable_get;
static const char* (*const fp_errmsg)(SpectralErrorDomain, int)
    = spectral_error_message;

static int g_fails = 0;
#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", (m)); g_fails++; } } while (0)

int main(void)
{
    CHECK(SPECTRAL_KERNEL_VERSION_MAJOR == 0 && SPECTRAL_KERNEL_VERSION_MINOR == 0 &&
          SPECTRAL_KERNEL_VERSION_PATCH == 1, "kernel API at 0.0.1 (pre-release)");
    CHECK(SPECTRAL_KERNEL_VERSION_AT_LEAST(0, 0) && !SPECTRAL_KERNEL_VERSION_AT_LEAST(1, 0),
          "AT_LEAST predicate (>=0.0, <1.0)");
    CHECK(strcmp(SPECTRAL_KERNEL_VERSION_STRING, "0.0.1") == 0, "version string 0.0.1");

    /* the curated public entry points are present + correctly-signed (the fp_* initializers above
     * already enforced the signatures at compile time; this also confirms they linked) */
    CHECK(fp_analyze && fp_synth && fp_synth_wt && fp_dispatch && fp_read && fp_write &&
          fp_renderer && fp_for_timbre && fp_wt_get && fp_errmsg,
          "the frozen 1.0 public entry points are present + linked");

    if (g_fails == 0) fprintf(stderr, "PASS: kernel API surface (0.0.1)\n");
    return g_fails ? 1 : 0;
}
