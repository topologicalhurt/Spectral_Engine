/* spectral_synth_hybrid_wavetable.h - opt-in wavetable IFFT fast path
 * (RENDERER_ABSTRACTION_PLAN.md, Stage 3a-3)
 *
 * The wavetable analog of the sine hybrid (spectral_synth_hybrid.h): renders a dense set of
 * stationary wavetable partials via the IFFT path — each tracked partial expanded into its
 * wavetable's harmonics (spectral_wavetable_harmonics + spectral_wavetable_expand) and deposited
 * — instead of the per-sample table read. Like the sine hybrid this is an OPT-IN FAST PATH that
 * lives in its OWN TU compiled only into its parity-test target, NOT wired into the engine
 * dispatch until the F3 golden, so the default synth_cpu_wavetable path is byte-identical by
 * construction (the engine codegen does not change).
 *
 * Unlike the sine hybrid (which byte-matches the exact oscillator sum), the IFFT render here is
 * BAND-LIMITED while the default table read ALIASES above the table Nyquist — so it is NOT a byte
 * match for the default. Its correctness is pinned against the exact band-limited additive sum
 * (the ideal it targets, as in test_wavetable_render_parity). Whether the band-limited render is
 * an acceptable substitute for the aliasing default is the F3 maintainer judgment.
 */
#ifndef SPECTRAL_SYNTH_HYBRID_WAVETABLE_H
#define SPECTRAL_SYNTH_HYBRID_WAVETABLE_H

#include <stddef.h>

#include "spectral_common.h"
#include "spectral_config.h"
#include "spectral_wavetable.h"       /* SpectralWavetable */
#include "spectral_synth_hybrid.h"    /* SpectralHybridResult + the shared segment-eligibility gate */

#ifdef __cplusplus
extern "C" {
#endif

/* Attempt the wavetable IFFT fast path on a single resolved `table` (the caller resolves the
 * bank+timbre -> table; that keeps this TU free of the wavetable loader/fs dependencies).
 * Returns SPECTRAL_HYBRID_RENDERED with `out` (out_len samples) filled, or SPECTRAL_HYBRID_DECLINED
 * with `out` left strictly UNTOUCHED — in which case the caller renders via synth_cpu_wavetable().
 * Declines on: a NULL/invalid table; non-identity stretch/pitch; an empty/below-crossover set; a
 * table with no harmonics; a worst-case partial count that would exceed the frame buffer; or any
 * segment failing the (sine-shared) stationarity + in-domain-fundamental eligibility. */
SpectralHybridResult spectral_synth_hybrid_try_render_wavetable(
    SegmentArray sa, const SpectralWavetable* table, float* out, size_t out_len,
    float stretch, float pitch);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_SYNTH_HYBRID_WAVETABLE_H */
