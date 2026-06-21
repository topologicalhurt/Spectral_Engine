/* spectral_segment_convert.h - runtime float Segment -> Q15 segment policy.
 *
 * THE canonical conversion for any embedded-bound synthesis path that takes
 * desktop float segments at runtime. Unlike the offline converter
 * (cmd/convert_segments.c, an at-rest transcription with no playback
 * parameters), this policy folds the runtime request into the encoding:
 *   - start/length  : scaled by stretch (length clamped to the 16-bit field)
 *   - freq_q88      : omega*pitch*inv_stretch encoded via OMEGA_TO_Q88
 *   - amp_q15/da_q15: scaled by amp_scale (Q15 headroom normalization)
 *   - df_q15        : forced 0 — the ARM32 hot path does not consume chirp
 *                     and spectral_arm32_load() rejects df_q15 != 0.
 * Units are per-sample throughout (the kernel applies amp + da*offset per
 * sample); both converters' encodings ride the same named boundary macros.
 */
#ifndef SPECTRAL_SEGMENT_CONVERT_H
#define SPECTRAL_SEGMENT_CONVERT_H

#include <stddef.h>

#include "spectral_common.h"
#include "spectral_q.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 on a loadable segment, 0 if the segment must be dropped
 * (invalid fields, or start lands outside [0, out_len) after stretch). */
int spectral_segment_to_q15_runtime(const Segment* src, SpectralSegmentQ15* dst,
                                    float amp_scale, const SynthParams* params,
                                    size_t out_len);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_SEGMENT_CONVERT_H */
