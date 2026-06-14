/* spectral_processing_chain.h - Optional segment processing mask pipeline */
#ifndef SPECTRAL_PROCESSING_CHAIN_H
#define SPECTRAL_PROCESSING_CHAIN_H

#include "spectral_common.h"
#include "spectral_error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t SpectralProcessMask;

/* Contributor workflow (baseline vs optional layers)
 *
 * 1) Baseline path is always-on and correctness-first:
 *    analysis/tracking/synthesis in existing engine code remains authoritative.
 *
 * 2) Processing-chain stages are optional post-analysis transforms:
 *    they modify SegmentArray after baseline analysis/load and before synth/export.
 *
 * 3) New methods should be additive:
 *    add a mask bit + stage module; do not move baseline semantics into optional stages.
 */

/* Technique references (centralized)
 *
 * Serra & Smith (1990):
 *   "Spectral Modeling Synthesis: A Sound Analysis/Synthesis System Based on a
 *   Deterministic Plus Stochastic Decomposition"
 *   Link: https://ccrma.stanford.edu/~jos/sasp/Spectral_Modeling_Synthesis.html
 *   Why: Splits tonal structure from residual/noise, reducing sinusoidal burden.
 *
 * Johnston (1988):
 *   "Transform Coding of Audio Signals Using Perceptual Noise Criteria"
 *   Link: https://ieeexplore.ieee.org/document/1164959
 *   Why: Uses masking to remove inaudible components while preserving perception.
 *
 * Adaptive track-density policy (informed by McAulay & Quatieri 1986):
 *   "Speech Analysis/Synthesis Based on a Sinusoidal Representation"
 *   Link: https://ieeexplore.ieee.org/document/1164600
 *   Why: Uses trajectory salience ideas for optional segment-budget control.
 *   Note: baseline harmonic-bin/tracking logic elsewhere remains always-on.
 */
enum {
    SPECTRAL_PROC_NONE                    = 0u,
    SPECTRAL_PROC_SERRA_SMITH_1990        = 1u << 0,
    SPECTRAL_PROC_JOHNSTON_1988           = 1u << 1,
    SPECTRAL_PROC_ADAPTIVE_TRACK_DENSITY  = 1u << 2,
    SPECTRAL_PROC_REASSIGNED_TRACKING     = 1u << 3,
    SPECTRAL_PROC_HYBRID_RENDER_SWITCH    = 1u << 4,
    SPECTRAL_PROC_EVENT_BUCKET_SCHED      = 1u << 5,
    SPECTRAL_PROC_HIGHER_ORDER_INTERP     = 1u << 6,
    SPECTRAL_PROC_QNOISE_SHAPING          = 1u << 7,

    SPECTRAL_PROC_ALL_KNOWN =
        SPECTRAL_PROC_SERRA_SMITH_1990 |
        SPECTRAL_PROC_JOHNSTON_1988 |
        SPECTRAL_PROC_ADAPTIVE_TRACK_DENSITY |
        SPECTRAL_PROC_REASSIGNED_TRACKING |
        SPECTRAL_PROC_HYBRID_RENDER_SWITCH |
        SPECTRAL_PROC_EVENT_BUCKET_SCHED |
        SPECTRAL_PROC_HIGHER_ORDER_INTERP |
        SPECTRAL_PROC_QNOISE_SHAPING
};

typedef struct {
    SpectralProcessMask requested;
    SpectralProcessMask applied;
    SpectralProcessMask pending;
} SpectralProcessReport;

typedef struct {
    uint32_t adaptive_max_segments;
    float adaptive_keep_ratio;
    int hop;    /* STFT hop size (samples); 0 if unknown */
    int n_fft;  /* STFT length (samples); 0 if unknown */
} SpectralProcessParams;

void spectral_process_params_default(SpectralProcessParams* params);

int spectral_process_mask_parse(const char* spec, SpectralProcessMask* out_mask);
void spectral_process_mask_to_string(SpectralProcessMask mask, char* out, size_t out_size);

SpectralError spectral_process_chain_apply(
    SegmentArray* sa,
    int sample_rate,
    int hop,
    int n_fft,
    SpectralProcessMask mask,
    SpectralProcessReport* report);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_PROCESSING_CHAIN_H */
