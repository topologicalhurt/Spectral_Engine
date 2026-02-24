/* spectral_proc_adaptive_track_density.h - Adaptive track-density stage */
#ifndef SPECTRAL_PROC_ADAPTIVE_TRACK_DENSITY_H
#define SPECTRAL_PROC_ADAPTIVE_TRACK_DENSITY_H

#include "spectral_error.h"
#include "spectral_processing_chain.h"

#ifdef __cplusplus
extern "C" {
#endif

SpectralError spectral_proc_adaptive_track_density_apply(
    SegmentArray* sa,
    int sample_rate,
    const SpectralProcessParams* params);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_PROC_ADAPTIVE_TRACK_DENSITY_H */
