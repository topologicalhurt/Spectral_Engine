/* spectral_proc_adaptive_track_density.h - Adaptive track-density stage */
#ifndef SPECTRAL_PROC_ADAPTIVE_TRACK_DENSITY_H
#define SPECTRAL_PROC_ADAPTIVE_TRACK_DENSITY_H

#include "spectral_processing_chain.h"

SpectralError spectral_proc_adaptive_track_density_apply(
    SegmentArray* sa,
    int sample_rate,
    const SpectralProcessParams* params);

#endif /* SPECTRAL_PROC_ADAPTIVE_TRACK_DENSITY_H */
