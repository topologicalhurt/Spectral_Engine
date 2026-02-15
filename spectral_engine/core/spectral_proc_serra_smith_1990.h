/* spectral_proc_serra_smith_1990.h - Serra & Smith (1990) stage */
#ifndef SPECTRAL_PROC_SERRA_SMITH_1990_H
#define SPECTRAL_PROC_SERRA_SMITH_1990_H

#include "spectral_processing_chain.h"

SpectralError spectral_proc_serra_smith_1990_apply(
    SegmentArray* sa,
    int sample_rate,
    const SpectralProcessParams* params);

#endif /* SPECTRAL_PROC_SERRA_SMITH_1990_H */
