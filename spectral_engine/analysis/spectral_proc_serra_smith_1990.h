/* spectral_proc_serra_smith_1990.h - Serra & Smith (1990) stage */
#ifndef SPECTRAL_PROC_SERRA_SMITH_1990_H
#define SPECTRAL_PROC_SERRA_SMITH_1990_H

#include "spectral_error.h"
#include "spectral_processing_chain.h"

#ifdef __cplusplus
extern "C" {
#endif

SpectralError spectral_proc_serra_smith_1990_apply(
    SegmentArray* sa,
    int sample_rate,
    const SpectralProcessParams* params);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_PROC_SERRA_SMITH_1990_H */
