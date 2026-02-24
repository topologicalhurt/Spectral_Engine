/* spectral_proc_johnston_1988.h - Johnston (1988) masking stage */
#ifndef SPECTRAL_PROC_JOHNSTON_1988_H
#define SPECTRAL_PROC_JOHNSTON_1988_H

#include "spectral_error.h"
#include "spectral_processing_chain.h"

#ifdef __cplusplus
extern "C" {
#endif

SpectralError spectral_proc_johnston_1988_apply(
    SegmentArray* sa,
    int sample_rate,
    const SpectralProcessParams* params);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_PROC_JOHNSTON_1988_H */
