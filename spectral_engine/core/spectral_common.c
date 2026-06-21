/* spectral_common.c - cache-aligned host allocation */
#include "spectral_common.h"
#include <stdlib.h>

void* spectral_aligned_alloc(size_t size) {
    if (size == 0) return NULL;
    if (size > (SIZE_MAX - (SPECTRAL_CACHE_ALIGN - 1))) return NULL;
    size_t aligned_size = (size + SPECTRAL_CACHE_ALIGN - 1) & ~(SPECTRAL_CACHE_ALIGN - 1);
    return aligned_alloc(SPECTRAL_CACHE_ALIGN, aligned_size);
}
