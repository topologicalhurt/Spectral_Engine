/* spectral_common.c - Core Utilities */
#include "spectral_common.h"
#include <stdlib.h>

void* spectral_aligned_alloc(size_t size) {
    return aligned_alloc(CACHE_ALIGN, (size + CACHE_ALIGN - 1) & ~(CACHE_ALIGN - 1));
}
