/* oscillator_dispatch.c - profile-agnostic oscillator dispatch state.
 *
 * The SIMD segment implementations are build-selected per profile
 * (core/port/host/oscillator_simd.c for SIMDe, core/port/embedded/oscillator_simd.c
 * for CMSIS) behind oscillator_dispatch.h. This file holds only the
 * device-agnostic native-backend availability flag, shared by every profile. */
#include "oscillator_dispatch.h"

static int g_native_backend_available = 0;
void osc_set_native_available(int available) { g_native_backend_available = available; }
int osc_native_available(void) { return g_native_backend_available; }
