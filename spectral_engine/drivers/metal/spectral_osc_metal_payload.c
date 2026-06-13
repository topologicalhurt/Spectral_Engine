/* spectral_osc_metal_payload.c - carries the generated MSL shader strings.
 *
 * The generated header defines oscillator_metal_source and
 * spectral_segment_math_metal_source (codegen from the C synthesis contract;
 * see metal-osc-codegen.cmake). This TU exists so ONLY the Metal driver
 * compiles the payload — when it lived in core/oscillator.c, every target
 * carried the shader strings as dead rodata and the codegen verify gate had
 * to run for all six builds instead of just the desktop.
 */
#include "spectral_metal.h"

#include "spectral_osc_metal_generated.h"
