/* spectral_osc_metal_payload.c - carries the generated MSL shader strings.
 *
 * The generated header defines oscillator_metal_source and
 * spectral_segment_math_metal_source (codegen from the C synthesis contract;
 * see metal-osc-codegen.cmake). This TU exists so ONLY the Metal driver
 * compiles the payload: compiled into core, the shader strings would be dead
 * rodata in every non-GPU target and the codegen verify gate would run for all
 * six builds instead of the desktop alone.
 */
#include "spectral_metal.h"

#include "spectral_osc_metal_generated.h"
