# Metal MSL oscillator codegen — single source of truth is the C synthesis
# contract (spectral_osc_formulas.h / spectral_segment_math.h / oscillator.h).
# tools/spectral_tools/generators/metal_osc.py emits the committed header
# core/spectral_osc_metal_generated.h; the verify target fails the build if that
# header drifts from the current C formulas (replaces the old _Static_assert
# version locks).  python_env.cmake is already included by utilities.cmake.

set(SPECTRAL_METAL_OSC_SCRIPT "${SPECTRAL_REPO_ROOT}/tools/spectral_tools/generators/metal_osc.py")
set(SPECTRAL_METAL_OSC_OUTPUT "${SPECTRAL_ENGINE_ROOT}/drivers/metal/spectral_osc_metal_generated.h")
set(SPECTRAL_METAL_OSC_RUNNER "${CMAKE_CURRENT_BINARY_DIR}/run_metal_osc.cmake")
set(SPECTRAL_METAL_OSC_INPUTS
    "${SPECTRAL_CORE_DIR}/spectral_osc_formulas.h"
    "${SPECTRAL_CORE_DIR}/spectral_segment_math.h"
    "${SPECTRAL_CORE_DIR}/oscillator.h"
    "${SPECTRAL_CORE_DIR}/spectral_common.h"
    "${SPECTRAL_CORE_DIR}/spectral_synth_internal.h")
configure_file(
    "${SPECTRAL_ENGINE_ROOT}/cmake/scripts/run_metal_osc.cmake.in"
    "${SPECTRAL_METAL_OSC_RUNNER}"
    @ONLY)

add_custom_command(
    OUTPUT "${SPECTRAL_METAL_OSC_OUTPUT}"
    COMMAND ${CMAKE_COMMAND}
            -DSPECTRAL_METAL_OSC_MODE=generate
            -P "${SPECTRAL_METAL_OSC_RUNNER}"
    DEPENDS
        "${SPECTRAL_PYTHON_ENV_STAMP}"
        "${SPECTRAL_METAL_OSC_SCRIPT}"
        "${SPECTRAL_METAL_OSC_RUNNER}"
        ${SPECTRAL_METAL_OSC_INPUTS}
    VERBATIM)

add_custom_target(generate_metal_osc
    DEPENDS "${SPECTRAL_METAL_OSC_OUTPUT}")
add_dependencies(generate_metal_osc prepare_python_tools)

set_source_files_properties(
    "${SPECTRAL_METAL_OSC_OUTPUT}"
    PROPERTIES GENERATED TRUE)

add_custom_target(verify_metal_osc
    COMMAND ${CMAKE_COMMAND}
            -DSPECTRAL_METAL_OSC_MODE=verify
            -P "${SPECTRAL_METAL_OSC_RUNNER}"
    DEPENDS
        prepare_python_tools
        generate_metal_osc
        "${SPECTRAL_METAL_OSC_SCRIPT}"
        "${SPECTRAL_METAL_OSC_RUNNER}"
        ${SPECTRAL_METAL_OSC_INPUTS}
    VERBATIM)

# Only the Metal driver's payload TU includes the generated header, so only
# the desktop build (the one target that links the driver) gates on verify.
if(TARGET desktop)
    add_dependencies(desktop verify_metal_osc)
endif()
