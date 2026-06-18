# Metal MSL oscillator codegen — single source of truth is the C synthesis
# contract (spectral_osc_formulas.h / spectral_segment_math.h / spectral_oscillator.h).
# tools/spectral_tools/generators/metal_osc.py emits the committed header
# core/spectral_osc_metal_generated.h; the verify target fails the build if that
# header drifts from the current C formulas (replaces the old _Static_assert
# version locks).  python_env.cmake is already included by utilities.cmake.

set(SPECTRAL_METAL_OSC_SCRIPT "${SPECTRAL_REPO_ROOT}/tools/spectral_tools/generators/metal_osc.py")
set(SPECTRAL_METAL_OSC_OUTPUT "${SPECTRAL_ENGINE_ROOT}/drivers/metal/spectral_osc_metal_generated.h")
# The generated header is COMMITTED in-source. Its custom command's OUTPUT must be a
# build-tree stamp, not the committed header: CMake adds every custom-command OUTPUT to
# the `clean` list, so `cmake --build . --target clean` would delete a git-tracked file.
# The generator still rewrites the committed header in place (a side effect clean cannot
# see), gated on the same DEPENDS for incremental rebuilds.
set(SPECTRAL_METAL_OSC_STAMP "${CMAKE_CURRENT_BINARY_DIR}/generate_metal_osc.stamp")
set(SPECTRAL_METAL_OSC_RUNNER "${CMAKE_CURRENT_BINARY_DIR}/run_metal_osc.cmake")
set(SPECTRAL_METAL_OSC_INPUTS
    "${SPECTRAL_CORE_DIR}/spectral_osc_formulas.h"
    "${SPECTRAL_CORE_DIR}/spectral_segment_math.h"
    "${SPECTRAL_CORE_DIR}/spectral_oscillator.h"
    "${SPECTRAL_CORE_DIR}/spectral_common.h"
    "${SPECTRAL_CORE_DIR}/spectral_synth_internal.h")
configure_file(
    "${SPECTRAL_ENGINE_ROOT}/cmake/scripts/run_metal_osc.cmake.in"
    "${SPECTRAL_METAL_OSC_RUNNER}"
    @ONLY)

add_custom_command(
    OUTPUT "${SPECTRAL_METAL_OSC_STAMP}"
    COMMAND ${CMAKE_COMMAND}
            -DSPECTRAL_METAL_OSC_MODE=generate
            -P "${SPECTRAL_METAL_OSC_RUNNER}"
    COMMAND ${CMAKE_COMMAND} -E touch "${SPECTRAL_METAL_OSC_STAMP}"
    DEPENDS
        "${SPECTRAL_PYTHON_ENV_STAMP}"
        "${SPECTRAL_METAL_OSC_SCRIPT}"
        "${SPECTRAL_METAL_OSC_RUNNER}"
        ${SPECTRAL_METAL_OSC_INPUTS}
    COMMENT "Generating committed Metal MSL oscillator header (${SPECTRAL_METAL_OSC_OUTPUT})"
    VERBATIM)

add_custom_target(generate_metal_osc
    DEPENDS "${SPECTRAL_METAL_OSC_STAMP}")
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
