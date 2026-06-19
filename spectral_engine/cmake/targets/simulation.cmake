# Simulation targets.

add_executable(simulate EXCLUDE_FROM_ALL
    ${SPECTRAL_SOURCES_TARGET_SIMULATE})
set_target_properties(simulate PROPERTIES OUTPUT_NAME ${TARGET_SIMULATION})
spectral_apply_common_target(simulate)
target_link_libraries(simulate PRIVATE ${SPECTRAL_CPU_LINK_LIBS})
# Force the real M7 codepath (coupled-recurrence oscillator + dual-MAC) the same way
# arm_core_test does, so a host roundtrip (audio file -> embedded synth -> WAV) renders the
# EXACT path the Daisy/M7 binary runs -- not the generic LUT-gather fallback the host would
# otherwise auto-select. Fixed-point is deterministic, so this audio is bit-identical to the
# QEMU/M7 render; only the LUT-vs-coupled oscillator choice differs from the unforced sim.
target_compile_definitions(simulate PRIVATE ${SPECTRAL_SIMULATION_DEFINES}
    SPECTRAL_ARM_M7=1 SPECTRAL_HAS_DUAL_MAC=1)

add_executable(simulate_daisy EXCLUDE_FROM_ALL
    ${SPECTRAL_SOURCES_TARGET_SIMULATE_DAISY})
set_target_properties(simulate_daisy PROPERTIES OUTPUT_NAME ${TARGET_SIMULATION_DAISY})
spectral_apply_common_target(simulate_daisy)
target_link_libraries(simulate_daisy PRIVATE ${SPECTRAL_CPU_LINK_LIBS})
target_compile_definitions(simulate_daisy PRIVATE ${SPECTRAL_SIMULATION_DAISY_DEFINES})

if(SPECTRAL_SIMULATION_BOARD STREQUAL "daisy")
    add_custom_target(simulate_board DEPENDS simulate_daisy)
else()
    add_custom_target(simulate_board DEPENDS simulate)
endif()
