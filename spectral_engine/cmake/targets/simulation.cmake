# Simulation targets.

add_executable(simulate EXCLUDE_FROM_ALL
    ${SPECTRAL_SOURCES_TARGET_SIMULATE})
set_target_properties(simulate PROPERTIES OUTPUT_NAME ${TARGET_SIMULATION})
spectral_apply_common_target(simulate)
target_link_libraries(simulate PRIVATE ${SPECTRAL_CPU_LINK_LIBS})
target_compile_definitions(simulate PRIVATE ${SPECTRAL_SIMULATION_DEFINES})

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
