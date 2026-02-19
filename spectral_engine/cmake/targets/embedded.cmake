# Embedded host targets.

add_executable(embedded_arm EXCLUDE_FROM_ALL
    ${SPECTRAL_SOURCES_TARGET_EMBEDDED_ARM})
set_target_properties(embedded_arm PROPERTIES OUTPUT_NAME ${TARGET_EMBEDDED_ARM})
spectral_apply_embedded_target(embedded_arm)
target_compile_definitions(embedded_arm PRIVATE ${SPECTRAL_SIMULATION_DEFINES})

add_executable(embedded_arm_float EXCLUDE_FROM_ALL
    ${SPECTRAL_SOURCES_TARGET_EMBEDDED_ARM})
set_target_properties(embedded_arm_float PROPERTIES OUTPUT_NAME ${TARGET_EMBEDDED_ARM_FLOAT})
spectral_apply_embedded_target(embedded_arm_float)
target_compile_definitions(embedded_arm_float PRIVATE ${SPECTRAL_SIMULATION_DEFINES} ${SPECTRAL_EMBEDDED_FLOAT_DEFINES})

add_executable(embedded_arm_restricted EXCLUDE_FROM_ALL
    ${SPECTRAL_SOURCES_TARGET_EMBEDDED_ARM_RESTRICTED})
set_target_properties(embedded_arm_restricted PROPERTIES OUTPUT_NAME ${TARGET_EMBEDDED_ARM_RES})
spectral_apply_embedded_target(embedded_arm_restricted)
target_compile_options(embedded_arm_restricted PRIVATE -fno-exceptions -fno-threadsafe-statics)
target_compile_definitions(embedded_arm_restricted PRIVATE ${SPECTRAL_SIMULATION_DEFINES} ${SPECTRAL_EMBEDDED_RESTRICTED_DEFINES})
