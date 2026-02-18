# Segment conversion utility target.

add_executable(convert_segments EXCLUDE_FROM_ALL
    "${SPECTRAL_SOURCE_CONVERT_SEGMENTS_ENTRY}"
    "${SPECTRAL_RUNTIME_DIR}/spectral_utils.c"
    "${SPECTRAL_CORE_DIR}/spectral_log.c")
set_target_properties(convert_segments PROPERTIES OUTPUT_NAME convert_segments)
target_include_directories(convert_segments PRIVATE ${SPECTRAL_INCLUDE_DIRS})
target_compile_options(convert_segments PRIVATE ${SPECTRAL_COMMON_COMPILE_OPTIONS} ${SPECTRAL_PGO_COMPILE_OPTIONS})
target_link_options(convert_segments PRIVATE ${SPECTRAL_COMMON_LINK_OPTIONS} ${SPECTRAL_PGO_LINK_OPTIONS})
target_link_libraries(convert_segments PRIVATE m)
