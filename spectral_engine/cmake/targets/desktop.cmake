# Desktop build target.

set(DESKTOP_SOURCES ${SPECTRAL_SOURCES_TARGET_DESKTOP})

if(APPLE)
    list(APPEND DESKTOP_SOURCES ${SPECTRAL_SOURCES_SYNTH_METAL})
    set_source_files_properties(${SPECTRAL_SOURCES_SYNTH_METAL} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
elseif(SPECTRAL_USE_CUDA)
    list(APPEND DESKTOP_SOURCES ${SPECTRAL_SOURCES_SYNTH_CUDA})
endif()

add_executable(desktop ${DESKTOP_SOURCES})
set_target_properties(desktop PROPERTIES OUTPUT_NAME ${TARGET_DESKTOP})
spectral_apply_common_target(desktop)
target_link_libraries(desktop PRIVATE ${SPECTRAL_GPU_LINK_LIBS})

if(SPECTRAL_USE_CUDA)
    target_compile_definitions(desktop PRIVATE USE_CUDA)
    target_link_libraries(desktop PRIVATE CUDA::cudart)
    target_compile_options(desktop PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SPECTRAL_CUDA_COMPILE_OPTIONS}>)
endif()
