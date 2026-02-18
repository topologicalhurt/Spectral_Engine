# Daisy firmware orchestration targets.

spectral_daisy_collect_configure_args(OFF SPECTRAL_DAISY_CONFIGURE_ARGS)
spectral_daisy_collect_configure_args(ON SPECTRAL_DAISY_CONFIGURE_ARGS_EXAMPLE)
spectral_daisy_collect_build_args(SPECTRAL_DAISY_BUILD_ARGS)

add_custom_target(daisy_configure
    COMMAND ${CMAKE_COMMAND} ${SPECTRAL_DAISY_CONFIGURE_ARGS}
    WORKING_DIRECTORY "${SPECTRAL_REPO_ROOT}"
    VERBATIM)

add_custom_target(daisy_configure_example
    COMMAND ${CMAKE_COMMAND} ${SPECTRAL_DAISY_CONFIGURE_ARGS_EXAMPLE}
    WORKING_DIRECTORY "${SPECTRAL_REPO_ROOT}"
    VERBATIM)

add_custom_target(daisy
    COMMAND ${CMAKE_COMMAND} ${SPECTRAL_DAISY_BUILD_ARGS} --target daisy_api
    DEPENDS daisy_configure
    WORKING_DIRECTORY "${SPECTRAL_REPO_ROOT}"
    VERBATIM)

add_custom_target(daisy_flash
    COMMAND ${CMAKE_COMMAND} ${SPECTRAL_DAISY_BUILD_ARGS} --target daisy_api_flash
    DEPENDS daisy_configure
    WORKING_DIRECTORY "${SPECTRAL_REPO_ROOT}"
    VERBATIM)

add_custom_target(daisy_example
    COMMAND ${CMAKE_COMMAND} ${SPECTRAL_DAISY_BUILD_ARGS} --target daisy_example
    DEPENDS daisy_configure_example
    WORKING_DIRECTORY "${SPECTRAL_REPO_ROOT}"
    VERBATIM)

add_custom_target(daisy_example_flash
    COMMAND ${CMAKE_COMMAND} ${SPECTRAL_DAISY_BUILD_ARGS} --target daisy_example_flash
    DEPENDS daisy_configure_example
    WORKING_DIRECTORY "${SPECTRAL_REPO_ROOT}"
    VERBATIM)

add_custom_target(daisy_clean
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${SPECTRAL_DAISY_BUILD_DIR}" "${SPECTRAL_DAISY_BIN_DIR}"
    WORKING_DIRECTORY "${SPECTRAL_REPO_ROOT}"
    VERBATIM)
