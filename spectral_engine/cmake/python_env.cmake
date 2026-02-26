# Python environment bootstrap for build-time tooling scripts.

option(SPECTRAL_PYTHON_AUTO_INSTALL_DEPS
    "Create/use repo .venv and install tools/requirements.txt for build-time Python scripts."
    ON)

set(SPECTRAL_PYTHON_REQUIREMENTS_FILE
    "${SPECTRAL_REPO_ROOT}/tools/requirements.txt"
    CACHE FILEPATH
    "Python requirements for tools scripts.")

set(SPECTRAL_PYTHON_VENV_DIR
    "${SPECTRAL_REPO_ROOT}/.venv"
    CACHE PATH
    "Virtual environment directory for build-time Python scripts.")

find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(SPECTRAL_PYTHON_SYSTEM "${Python3_EXECUTABLE}")
set(SPECTRAL_PYTHON "${SPECTRAL_PYTHON_SYSTEM}")
set(SPECTRAL_PYTHON_ENV_STAMP "${CMAKE_CURRENT_BINARY_DIR}/python_env_ready.stamp")

if(SPECTRAL_PYTHON_AUTO_INSTALL_DEPS)
    set(SPECTRAL_PYTHON "${SPECTRAL_PYTHON_VENV_DIR}/bin/python3")
    set(SPECTRAL_PYTHON_ENV_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/ensure_python_env.cmake")
    configure_file(
        "${SPECTRAL_ENGINE_ROOT}/cmake/scripts/ensure_python_env.cmake.in"
        "${SPECTRAL_PYTHON_ENV_SCRIPT}"
        @ONLY)

    add_custom_command(
        OUTPUT "${SPECTRAL_PYTHON_ENV_STAMP}"
        COMMAND ${CMAKE_COMMAND} -P "${SPECTRAL_PYTHON_ENV_SCRIPT}"
        COMMAND ${CMAKE_COMMAND} -E touch "${SPECTRAL_PYTHON_ENV_STAMP}"
        DEPENDS
            "${SPECTRAL_PYTHON_ENV_SCRIPT}"
            "${SPECTRAL_PYTHON_REQUIREMENTS_FILE}"
        VERBATIM)

    add_custom_target(prepare_python_tools
        DEPENDS "${SPECTRAL_PYTHON_ENV_STAMP}")
    message(STATUS "Build-time Python env: .venv + requirements auto-install enabled")
else()
    add_custom_command(
        OUTPUT "${SPECTRAL_PYTHON_ENV_STAMP}"
        COMMAND ${CMAKE_COMMAND} -E touch "${SPECTRAL_PYTHON_ENV_STAMP}"
        VERBATIM)
    add_custom_target(prepare_python_tools
        DEPENDS "${SPECTRAL_PYTHON_ENV_STAMP}")
    message(STATUS "Build-time Python env: auto-install disabled; using system interpreter ${SPECTRAL_PYTHON_SYSTEM}")
endif()

message(STATUS "Build-time Python executable: ${SPECTRAL_PYTHON}")
