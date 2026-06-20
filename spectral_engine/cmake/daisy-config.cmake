# Daisy Seed helper functions. Build options and cache variables live in
# spectral_engine/cmake/options.cmake.

if(DEFINED SPECTRAL_DAISY_CONFIG_INCLUDED)
    return()
endif()
set(SPECTRAL_DAISY_CONFIG_INCLUDED TRUE)

if(NOT DEFINED SPECTRAL_DAISY_FORWARD_CACHE_VARS)
    message(FATAL_ERROR
        "SPECTRAL_DAISY_FORWARD_CACHE_VARS is not defined. "
        "Include spectral_engine/cmake/options.cmake before daisy-config.cmake.")
endif()

function(spectral_daisy_parse_bool in_value out_var)
    string(TOUPPER "${in_value}" _value_upper)
    if(_value_upper STREQUAL "1" OR
       _value_upper STREQUAL "ON" OR
       _value_upper STREQUAL "TRUE" OR
       _value_upper STREQUAL "YES")
        set(${out_var} ON PARENT_SCOPE)
    else()
        set(${out_var} OFF PARENT_SCOPE)
    endif()
endfunction()

function(spectral_daisy_collect_configure_cache_args out_var)
    set(_args)
    foreach(_cache_var IN LISTS SPECTRAL_DAISY_FORWARD_CACHE_VARS)
        if(DEFINED ${_cache_var})
            list(APPEND _args "-D${_cache_var}=${${_cache_var}}")
        endif()
    endforeach()
    set(${out_var} "${_args}" PARENT_SCOPE)
endfunction()

function(spectral_daisy_collect_configure_args include_example_target out_var)
    spectral_daisy_collect_configure_cache_args(_cache_args)

    set(_args
        -S "${SPECTRAL_DAISY_SOURCE_DIR}"
        -B "${SPECTRAL_DAISY_BUILD_DIR}"
        -DCMAKE_TOOLCHAIN_FILE=${SPECTRAL_DAISY_TOOLCHAIN_FILE}
        ${_cache_args})

    if(include_example_target)
        list(APPEND _args -DSPECTRAL_DAISY_BUILD_EXAMPLE=ON)
    endif()

    if(NOT "${CMAKE_GENERATOR}" STREQUAL "")
        list(APPEND _args -G "${CMAKE_GENERATOR}")
    endif()

    if(NOT CMAKE_CONFIGURATION_TYPES AND NOT "${CMAKE_BUILD_TYPE}" STREQUAL "")
        list(APPEND _args -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
    endif()

    if(DEFINED CMAKE_MAKE_PROGRAM AND NOT "${CMAKE_MAKE_PROGRAM}" STREQUAL "")
        list(APPEND _args -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM})
    endif()

    set(${out_var} "${_args}" PARENT_SCOPE)
endfunction()

function(spectral_daisy_collect_build_args out_var)
    set(_args --build "${SPECTRAL_DAISY_BUILD_DIR}" --parallel)
    if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "")
        list(APPEND _args --config "${CMAKE_BUILD_TYPE}")
    endif()
    set(${out_var} "${_args}" PARENT_SCOPE)
endfunction()

function(spectral_daisy_collect_mcu_flags out_var)
    set(_flags
        "${SPECTRAL_DAISY_CPU}"
        "-mthumb"
        "${SPECTRAL_DAISY_FPU}"
        "${SPECTRAL_DAISY_FLOAT_ABI}")
    set(${out_var} "${_flags}" PARENT_SCOPE)
endfunction()

function(spectral_daisy_collect_compile_definitions out_var)
    set(_defs
        CORE_CM7
        STM32H750xx
        STM32H750IB
        ARM_MATH_CM7
        USE_HAL_DRIVER
        SPECTRAL_PLATFORM_DAISY
        SPECTRAL_EMBEDDED=1
        SPECTRAL_NO_PERF=1
        SPECTRAL_COMPACT_SEG=1
        # Daisy codec block is DAISY_AUDIO_BLOCK_SIZE=48; tighten the arm32 block cap from the
        # generic 256 default to 64 (>=48, with headroom) so accum[]/temp[] in DTCM aren't
        # over-allocated. Must stay >= DAISY_AUDIO_BLOCK_SIZE (arm32 _Static_assert bounds it).
        SPECTRAL_EMBEDDED_DEFAULT_BLOCK_SIZE=64
        SPECTRAL_BSP_MEM_HEADER=\"daisy_seed_mem.h\"
        HSE_VALUE=16000000
        DAISY_HAS_FATFS=1)

    spectral_daisy_parse_bool("${SPECTRAL_DAISY_DEBUG}" _daisy_debug_enabled)
    if(_daisy_debug_enabled)
        list(APPEND _defs DAISY_DEBUG=1 SPECTRAL_DEBUG=1)
    endif()

    spectral_daisy_parse_bool("${SPECTRAL_DAISY_WAVETABLE}" _daisy_wavetable_enabled)
    if(_daisy_wavetable_enabled)
        list(APPEND _defs SPECTRAL_USE_WAVETABLE_LUT=1)
    endif()

    if(NOT "${SPECTRAL_DAISY_BLINK_PLAYING}" STREQUAL "")
        list(APPEND _defs "SPECTRAL_LED_BLINK_PLAYING_MS=${SPECTRAL_DAISY_BLINK_PLAYING}")
    endif()
    if(NOT "${SPECTRAL_DAISY_BLINK_DONE}" STREQUAL "")
        list(APPEND _defs "SPECTRAL_LED_BLINK_DONE_MS=${SPECTRAL_DAISY_BLINK_DONE}")
    endif()
    if(NOT "${SPECTRAL_DAISY_ERROR_BLINK_ON}" STREQUAL "")
        list(APPEND _defs "SPECTRAL_ERROR_BLINK_ON_MS=${SPECTRAL_DAISY_ERROR_BLINK_ON}")
    endif()
    if(NOT "${SPECTRAL_DAISY_ERROR_BLINK_OFF}" STREQUAL "")
        list(APPEND _defs "SPECTRAL_ERROR_BLINK_OFF_MS=${SPECTRAL_DAISY_ERROR_BLINK_OFF}")
    endif()

    set(${out_var} "${_defs}" PARENT_SCOPE)
endfunction()

function(spectral_daisy_collect_include_dirs out_var)
    set(_includes
        "${SPECTRAL_DAISY_LIBDAISY_DIR}/src"
        "${SPECTRAL_DAISY_LIBDAISY_DIR}/src/per"
        "${SPECTRAL_DAISY_LIBDAISY_DIR}/src/sys"
        "${SPECTRAL_DAISY_LIBDAISY_DIR}/src/usbd"
        "${SPECTRAL_DAISY_LIBDAISY_DIR}/src/usbh"
        "${SPECTRAL_DAISY_LIBDAISY_DIR}/src/dev"
        "${SPECTRAL_DAISY_LIBDAISY_DIR}/src/util"
        "${SPECTRAL_DAISY_LIBDAISY_DIR}/Drivers/CMSIS/Include"
        "${SPECTRAL_DAISY_LIBDAISY_DIR}/Drivers/CMSIS/Device/ST/STM32H7xx/Include"
        "${SPECTRAL_DAISY_LIBDAISY_DIR}/Drivers/STM32H7xx_HAL_Driver/Inc"
        "${SPECTRAL_DAISY_LIBDAISY_DIR}/Middlewares/ST/STM32_USB_Device_Library/Core/Inc"
        "${SPECTRAL_DAISY_LIBDAISY_DIR}/Middlewares/ST/STM32_USB_Host_Library/Core/Inc"
        "${SPECTRAL_DAISY_DAISYSP_DIR}/Source")
    set(${out_var} "${_includes}" PARENT_SCOPE)
endfunction()

function(spectral_daisy_collect_common_compile_options out_var)
    # FIRMWARE profile, assembled from the groups in profiles.cmake (the SSOT). The
    # firmware uses the board opt level and its own -Wno; the section-GC, omit-fp,
    # no-unwind, and embedded fast-math groups are shared with the host-sim mirror.
    set(_options
        "-O${SPECTRAL_DAISY_OPTIMIZE}"
        ${SPECTRAL_FLAGS_WALL_WEXTRA}
        -Wno-missing-field-initializers
        ${SPECTRAL_FLAGS_SECTION_GC}
        ${SPECTRAL_FLAGS_OMIT_FP}
        ${SPECTRAL_FLAGS_FIRMWARE_NO_UNWIND})

    # Firmware math knob (SPECTRAL_DAISY_SAFE_MATH ON => deterministic, no unsafe math).
    spectral_daisy_parse_bool("${SPECTRAL_DAISY_SAFE_MATH}" _daisy_safe_math)
    if(NOT _daisy_safe_math)
        list(APPEND _options ${SPECTRAL_FLAGS_EMBEDDED_FASTMATH})
    endif()

    spectral_daisy_parse_bool("${SPECTRAL_DAISY_DEBUG}" _daisy_debug_enabled)
    if(_daisy_debug_enabled)
        list(APPEND _options -g3 -gdwarf-2)
    endif()

    set(${out_var} "${_options}" PARENT_SCOPE)
endfunction()

function(spectral_daisy_collect_example_cxx_options out_var)
    spectral_daisy_parse_bool("${SPECTRAL_DAISY_DEBUG}" _daisy_debug_enabled)
    if(_daisy_debug_enabled)
        separate_arguments(_opt_flags NATIVE_COMMAND "${SPECTRAL_DAISY_EXAMPLE_OPT_DEBUG}")
    else()
        separate_arguments(_opt_flags NATIVE_COMMAND "${SPECTRAL_DAISY_EXAMPLE_OPT_RELEASE}")
    endif()

    set(_options
        ${_opt_flags}
        -fno-exceptions
        -fno-rtti
        -fno-threadsafe-statics)

    set(${out_var} "${_options}" PARENT_SCOPE)
endfunction()

function(spectral_daisy_collect_link_directories out_var)
    set(_dirs
        "${SPECTRAL_DAISY_LIBDAISY_DIR}/build"
        "${SPECTRAL_DAISY_DAISYSP_DIR}/build")
    set(${out_var} "${_dirs}" PARENT_SCOPE)
endfunction()

function(spectral_daisy_collect_link_libraries out_var)
    set(_libs daisy DaisySP c m nosys)
    set(${out_var} "${_libs}" PARENT_SCOPE)
endfunction()

function(spectral_daisy_collect_link_options map_file out_var)
    spectral_daisy_collect_mcu_flags(_mcu_flags)

    set(_options
        ${_mcu_flags}
        -specs=nano.specs
        -specs=nosys.specs
        "-T${SPECTRAL_DAISY_LDSCRIPT}"
        "-Wl,-Map=${map_file},--cref"
        -Wl,--gc-sections
        -Wl,--print-memory-usage)

    set(${out_var} "${_options}" PARENT_SCOPE)
endfunction()

function(spectral_daisy_ensure_toolchain_tools)
    if(NOT CMAKE_OBJCOPY)
        set(CMAKE_OBJCOPY "${SPECTRAL_DAISY_TOOLCHAIN_PREFIX}objcopy" PARENT_SCOPE)
    endif()
    if(NOT CMAKE_SIZE)
        set(CMAKE_SIZE "${SPECTRAL_DAISY_TOOLCHAIN_PREFIX}size" PARENT_SCOPE)
    endif()
endfunction()

function(spectral_daisy_validate_required_paths)
    if(NOT EXISTS "${SPECTRAL_DAISY_LIBDAISY_DIR}")
        message(FATAL_ERROR
            "libDaisy directory not found: ${SPECTRAL_DAISY_LIBDAISY_DIR}\n"
            "Set SPECTRAL_DAISY_LIBDAISY_DIR or DAISY_PATH.")
    endif()

    if(NOT EXISTS "${SPECTRAL_DAISY_DAISYSP_DIR}")
        message(FATAL_ERROR
            "DaisySP directory not found: ${SPECTRAL_DAISY_DAISYSP_DIR}\n"
            "Set SPECTRAL_DAISY_DAISYSP_DIR or DAISY_PATH.")
    endif()

    if(NOT EXISTS "${SPECTRAL_DAISY_LDSCRIPT}")
        message(FATAL_ERROR
            "Daisy linker script not found: ${SPECTRAL_DAISY_LDSCRIPT}\n"
            "Set SPECTRAL_DAISY_LDSCRIPT to a valid STM32H750 linker script.")
    endif()
endfunction()
