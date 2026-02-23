# Shared host build configuration and target helper functions.

set(SPECTRAL_COMMON_COMPILE_OPTIONS
    -O3
    -Wall
    -Wextra
    -Wno-unknown-pragmas
    -funroll-loops
    -ftree-vectorize
    -fomit-frame-pointer)

if(NOT SPECTRAL_REPRO_BUILD)
    list(APPEND SPECTRAL_COMMON_COMPILE_OPTIONS
        -ffast-math
        -fno-signed-zeros
        -fno-trapping-math
        -fassociative-math
        -freciprocal-math
        -ffp-contract=fast
        -march=native
        -mtune=native)
endif()

set(SPECTRAL_COMMON_LINK_OPTIONS)
set(SPECTRAL_CPU_LINK_LIBS)
set(SPECTRAL_GPU_LINK_LIBS)
set(SPECTRAL_PLATFORM_INCLUDE_DIRS)

if(APPLE)
    execute_process(
        COMMAND brew --prefix
        OUTPUT_VARIABLE BREW_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(BREW_PREFIX STREQUAL "")
        set(BREW_PREFIX "/opt/homebrew")
    endif()

    set(OMP_PREFIX "${BREW_PREFIX}/opt/libomp")
    set(SNDFILE_PREFIX "${BREW_PREFIX}/opt/libsndfile")

    if(NOT EXISTS "${OMP_PREFIX}/lib/libomp.dylib")
        message(WARNING "libomp not found at ${OMP_PREFIX}. Install with: brew install libomp")
    endif()
    if(NOT EXISTS "${SNDFILE_PREFIX}/lib/libsndfile.dylib")
        message(WARNING "libsndfile not found at ${SNDFILE_PREFIX}. Install with: brew install libsndfile")
    endif()

    list(APPEND SPECTRAL_COMMON_COMPILE_OPTIONS
        -flto=thin
        -fvectorize
        -fslp-vectorize
        -Xpreprocessor
        -fopenmp)
    list(APPEND SPECTRAL_COMMON_LINK_OPTIONS -flto=thin)
    list(APPEND SPECTRAL_PLATFORM_INCLUDE_DIRS
        "${OMP_PREFIX}/include"
        "${SNDFILE_PREFIX}/include")

    find_library(SPECTRAL_OMP_LIB NAMES omp PATHS "${OMP_PREFIX}/lib")
    if(NOT SPECTRAL_OMP_LIB)
        set(SPECTRAL_OMP_LIB omp)
    endif()

    find_library(SPECTRAL_SNDFILE_LIB NAMES sndfile PATHS "${SNDFILE_PREFIX}/lib")
    if(NOT SPECTRAL_SNDFILE_LIB)
        set(SPECTRAL_SNDFILE_LIB sndfile)
    endif()

    find_library(SPECTRAL_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    find_library(SPECTRAL_METAL_FRAMEWORK Metal REQUIRED)
    find_library(SPECTRAL_FOUNDATION_FRAMEWORK Foundation REQUIRED)

    set(SPECTRAL_CPU_LINK_LIBS
        m
        ${SPECTRAL_OMP_LIB}
        ${SPECTRAL_SNDFILE_LIB}
        ${SPECTRAL_ACCELERATE_FRAMEWORK}
        spectral_xxhash)

    set(SPECTRAL_GPU_LINK_LIBS
        ${SPECTRAL_CPU_LINK_LIBS}
        ${SPECTRAL_METAL_FRAMEWORK}
        ${SPECTRAL_FOUNDATION_FRAMEWORK})
else()
    find_package(OpenMP REQUIRED)

    list(APPEND SPECTRAL_COMMON_COMPILE_OPTIONS
        -flto=auto
        -floop-nest-optimize
        -fgraphite-identity
        -fipa-pta
        ${OpenMP_C_FLAGS})
    list(APPEND SPECTRAL_COMMON_LINK_OPTIONS -flto=auto ${OpenMP_C_FLAGS})

    find_library(SPECTRAL_SNDFILE_LIB NAMES sndfile REQUIRED)
    find_library(SPECTRAL_FFTW3F_LIB NAMES fftw3f REQUIRED)

    set(SPECTRAL_CPU_LINK_LIBS
        m
        ${SPECTRAL_SNDFILE_LIB}
        ${SPECTRAL_FFTW3F_LIB}
        spectral_xxhash)

    set(SPECTRAL_GPU_LINK_LIBS ${SPECTRAL_CPU_LINK_LIBS})
endif()

list(APPEND SPECTRAL_INCLUDE_DIRS ${SPECTRAL_PLATFORM_INCLUDE_DIRS})

set(SPECTRAL_PGO_COMPILE_OPTIONS)
set(SPECTRAL_PGO_LINK_OPTIONS)
if(SPECTRAL_PGO STREQUAL "gen")
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        list(APPEND SPECTRAL_PGO_COMPILE_OPTIONS -fprofile-instr-generate)
        list(APPEND SPECTRAL_PGO_LINK_OPTIONS -fprofile-instr-generate)
    else()
        file(MAKE_DIRECTORY "${SPECTRAL_PGO_DIR}")
        list(APPEND SPECTRAL_PGO_COMPILE_OPTIONS "-fprofile-generate=${SPECTRAL_PGO_DIR}")
        list(APPEND SPECTRAL_PGO_LINK_OPTIONS "-fprofile-generate=${SPECTRAL_PGO_DIR}")
    endif()
elseif(SPECTRAL_PGO STREQUAL "use")
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        list(APPEND SPECTRAL_PGO_COMPILE_OPTIONS "-fprofile-instr-use=${SPECTRAL_PGO_PROFILE}")
        list(APPEND SPECTRAL_PGO_LINK_OPTIONS "-fprofile-instr-use=${SPECTRAL_PGO_PROFILE}")
    else()
        list(APPEND SPECTRAL_PGO_COMPILE_OPTIONS "-fprofile-use=${SPECTRAL_PGO_DIR}" -fprofile-correction)
        list(APPEND SPECTRAL_PGO_LINK_OPTIONS "-fprofile-use=${SPECTRAL_PGO_DIR}" -fprofile-correction)
    endif()
endif()

set(SPECTRAL_CUDA_COMPILE_OPTIONS
    -O3
    --use_fast_math
    --fmad=true
    -Xcompiler=-fPIC
    -Xcompiler=-ffast-math
    -Xcompiler=-funroll-loops
    -Xcompiler=-fno-lto)

set(SPECTRAL_EMBEDDED_COMPILE_OPTIONS
    -O3
    -Wall
    -Wextra
    -Wno-unknown-pragmas
    -fdata-sections
    -ffunction-sections
    -fomit-frame-pointer)
if(NOT SPECTRAL_REPRO_BUILD)
    list(APPEND SPECTRAL_EMBEDDED_COMPILE_OPTIONS -ffast-math -funsafe-math-optimizations)
endif()

function(spectral_apply_common_target target_name)
    target_include_directories(${target_name} PRIVATE ${SPECTRAL_INCLUDE_DIRS})
    target_include_directories(${target_name} SYSTEM PRIVATE ${SPECTRAL_SYSTEM_INCLUDE_DIRS})
    # Gate C/ObjC options behind a generator expression so they are never
    # forwarded to nvcc (which would pass -flto=auto to g++, embedding
    # LTO bytecode that is potentially incompatible with the GCC linker).
    target_compile_options(${target_name} PRIVATE
        $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${SPECTRAL_COMMON_COMPILE_OPTIONS} ${SPECTRAL_PGO_COMPILE_OPTIONS}>)
    target_link_options(${target_name} PRIVATE ${SPECTRAL_COMMON_LINK_OPTIONS} ${SPECTRAL_PGO_LINK_OPTIONS})
endfunction()

function(spectral_apply_embedded_target target_name)
    target_include_directories(${target_name} PRIVATE ${SPECTRAL_INCLUDE_DIRS})
    target_include_directories(${target_name} SYSTEM PRIVATE ${SPECTRAL_SYSTEM_INCLUDE_DIRS})
    target_compile_options(${target_name} PRIVATE ${SPECTRAL_EMBEDDED_COMPILE_OPTIONS})
    target_link_options(${target_name} PRIVATE ${SPECTRAL_COMMON_LINK_OPTIONS} ${SPECTRAL_PGO_LINK_OPTIONS})
    target_link_libraries(${target_name} PRIVATE ${SPECTRAL_CPU_LINK_LIBS})
endfunction()
