# CUDA build target.

if(SPECTRAL_USE_CUDA)
    add_custom_target(cuda DEPENDS desktop)
else()
    add_custom_target(cuda
        COMMAND ${CMAKE_COMMAND} -E echo "Error: nvcc not found. Please install CUDA toolkit."
        COMMAND ${CMAKE_COMMAND} -E echo "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"
        COMMAND ${CMAKE_COMMAND} -E echo "  Or download from: https://developer.nvidia.com/cuda-downloads"
        COMMAND ${CMAKE_COMMAND} -E false)
endif()
