# Test script for simulate_smoke (see simulate-smoke-test.cmake).
# Runs the simulation binary in WORK_DIR (its output lands in WORK_DIR/output)
# and asserts it produced audio, not just a WAV header.

execute_process(
    COMMAND "${SIM_BIN}" "${INPUT_WAV}"
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "simulate exited ${_rc}\nstdout:\n${_out}\nstderr:\n${_err}")
endif()

set(_wav "${WORK_DIR}/output/out_c.wav")
if(NOT EXISTS "${_wav}")
    message(FATAL_ERROR "simulate exited 0 but wrote no ${_wav}\nstdout:\n${_out}")
endif()
file(SIZE "${_wav}" _wav_size)
if(_wav_size LESS_EQUAL 44)
    message(FATAL_ERROR "output WAV holds no audio (${_wav_size} bytes <= WAV header)")
endif()
