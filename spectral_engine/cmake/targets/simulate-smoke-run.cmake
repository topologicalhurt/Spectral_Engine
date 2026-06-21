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

# The output WAV is named after the input stem (unique per input; see
# pipeline_unique_output_wav), not a fixed "out_c.wav" — glob so the test does not
# couple to the exact naming scheme, only that audio was produced.
file(GLOB _wavs "${WORK_DIR}/output/*.wav")
if(NOT _wavs)
    message(FATAL_ERROR "simulate exited 0 but wrote no .wav under ${WORK_DIR}/output\nstdout:\n${_out}")
endif()
list(GET _wavs 0 _wav)
file(SIZE "${_wav}" _wav_size)
if(_wav_size LESS_EQUAL 44)
    message(FATAL_ERROR "output WAV (${_wav}) holds no audio (${_wav_size} bytes <= WAV header)")
endif()
