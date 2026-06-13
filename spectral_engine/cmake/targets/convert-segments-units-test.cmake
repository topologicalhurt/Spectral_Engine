# convert_segments unit contract (CTest).
#
# Runs the real converter binary on a generated SPEC file and asserts every
# encoded SPQ field against an independent mirror of the boundary macros —
# the units gate for the offline conversion path (see the test script for
# the escaped defect class it pins). Skips (77) if the binary is not built.
#
# Run: cmake --build build --target convert_segments
#      && ctest --test-dir build -R convert_segments_units

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME convert_segments_units
    COMMAND ${Python3_EXECUTABLE}
        "${SPECTRAL_REPO_ROOT}/tests/core_contracts/convert_segments_units.py"
        "$<TARGET_FILE:convert_segments>")
set_tests_properties(convert_segments_units PROPERTIES SKIP_RETURN_CODE 77)
