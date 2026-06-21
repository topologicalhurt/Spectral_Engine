# Daisy memory-budget layout contract (CTest).
#
# Host-compiles api/daisy_seed/daisy_seed_config.h so its pool-tiling
# _Static_asserts stay live: the firmware target that normally compiles the
# header needs the vendor BSP, which host/CI machines lack.
#
# Run: cmake --build build --target daisy_config_layout_test
#      && ctest --test-dir build -R daisy_config_layout

add_executable(daisy_config_layout_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_daisy_config_layout.c")

spectral_apply_common_target(daisy_config_layout_test)
target_include_directories(daisy_config_layout_test PRIVATE
    "${SPECTRAL_REPO_ROOT}/api/daisy_seed")

add_test(NAME daisy_config_layout COMMAND daisy_config_layout_test)
