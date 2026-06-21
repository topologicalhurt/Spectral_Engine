# Renderer-abstraction dispatch contract (CTest) — RENDERER_ABSTRACTION_PLAN.md, Stage 1.
#
# Pins the renderer registry: id -> descriptor, the per-renderer capability flags, and the
# timbre -> renderer classification. Fail-on-bug: flip a cap or a mapping in spectral_renderer.c
# and a check fails. Links only the (pure, dependency-free) registry TU.
#
# Run: cmake --build build --target renderer_dispatch_test \
#      && ctest --test-dir build -R renderer_dispatch

add_executable(renderer_dispatch_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_renderer_dispatch.c"
    "${SPECTRAL_CORE_DIR}/spectral_renderer.c")

spectral_apply_common_target(renderer_dispatch_test)
spectral_apply_dead_strip(renderer_dispatch_test)

add_test(NAME renderer_dispatch COMMAND renderer_dispatch_test)
