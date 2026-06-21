# Resource path-canonicalization security contract (CTest).
#
# Pins spectral_resource_path_canonical() -- the path-traversal defense and the
# byte source for the FNV-1a resource ID -- against a hand-computed expectation:
# exact canonical bytes per transform phase (lowercase, separator, control strip,
# "."/".." resolution, trailing-dot + dot-dot-space NTFS bypasses, RLE token and
# >255 chaining) plus a structural "no '..' escapes the root" invariant over a
# traversal battery. Independent of the generator (which canonicalizes via this
# same C code), so a transform regression that the C-against-C verify target would
# miss fails here.
#
# Run: cmake --build build --target resource_path_canonical_test
#      && ctest --test-dir build -R resource_path_canonical

add_executable(resource_path_canonical_test EXCLUDE_FROM_ALL
    "${SPECTRAL_REPO_ROOT}/tests/core_contracts/test_resource_path_canonical.c"
    "${SPECTRAL_CORE_DIR}/spectral_resource_fs.c"
    "${SPECTRAL_ENGINE_ROOT}/runtime/spectral_utils.c")

spectral_apply_common_target(resource_path_canonical_test)
target_link_libraries(resource_path_canonical_test PRIVATE m)

# The test calls only spectral_resource_path_canonical; dead-strip the unused
# lookup functions so the generated resource table is not a link dependency.
spectral_apply_dead_strip(resource_path_canonical_test)

add_test(NAME resource_path_canonical COMMAND resource_path_canonical_test)
