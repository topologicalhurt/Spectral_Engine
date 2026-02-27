/* spectral_resource_bridge.c — Standalone compilation unit for the Python
 * ctypes bridge.
 *
 * Provides dummy definitions for the generated-table symbols that
 * spectral_resource_fs.c references in its lookup functions.  The bridge
 * library only needs the path canonicalization and FNV-1a ID functions;
 * the lookup functions are never called at build-tool time.
 *
 * Build:
 *   cc -shared -fPIC -I.. -o libspectral_resource_bridge.so \
 *       spectral_resource_fs.c spectral_resource_bridge.c
 */
#include "spectral_resource_fs.h"

/* Stub table — satisfies the linker for find_by_path / find_by_id. */
const SpectralResourceFsEntry spectral_resource_hashes[] = {{0}};
const size_t spectral_resource_hashes_count = 0;
