# Spectral_Engine audit + minimal cleanup plan

Scope audited: `topologicalhurt/Spectral_Engine`, default branch `main` at merge commit `64cf5c9b1aa47d506bebcf1fd09af8e0c4932454`.

## Executive summary

The repo is currently dominated by vendored third-party code, not by the actual Spectral Engine source. The most obvious cleanup is to undo PR #19 (`Add third_party back`), which reintroduced `third_party/cimgui` and changed generated resource hashes without evidence that the project uses cimgui. That one revert removes roughly 124k lines from the latest merge and restores the resource hash file and `third_party/libs.txt` to the prior state.

After that, the remaining minimal-correct target should keep only dependencies that are actually referenced by the active build:

- Keep `third_party/simde` for desktop SIMD paths.
- Keep `third_party/xxHash` for resource hashing.
- Treat Daisy/libDaisy/DaisySP as external SDK paths, because the CMake defaults already look outside the repo (`$DAISY_PATH` or `$HOME/daisy`) and the README shows explicit `-DSPECTRAL_DAISY_*=/path/to/...` usage.
- Remove or externalize unused vendor trees (`qemu`, `LLAC`, `cmsis_6`, `libDaisy`, `daisySP`, `SimSIMD`, `StringZilla`, `cimgui`) unless there is a hidden workflow that depends on them.

## Findings

### 1. Latest `main` merge should be reverted

PR #19 is titled `Add third_party back`, was merged into `main`, and changed 57 files with 124,744 additions and 2 deletions. Its changed-file list is almost entirely `third_party/cimgui/**`, plus a `third_party/libs.txt` line and two generated resource-hash ID changes.

The project search did not show active cimgui use in core build files, README usage, or CMake targets. The safest first cleanup is:

```bash
git revert -m 1 64cf5c9b1aa47d506bebcf1fd09af8e0c4932454
```

That should remove `third_party/cimgui`, remove the cimgui line from `third_party/libs.txt`, and restore `spectral_engine/core/spectral_hash_resources_xx32_xx3.c` to the pre-PR values.

### 2. The repo has a second giant merge labelled as undo-worthy

The previous merge, PR #18, is titled `TODO : undo` and changed 733 files with 134,175 additions and 10,277 deletions. It also renamed the vendor area from `subtrees/**` to `third_party/**` and touched core CMake and analysis files.

Do not blindly revert PR #18 until the code-level changes are reviewed, because it includes meaningful core refactors. But it explains why the repo feels chaotic: a large source refactor and a large vendor-tree relocation were merged together.

### 3. Build system is reasonably clean in concept

The root build entrypoint is small: root `CMakeLists.txt` only adds `spectral_engine/`, and the root `Makefile` wraps CMake targets. `spectral_engine/cmake/source-manifest.cmake` is the canonical source list, which is good. The current cleanup should preserve this model and avoid reintroducing globbing or duplicate source lists.

### 4. Required vs optional dependencies

Active in core build:

- `third_party/xxHash/xxhash.c` is compiled into `spectral_xxhash`.
- `third_party/simde` is an include directory, and SIMD oscillator code uses SIMDe types/functions.

Optional/external SDK:

- Daisy builds require `SPECTRAL_DAISY_LIBDAISY_DIR` and `SPECTRAL_DAISY_DAISYSP_DIR`; defaults point to `$DAISY_PATH` or `$HOME/daisy`, not `third_party`.

Likely removable from the repo as vendored source:

- `third_party/cimgui` — recently re-added, no active use found.
- `third_party/qemu` — mentioned as future direction, no active build use found.
- `third_party/LLAC` — no active build use found.
- `third_party/cmsis_6` — no active build use found; Daisy path uses libDaisy's CMSIS includes.
- `third_party/libDaisy`, `third_party/daisySP` — build takes external paths.
- `third_party/SimSIMD`, `third_party/StringZilla` — mentioned as TODO/investigation only.

## Recommended cleanup sequence

### Phase 1: Safe revert

1. Create a cleanup branch from `main`.
2. Revert merge commit `64cf5c9b1aa47d506bebcf1fd09af8e0c4932454`.
3. Run at least:
   ```bash
   make clean
   make configure CMAKE_BUILD_TYPE=Debug
   make desktop CMAKE_BUILD_TYPE=Debug
   make syntax-test CMAKE_BUILD_TYPE=Debug || true
   ```
4. Inspect the resulting diff. It should be essentially cimgui removal, `libs.txt` restoration, and resource hash restoration.

### Phase 2: Aggressive minimal vendor cleanup

After Phase 1 builds, remove unused vendor trees and their `third_party/libs.txt` entries. Keep only `simde` and `xxHash` in-tree unless there is a real dependency discovered by grep/build failure.

Suggested removals:

```text
third_party/LLAC
third_party/qemu
third_party/cmsis_6
third_party/libDaisy
third_party/daisySP
third_party/SimSIMD
third_party/StringZilla
third_party/cimgui
```

Suggested `third_party/libs.txt` contents after aggressive cleanup:

```text
https://github.com/simd-everywhere/simde, third_party/simde, master
https://github.com/Cyan4973/xxHash, third_party/xxHash, dev
```

### Phase 3: Make optional dependencies explicit

If you want the repo to be truly minimal, switch third-party vendoring to either:

- Git submodules for `simde` and `xxHash`, or
- CMake FetchContent with pinned commits, plus a lock file.

Do not vendor QEMU, cimgui, libDaisy, DaisySP, LLAC, CMSIS, SimSIMD, or StringZilla unless a build target directly consumes them.

### Phase 4: Correctness hardening

Add CI that runs on every PR:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DSPECTRAL_USE_CUDA=OFF
cmake --build build --target desktop --parallel
cmake --build build --target verify_resource_hashes --parallel || true
cmake --build build --target syntax_test --parallel || true
```

Then add platform-specific jobs later for macOS/Metal, Linux/CUDA, and Daisy cross-compile.

## Notes

I could not run the build locally because the working environment did not have a repository checkout and network cloning failed. The cleanup script bundled with this report is designed to be run from a normal local checkout.
