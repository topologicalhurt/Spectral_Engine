#!/usr/bin/env bash
set -euo pipefail

# Run from the Spectral_Engine repo root.
# This prints high-signal dependency usage and ignores vendored source itself.

if [[ ! -d spectral_engine || ! -d third_party ]]; then
  echo "Run from the Spectral_Engine repository root." >&2
  exit 1
fi

printf '\n== Repo size by top-level directory ==\n'
du -sh ./* 2>/dev/null | sort -h

printf '\n== third_party size ==\n'
du -sh third_party/* 2>/dev/null | sort -h

printf '\n== Active references outside third_party ==\n'
for term in simde xxHash qemu cimgui LLAC cmsis_6 libDaisy DaisySP SimSIMD StringZilla; do
  printf '\n-- %s --\n' "$term"
  grep -RInF "$term" \
    --exclude-dir=.git \
    --exclude-dir=build \
    --exclude-dir=third_party \
    . || true
 done

printf '\n== Build files that mention third_party ==\n'
grep -RInF 'third_party' \
  --exclude-dir=.git \
  --exclude-dir=build \
  --exclude-dir=third_party \
  CMakeLists.txt Makefile spectral_engine api examples tools .gitattributes .gitignore 2>/dev/null || true
