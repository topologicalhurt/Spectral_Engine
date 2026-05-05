#!/usr/bin/env bash
set -euo pipefail

# Spectral_Engine minimal cleanup helper.
# Run from the repository root of topologicalhurt/Spectral_Engine.
#
# Default mode: safe cleanup only (revert PR #19 / cimgui restore).
# Aggressive mode: AGGRESSIVE=1 ./cleanup_minimal_repo.sh
#   also removes unused vendored trees, keeping only simde and xxHash in third_party/libs.txt.

REVERT_PR19_COMMIT="64cf5c9b1aa47d506bebcf1fd09af8e0c4932454"
BRANCH_NAME="${BRANCH_NAME:-cleanup/minimal-correct-repo}"
AGGRESSIVE="${AGGRESSIVE:-0}"

require_clean_tree() {
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Working tree has uncommitted changes. Commit/stash them first." >&2
    exit 1
  fi
}

require_repo_root() {
  if [[ ! -f CMakeLists.txt || ! -d spectral_engine || ! -d third_party ]]; then
    echo "Run this from the Spectral_Engine repository root." >&2
    exit 1
  fi
}

remove_lines_matching() {
  local file="$1"
  shift
  [[ -f "$file" ]] || return 0
  local tmp
  tmp="$(mktemp)"
  cp "$file" "$tmp"
  for pattern in "$@"; do
    grep -v -F "$pattern" "$tmp" > "${tmp}.next" || true
    mv "${tmp}.next" "$tmp"
  done
  cat "$tmp" > "$file"
  rm -f "$tmp"
}

run_build_smoke_tests() {
  if command -v cmake >/dev/null 2>&1; then
    make clean || true
    make configure CMAKE_BUILD_TYPE=Debug CMAKE_CONFIGURE_ARGS='-DSPECTRAL_USE_CUDA=OFF'
    make desktop CMAKE_BUILD_TYPE=Debug
  else
    echo "cmake not found; skipping build smoke test" >&2
  fi
}

require_repo_root
require_clean_tree

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$current_branch" != "$BRANCH_NAME" ]]; then
  if git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
    git checkout "$BRANCH_NAME"
  else
    git checkout -b "$BRANCH_NAME"
  fi
fi

# Safe cleanup: undo PR #19. This removes third_party/cimgui and restores the
# resource hash table + third_party/libs.txt from the parent commit.
if git merge-base --is-ancestor "$REVERT_PR19_COMMIT" HEAD; then
  if git log --format=%B --max-count=50 | grep -Fq "Revert \"Merge pull request #19"; then
    echo "PR #19 already appears to be reverted; skipping revert."
  else
    git revert --no-edit -m 1 "$REVERT_PR19_COMMIT"
  fi
else
  echo "Commit $REVERT_PR19_COMMIT is not an ancestor of HEAD; skipping PR #19 revert." >&2
fi

# Defensive cleanup in case the revert hit conflicts or this is run from a branch
# where cimgui was added independently.
git rm -r --ignore-unmatch third_party/cimgui >/dev/null 2>&1 || true
remove_lines_matching third_party/libs.txt "https://github.com/cimgui/cimgui"

if [[ "$AGGRESSIVE" == "1" ]]; then
  # Keep only third-party code that is referenced by the active core build:
  #   - simde: desktop SIMD backend
  #   - xxHash: resource/content hashing
  # Remove vendor trees that are optional, future-only, or external SDKs.
  git rm -r --ignore-unmatch \
    third_party/LLAC \
    third_party/qemu \
    third_party/cmsis_6 \
    third_party/libDaisy \
    third_party/daisySP \
    third_party/SimSIMD \
    third_party/StringZilla \
    third_party/cimgui >/dev/null 2>&1 || true

  cat > third_party/libs.txt <<'EOF'
https://github.com/simd-everywhere/simde, third_party/simde, master
https://github.com/Cyan4973/xxHash, third_party/xxHash, dev
EOF

  git add third_party/libs.txt
  git commit -m "cleanup: remove unused vendored third-party trees" || true
fi

# Show references that would make aggressive removals unsafe.
echo "\nRemaining references to removable vendors (expected: README/TODO/libs only):"
grep -RInE 'cimgui|qemu|LLAC|cmsis_6|libDaisy|DaisySP|SimSIMD|StringZilla' \
  --exclude-dir=.git \
  --exclude-dir=build \
  --exclude-dir=third_party \
  . || true

echo "\nCurrent diff summary:"
git status --short
git diff --stat HEAD~3..HEAD 2>/dev/null || git diff --stat

echo "\nOptional smoke test: RUN_BUILD=1 ./cleanup_minimal_repo.sh"
if [[ "${RUN_BUILD:-0}" == "1" ]]; then
  run_build_smoke_tests
fi

echo "\nDone. Review with: git log --oneline --decorate -5 && git diff main..HEAD --stat"
