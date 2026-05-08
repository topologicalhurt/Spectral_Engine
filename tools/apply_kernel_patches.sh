#!/usr/bin/env bash
set -euo pipefail

# TODO:
# Have we already applied this patch or patchset before? I.e. persistence

KERNEL_RE='^spectral_engine_core_pass_[1-9][0-9]*_bundle\.zip$'     # Regex to group passes under in the given parent directory
SCRIPT_LOC='scripts/apply_pass_'                                    # Patch apply script/hook; relative to parent patch folder
DEFAULT_N_MAX=5                                                     # Number of workers we use to unzip in parallel; will NOT exhaust any more than this many patches
APPLY_LIST=()                                                       # List of patch indices to run the apply script against; will NOT run any by default
DELETE=false                                                        # Delete the archive after inflating (y/n)? Default is n.

function help() {
  [[ -n "${1:-}" ]] && echo "Unknown option: $1" >&2
  echo "Usage: $0 -f directory [-n patches (default: $DEFAULT_N_MAX)] [-d] [-a|--apply index1 index2 ...]" >&2
  exit 1
}

# Parse initial short flags
while getopts "f:n:d" opt; do
  case $opt in
    f) FILE="$OPTARG" ;;
    n) N_PATCHES="$OPTARG" ;;
    d) DELETE=true ;;
    *) help ;;
  esac
done

shift $((OPTIND - 1))

# Parse remaining long flags or aliases
while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--apply)
      shift
      # The list of indices we apply IN-ORDER against the patchset
      while [[ $# -gt 0 && "$1" != -* ]]; do
        APPLY_LIST+=("$1")
        shift
      done
      ;;
    -h|--help)
      help
      ;;
    *)
      help "$1"
      ;;
  esac
done

# Validate and apply defaults
: "${FILE:?Usage: $0 -f directory [-n patches (default: $DEFAULT_N_MAX)]}"
: "${N_PATCHES:=$DEFAULT_N_MAX}"

if ! [[ "$N_PATCHES" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: -n must be a positive integer >= 1" >&2
  exit 1
fi

[[ -d "$FILE" ]] || { echo "Error: '$FILE' is not a directory" >&2; exit 1; }

# 3 birds 1 stone: 
#   (1) Sort at collection time so indices are deterministic
#   (2) Ensure that we found patches
#   (3) Get the number of patches we found

mapfile -t PATCHES < <(ls "$FILE" | grep -E "$KERNEL_RE" | sort || true)

if [[ ${#PATCHES[@]} -eq 0 ]]; then
  echo "No patch bundles found in '$FILE'" >&2
  exit 0
fi

echo "Found ${#PATCHES[@]} patch bundle(s) in '$FILE'"

# Unzip patches, conditionally deleting the archive based on -d flag
printf '%s\n' "${PATCHES[@]}" \
  | xargs -P "$N_PATCHES" -I {} \
      sh -c 'unzip "$3/$1" -d "$3" && if [[ "$2" == "true" ]]; then rm "$3/$1"; fi' _ {} "$DELETE" "$FILE"

# Run apply scripts for requested indices
if [[ ${#APPLY_LIST[@]} -gt 0 ]]; then
  for idx in "${APPLY_LIST[@]}"; do

    # Validate index is a non-negative integer
    if ! [[ "$idx" =~ ^[0-9]+$ ]]; then
      echo "Error: apply index '$idx' must be a non-negative integer" >&2
      exit 1
    fi

    # Validate index is in range
    if [[ "$idx" -ge "${#PATCHES[@]}" ]]; then
      echo "Error: index $idx out of range (valid: 0-$((${#PATCHES[@]} - 1)))" >&2
      exit 1
    fi

    patch="${PATCHES[$idx]}"
    PASS_NUM=$(echo "$patch" | sed 's/^spectral_engine_core_pass_\([0-9]*\)_bundle\.zip$/\1/')
    BUNDLE_DIR="$FILE/${patch%.zip}"
    APPLY_SCRIPT="$BUNDLE_DIR/${SCRIPT_LOC}$PASS_NUM"

    if [[ ! -f "$APPLY_SCRIPT" ]]; then
      echo "Error: apply script not found at '$APPLY_SCRIPT'" >&2
      exit 1
    fi

    # Finally, we apply the patch script if we supplied an index to apply against
    python3 "$APPLY_SCRIPT"
  done
fi

exit 0
