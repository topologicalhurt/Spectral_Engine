#!/usr/bin/env bash
set -euo pipefail

# Patch bundle names accepted by pass():
# E.g.
#   spectral_engine_core_pass_123_bundle.zip
#   spectral_engine_core_pass123_bundle.zip
#   pass_123_bundle.zip
#   pass123_bundle.zip
# The public CLI treats --start and --apply values as pass numbers, not array indices.

# TODO:
# (I) Have we already applied this patch or patchset before? I.e. persistence. Would need to belong in remote git history.
# (II) Ensure adherence to git workflow
# (III) Ensure apply script is signed off / securely hashed; that the patch is coming from a secure source
# (IV) Run against suite of external checks (that the control surface it is modifying uses pre-existing modules etc.)

SCRIPT_LOC='scripts/apply_pass'                                    # Patch apply script/hook; relative to parent patch folder
DEFAULT_N_MAX=5                                                     # Number of workers we use to unzip in parallel; will NOT exhaust any more than this many patches
APPLY_LIST=()                                                       # List of patch indices to run the apply script against; will NOT run any by default
DELETE=true                                                         # Delete the archive after inflating (y/n)? Default is y.

usage(){ echo "Usage: $0 [directory] [-f directory] [-n workers] [-d] [-a|--apply pass ...] [-s|--start pass] [-- extra args]" >&2; exit 1; }
die(){ echo "Error: $*" >&2; exit 1; }

# Return the numeric pass suffix from a bundle/archive name; fail if it is not a bundle.
pass(){
  local s="${1##*/}"
  s="${s%.zip}"
  [[ $s =~ ^(spectral_engine_core_)?pass_?([1-9][0-9]*)_bundle$ ]] && printf '%s\n' "${BASH_REMATCH[2]}"
}

# Emit sorted records as: pass<TAB>basename.
# kind=zip scans archives; kind=dir scans already-inflated bundle directories.
collect(){
  local kind=$1 dir=$2 f n p
  while IFS= read -r -d '' f; do
    n=${f##*/}
    p=$(pass "$n") || continue
    printf '%s\t%s\n' "$p" "$n"
  done < <(
    if [[ $kind == zip ]]; then
      find "$dir" -maxdepth 1 -type f -name '*.zip' -print0
    else
      find "$dir" -mindepth 1 -maxdepth 1 -type d -print0
    fi
  ) | sort -n -k1,1
}

# Resolve a pass number to exactly one inflated bundle directory.
bundle(){
  local want=$1 p n hit=
  while IFS=$'\t' read -r p n; do
    ((p == want)) || continue
    [[ -z $hit ]] || { echo "multiple inflated bundles for pass $want" >&2; return 2; }
    hit=$n
  done < <(collect dir "$DIR")
  [[ -n $hit ]] && printf '%s\n' "$DIR/$hit"
}

# Accept a leading positional directory before parsing flags.
[[ $# -gt 0 && $1 != -* ]] && { DIR=$1; shift; }

# Parse options. Values after -- are forwarded to each apply script.
while [[ $# -gt 0 ]]; do
  case $1 in
    -f) DIR=${2:?}; shift 2 ;;
    -n) N_PATCHES=${2:?}; shift 2 ;;
    -d) DELETE=false; shift ;;
    -a|--apply) shift; while [[ $# -gt 0 && $1 != -* ]]; do APPLY_LIST+=("$1"); shift; done ;;
    -s|--start) START_IDX=${2:?}; shift 2 ;;
    -h|--help) usage ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -*) usage ;;
  esac
done

# Validate required inputs and defaults.
[[ -n ${DIR:-} ]] || usage
N_PATCHES=${N_PATCHES:-$DEFAULT_N_MAX}
START_IDX=${START_IDX:-0}
[[ -d $DIR ]] || die "'$DIR' is not a directory"
[[ $N_PATCHES =~ ^[1-9][0-9]*$ ]] || die "-n must be >= 1"
[[ $START_IDX =~ ^[0-9]+$ ]] || die "--start must be >= 0"

# Collect archives and already-inflated bundles for logging and inflation.
PATCHES=()
while IFS=$'\t' read -r _ n; do PATCHES+=("$n"); done < <(collect zip "$DIR")

INFLATED_PASSES=()
INFLATED_GT_START=()
while IFS=$'\t' read -r p _; do
  INFLATED_PASSES+=("$p")
  ((p >= START_IDX)) && INFLATED_GT_START+=("$p")
done < <(collect dir "$DIR")

N_PATCHES_FOUND=${#PATCHES[@]}
N_INFLATED_FOUND=${#INFLATED_PASSES[@]}
N_APPS=${#APPLY_LIST[@]}

if ((N_PATCHES_FOUND > 0)); then
  echo ""
  echo "Found ${N_PATCHES_FOUND} patch archive(s) in '$DIR'"
fi

if ((N_INFLATED_FOUND > 0)); then
  echo ""
  echo "Found ${N_INFLATED_FOUND} inflated patch bundle(s) in '$DIR'"

  N_INFLATED_GT_START=${#INFLATED_GT_START[@]}
  N_INFLATED_DROPPED=$((N_INFLATED_FOUND - N_INFLATED_GT_START))
  MIN_INFLATED_KEPT=${INFLATED_GT_START[0]:-none}

  echo "$N_INFLATED_DROPPED of these were dropped. The patchset is being applied relative to $MIN_INFLATED_KEPT"
  echo ""
  [[ $N_APPS -gt 0 || $START_IDX -gt 0 ]] && echo "Note: --apply will run against inflated bundles where no archive is present."
fi

# Inflate all matching archives first. The later apply phase always resolves from dirs,
# so newly-unzipped and pre-existing bundles are handled uniformly.
if ((${#PATCHES[@]})); then
  printf '%s\0' "${PATCHES[@]}" |
    xargs -0 -P "$N_PATCHES" -I{} bash -c \
      'set -euo pipefail; unzip -q "$3/$1" -d "$3"; [[ $2 != true ]] || rm -- "$3/$1"' \
      _ {} "$DELETE" "$DIR"
fi

# If no explicit --apply list is provided, --start auto-selects every inflated pass >= start.
if ((${#APPLY_LIST[@]} == 0 && START_IDX > 0)); then
  while IFS=$'\t' read -r p _; do
    ((p >= START_IDX)) && APPLY_LIST+=("$p")
  done < <(collect dir "$DIR")
fi

# Apply requested pass numbers in user-provided order, ignoring anything before --start.
for pass_num in "${APPLY_LIST[@]}"; do
  [[ $pass_num =~ ^[1-9][0-9]*$ ]] || die "apply pass '$pass_num' must be >= 1"
  ((pass_num >= START_IDX)) || continue

  BUNDLE_DIR=$(bundle "$pass_num") || die "pass $pass_num has no unique inflated bundle in '$DIR'"
  APPLY_SCRIPT="$BUNDLE_DIR/${SCRIPT_LOC}$pass_num.py"

  [[ -f $APPLY_SCRIPT ]] || die "apply script not found at '$APPLY_SCRIPT'"
  python3 "$APPLY_SCRIPT" "${EXTRA_ARGS[@]}"
done
