#!/usr/bin/env bash

# Subtree management script
#
# Usage:
#   ./update.sh pull [--dry-run] [<path>...]   Pull all (or matching) subtrees
#   ./update.sh push [--dry-run] [<path>...]   Push all (or matching) subtrees
#   ./update.sh remove <path>...               Remove directory and commit
#   ./update.sh list|status                    List configured subtrees
#   ./update.sh help                           Show this help
#
# libs.txt format (one entry per line):
#   <repo_url>, <local_path>, <branch>
#
# Blank lines and lines starting with # are ignored.
# Branch defaults to 'main' if omitted.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_FILE="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
SUBTREE_STASH_REF=""
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIBS_FILE="$SCRIPT_DIR/libs.txt"

# --- Color output ----------------------------------------------------------

USE_COLOR=0
[[ -t 1 ]] && USE_COLOR=1

_green()  { [[ $USE_COLOR -eq 1 ]] && printf '\033[32m%s\033[0m' "$1" || printf '%s' "$1"; }
_red()    { [[ $USE_COLOR -eq 1 ]] && printf '\033[31m%s\033[0m' "$1" || printf '%s' "$1"; }
_yellow() { [[ $USE_COLOR -eq 1 ]] && printf '\033[33m%s\033[0m' "$1" || printf '%s' "$1"; }
_bold()   { [[ $USE_COLOR -eq 1 ]] && printf '\033[1m%s\033[0m' "$1" || printf '%s' "$1"; }

die()  { printf 'Error: %s\n' "$*" >&2; exit 1; }
warn() { printf 'Warning: %s\n' "$*" >&2; }

# --- Utilities --------------------------------------------------------------

# Trim leading/trailing whitespace via parameter expansion (no subshells)
trim() {
    local v="$1"
    v="${v#"${v%%[![:space:]]*}"}"
    v="${v%"${v##*[![:space:]]}"}"
    printf '%s' "$v"
}

# Normalize path: strip ./ prefix and trailing /
normalize() {
    local p="$1"
    p="${p#./}"
    p="${p%/}"
    printf '%s' "$p"
}

# Check for any uncommitted changes (staged, unstaged, or untracked),
# but ignore updates to libs.txt and this script itself.
has_changes() {
    local -a ignore_abs=("$LIBS_FILE" "$SCRIPT_FILE")
    local -a ignore_rel=()
    local p rel

    # Convert ignored absolute paths to repo-relative paths (only if inside repo)
    for p in "${ignore_abs[@]}"; do
        if [[ "$p" == "$REPO_ROOT/"* ]]; then
            rel="${p#"$REPO_ROOT/"}"
            ignore_rel+=("$rel")
        fi
    done

    local -a excludes=()
    for rel in "${ignore_rel[@]}"; do
        excludes+=(":(exclude)$rel")
    done

    ! git -C "$REPO_ROOT" diff --cached --quiet -- . "${excludes[@]}" 2>/dev/null && return 0
    ! git -C "$REPO_ROOT" diff --quiet -- . "${excludes[@]}" 2>/dev/null && return 0

    local u
    while IFS= read -r u; do
        [[ -z "$u" ]] && continue
        local keep=1
        for rel in "${ignore_rel[@]}"; do
            [[ "$u" == "$rel" ]] && keep=0 && break
        done
        [[ $keep -eq 1 ]] && return 0
    done < <(git -C "$REPO_ROOT" ls-files --others --exclude-standard 2>/dev/null)

    return 1
}

require_clean_tree() {
    if has_changes; then
        die "Working tree is not clean. Commit or stash your changes first.
  git stash push --include-untracked"
    fi
}

_prepare_subtree_clean_tree() {
    local -a allow_abs=("$LIBS_FILE" "$SCRIPT_FILE")
    local -a allow_rel=()
    local p rel

    for p in "${allow_abs[@]}"; do
        [[ "$p" == "$REPO_ROOT/"* ]] || continue
        rel="${p#"$REPO_ROOT/"}"
        allow_rel+=("$rel")
    done

    # Collect dirty paths
    local -a dirty=()
    local rec xy path extra
    while IFS= read -r -d '' rec; do
        xy="${rec:0:2}"
        path="${rec:3}"
        dirty+=("$path")
        # porcelain -z: renames/copies have an extra NUL path
        if [[ "${xy:0:1}" == "R" || "${xy:0:1}" == "C" ]]; then
            IFS= read -r -d '' extra || true
            [[ -n "${extra:-}" ]] && dirty+=("$extra")
        fi
    done < <(git -C "$REPO_ROOT" status --porcelain=v1 -z)

    [[ ${#dirty[@]} -eq 0 ]] && return 0

    # Die if anything dirty outside allowed files
    local d ok
    for d in "${dirty[@]}"; do
        ok=0
        for rel in "${allow_rel[@]}"; do
            [[ "$d" == "$rel" ]] && ok=1 && break
        done
        [[ $ok -eq 1 ]] || die "Working tree has modifications in '$d'. Commit or stash before running subtree."
    done

    # Only allowed files are dirty: stash them so git subtree is happy
    local msg="subtree-temp-$$"
    git -C "$REPO_ROOT" stash push -u -m "$msg" -- "${allow_rel[@]}" >/dev/null
    if git -C "$REPO_ROOT" stash list -1 | grep -qF "$msg"; then
        SUBTREE_STASH_REF="stash@{0}"
    fi
}

_restore_subtree_clean_tree() {
    [[ -n "${SUBTREE_STASH_REF:-}" ]] || return 0
    git -C "$REPO_ROOT" stash pop -q "$SUBTREE_STASH_REF" || warn "stash pop had conflicts; resolve manually."
    SUBTREE_STASH_REF=""
}

# --- libs.txt parsing -------------------------------------------------------

# Globals set by parse_lib_entry
REPO=""
LOCAL_PATH=""
BRANCH=""

parse_lib_entry() {
    local line="$1"
    local rest

    REPO="${line%%,*}";   rest="${line#*,}"
    LOCAL_PATH="${rest%%,*}"; BRANCH="${rest#*,}"

    # If there was no second comma, BRANCH == LOCAL_PATH — clear it
    [[ "$BRANCH" == "$LOCAL_PATH" ]] && BRANCH=""

    REPO="$(trim "$REPO")"
    LOCAL_PATH="$(trim "$LOCAL_PATH")"
    BRANCH="$(trim "$BRANCH")"
    BRANCH="${BRANCH:-main}"
}

# Populate LIB_ENTRIES array from libs.txt
read_libs_file() {
    LIB_ENTRIES=()
    [[ ! -f "$LIBS_FILE" ]] && return 1

    while IFS= read -r line || [[ -n "$line" ]]; do
        # Trim the line
        line="$(trim "$line")"
        # Skip empty lines and comments
        [[ -z "$line" || "$line" == \#* ]] && continue
        LIB_ENTRIES+=("$line")
    done < "$LIBS_FILE"
    return 0
}

# Validate all entries upfront. Dies on fatal errors, warns on suspicious ones.
validate_entries() {
    local -a seen_paths=()
    local entry errors=0

    for entry in "${LIB_ENTRIES[@]}"; do
        parse_lib_entry "$entry"

        if [[ -z "$REPO" ]]; then
            printf '  %s: missing repo URL\n' "$entry" >&2
            errors=$((errors + 1))
            continue
        fi
        if [[ -z "$LOCAL_PATH" ]]; then
            printf '  %s: missing local path\n' "$entry" >&2
            errors=$((errors + 1))
            continue
        fi

        # Warn on repo URL that doesn't look like a URL
        if [[ "$REPO" != *"://"* && "$REPO" != *"@"* ]]; then
            warn "'$REPO' doesn't look like a URL (no :// or @)"
        fi

        # Check for duplicate paths
        local sp
        for sp in "${seen_paths[@]+"${seen_paths[@]}"}"; do
            if [[ "$sp" == "$LOCAL_PATH" ]]; then
                warn "duplicate path '$LOCAL_PATH' in libs.txt"
            fi
        done
        seen_paths+=("$LOCAL_PATH")
    done

    [[ $errors -gt 0 ]] && die "$errors invalid entries in libs.txt"
    return 0
}

# --- Filter matching --------------------------------------------------------

match_filter() {
    local target="$1"
    shift

    # No filters = match everything
    [[ $# -eq 0 ]] && return 0

    target="$(normalize "$target")"

    local filter
    for filter in "$@"; do
        filter="$(normalize "$filter")"
        [[ -z "$filter" ]] && continue
        [[ "$target" == "$filter" ]] && return 0
        [[ "$target" == "$filter"/* ]] && return 0
    done
    return 1
}

# --- Summary tracking -------------------------------------------------------

_ok_count=0
_fail_count=0
_skip_count=0

# Arrays to track per-entry results for the summary table
declare -a _summary_paths=()
declare -a _summary_results=()

record_ok()   { _ok_count=$((_ok_count + 1));   _summary_paths+=("$1"); _summary_results+=("ok"); }
record_fail() { _fail_count=$((_fail_count + 1)); _summary_paths+=("$1"); _summary_results+=("FAILED"); }
record_skip() { _skip_count=$((_skip_count + 1)); _summary_paths+=("$1"); _summary_results+=("skipped"); }

print_summary() {
    printf '\n'
    local i
    for i in "${!_summary_paths[@]}"; do
        local result="${_summary_results[$i]}"
        local path="${_summary_paths[$i]}"
        case "$result" in
            ok)      printf '  %-40s %s\n' "$path" "$(_green "$result")" ;;
            FAILED)  printf '  %-40s %s\n' "$path" "$(_red "$result")" ;;
            skipped) printf '  %-40s %s\n' "$path" "$(_yellow "$result")" ;;
        esac
    done
    printf '\n%s succeeded, %s failed, %s skipped\n' \
        "$(_green "$_ok_count")" \
        "$(_red "$_fail_count")" \
        "$(_yellow "$_skip_count")"
}

# --- Commands ---------------------------------------------------------------

_run_subtree() {
    local out
    if ! out="$(git -C "$REPO_ROOT" subtree "$@" 2>&1)"; then
        printf '\n%s\n' "$(_red "git subtree failed:")" >&2
        printf '%s\n' "$out" >&2
        return 1
    fi
    return 0
}

do_pull() {
    local dry_run=0
    local -a filters=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run|-n) dry_run=1; shift ;;
            *)            filters+=("$1"); shift ;;
        esac
    done

    read_libs_file || die "libs.txt not found at $LIBS_FILE"
    [[ ${#LIB_ENTRIES[@]} -eq 0 ]] && { printf 'No entries in libs.txt\n'; return 0; }
    validate_entries

    if [[ $dry_run -eq 0 ]]; then
        _prepare_subtree_clean_tree
        trap _restore_subtree_clean_tree RETURN
    fi

    for entry in "${LIB_ENTRIES[@]}"; do
        parse_lib_entry "$entry"
        [[ -z "$REPO" || -z "$LOCAL_PATH" ]] && continue
        match_filter "$LOCAL_PATH" "${filters[@]+"${filters[@]}"}" || continue

        if [[ $dry_run -eq 1 ]]; then
            if [[ -d "$REPO_ROOT/$LOCAL_PATH" ]]; then
                printf '  %s would pull %s from %s (%s)\n' \
                    "$(_yellow "[dry-run]")" "$(_bold "$LOCAL_PATH")" "$REPO" "$BRANCH"
            else
                printf '  %s would add %s from %s (%s)\n' \
                    "$(_yellow "[dry-run]")" "$(_bold "$LOCAL_PATH")" "$REPO" "$BRANCH"
            fi
            continue
        fi

        if [[ -d "$REPO_ROOT/$LOCAL_PATH" ]]; then
            printf '  pull  %s ← %s ... ' "$(_bold "$LOCAL_PATH")" "$BRANCH"
            if _run_subtree pull --prefix="$LOCAL_PATH" "$REPO" "$BRANCH" \
                    --squash -m "Update subtree $LOCAL_PATH"; then
                printf '%s\n' "$(_green "ok")"
                record_ok "$LOCAL_PATH"
            else
                printf '%s\n' "$(_red "FAILED")"
                record_fail "$LOCAL_PATH"
            fi
        else
            printf '  add   %s ← %s ... ' "$(_bold "$LOCAL_PATH")" "$BRANCH"
            if _run_subtree add --prefix="$LOCAL_PATH" "$REPO" "$BRANCH" \
                    --squash; then
                printf '%s\n' "$(_green "ok")"
                record_ok "$LOCAL_PATH"
            else
                printf '%s\n' "$(_red "FAILED")"
                record_fail "$LOCAL_PATH"
            fi
        fi
    done

    [[ $dry_run -eq 1 ]] || print_summary
    [[ $_fail_count -gt 0 ]] && return 1
    return 0
}

do_push() {
    local dry_run=0
    local -a filters=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run|-n) dry_run=1; shift ;;
            *)            filters+=("$1"); shift ;;
        esac
    done

    read_libs_file || die "libs.txt not found at $LIBS_FILE"
    [[ ${#LIB_ENTRIES[@]} -eq 0 ]] && { printf 'No entries in libs.txt\n'; return 0; }
    validate_entries

    if [[ $dry_run -eq 0 ]]; then
        _prepare_subtree_clean_tree
        trap _restore_subtree_clean_tree RETURN
    fi

    for entry in "${LIB_ENTRIES[@]}"; do
        parse_lib_entry "$entry"
        [[ -z "$REPO" || -z "$LOCAL_PATH" ]] && continue
        match_filter "$LOCAL_PATH" "${filters[@]+"${filters[@]}"}" || continue

        if [[ $dry_run -eq 1 ]]; then
            if [[ -d "$REPO_ROOT/$LOCAL_PATH" ]]; then
                printf '  %s would push %s to %s (%s)\n' \
                    "$(_yellow "[dry-run]")" "$(_bold "$LOCAL_PATH")" "$REPO" "$BRANCH"
            else
                printf '  %s %s does not exist, would skip\n' \
                    "$(_yellow "[dry-run]")" "$LOCAL_PATH"
            fi
            continue
        fi

        if [[ ! -d "$REPO_ROOT/$LOCAL_PATH" ]]; then
            printf '  push  %s ... %s (not present)\n' "$LOCAL_PATH" "$(_yellow "skipped")"
            record_skip "$LOCAL_PATH"
            continue
        fi

        printf '  push  %s → %s ... ' "$(_bold "$LOCAL_PATH")" "$BRANCH"
        if _run_subtree push --prefix="$LOCAL_PATH" "$REPO" "$BRANCH"; then
            printf '%s\n' "$(_green "ok")"
            record_ok "$LOCAL_PATH"
        else
            printf '%s\n' "$(_red "FAILED")"
            record_fail "$LOCAL_PATH"
        fi
    done

    [[ $dry_run -eq 1 ]] || print_summary
    [[ $_fail_count -gt 0 ]] && return 1
    return 0
}

do_remove() {
    local -a paths=("$@")
    [[ ${#paths[@]} -eq 0 ]] && die "remove requires at least one path"

    require_clean_tree

    for path in "${paths[@]}"; do
        path="$(normalize "$path")"
        local full="$REPO_ROOT/$path"

        if [[ ! -d "$full" ]]; then
            printf '  remove  %s ... %s (not found)\n' "$path" "$(_yellow "skipped")"
            record_skip "$path"
            continue
        fi

        printf '  remove  %s ... ' "$(_bold "$path")"
        rm -rf "$full"
        git -C "$REPO_ROOT" add -A "$path" 2>/dev/null || true

        if git -C "$REPO_ROOT" diff --cached --quiet; then
            printf '%s (already clean)\n' "$(_yellow "skipped")"
            record_skip "$path"
        elif git -C "$REPO_ROOT" commit -m "Remove subtree $path" >/dev/null 2>&1; then
            printf '%s\n' "$(_green "ok")"
            record_ok "$path"
        else
            printf '%s\n' "$(_red "FAILED")"
            record_fail "$path"
        fi
    done

    print_summary
    [[ $_fail_count -gt 0 ]] && return 1
    return 0
}

do_list() {
    if [[ ! -f "$LIBS_FILE" ]]; then
        printf 'No libs.txt at %s\n' "$LIBS_FILE"
        return 0
    fi

    read_libs_file
    [[ ${#LIB_ENTRIES[@]} -eq 0 ]] && { printf 'No entries in libs.txt\n'; return 0; }

    for entry in "${LIB_ENTRIES[@]}"; do
        parse_lib_entry "$entry"
        local status
        if [[ -d "$REPO_ROOT/$LOCAL_PATH" ]]; then
            status="$(_green "[present]")"
        else
            status="$(_red "[missing]")"
        fi
        printf '  %-40s %s  %s (%s)\n' "$LOCAL_PATH" "$status" "$REPO" "$BRANCH"
    done
}

show_help() {
    printf '%s\n' "$(_bold "Subtree management")"
    printf '\n'
    printf 'Usage:\n'
    printf '  %s pull [--dry-run|-n] [<path>...]   Pull (or add) subtrees\n' "$(basename "$0")"
    printf '  %s push [--dry-run|-n] [<path>...]   Push subtrees upstream\n' "$(basename "$0")"
    printf '  %s remove <path>...                  Remove and commit\n' "$(basename "$0")"
    printf '  %s list|status                       Show configured subtrees\n' "$(basename "$0")"
    printf '  %s help                              Show this help\n' "$(basename "$0")"
    printf '\n'
    printf 'libs.txt format:  repo_url, local_path, branch\n'
    printf 'Branch defaults to main if omitted.\n'
}

# --- Validate git repo ------------------------------------------------------

git -C "$REPO_ROOT" rev-parse --git-dir &>/dev/null || die "Not a git repository: $REPO_ROOT"

# --- Main dispatch -----------------------------------------------------------

ACTION="${1:-pull}"
shift || true

case "$ACTION" in
    pull)                do_pull "$@" ;;
    push)                do_push "$@" ;;
    remove|rm)           do_remove "$@" ;;
    list|ls|status)      do_list ;;
    help|-h|--help)      show_help ;;
    *)                   die "Unknown action: $ACTION (use: pull, push, remove, list, help)" ;;
esac
