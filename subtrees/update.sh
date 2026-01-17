#!/usr/bin/env bash

# Subtree management script
#
# Usage:
#   ./update.sh pull                    Pull all subtrees from libs.txt
#   ./update.sh pull <path>...          Pull only matching subtrees
#   ./update.sh push                    Push all subtrees to upstream
#   ./update.sh push <path>...          Push only matching subtrees
#   ./update.sh remove <path>...        Remove directory (no libs.txt needed)
#   ./update.sh list                    List configured subtrees
#
# libs.txt format (one entry per line):
#   <repo_url>,<local_path>,<branch>
#
# Example:
#   https://github.com/user/repo,subtrees/mylib,main

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIBS_FILE="$SCRIPT_DIR/libs.txt"

STASH_MSG=""
STASHED=0

die()  { echo "Error: $*" >&2; exit 1; }
warn() { echo "Warning: $*" >&2; }
info() { echo "$*"; }

# Normalize path: strip ./ prefix and trailing /
normalize() {
    local p="$1"
    p="${p#./}"
    p="${p%/}"
    echo "$p"
}

# Check for any uncommitted changes
has_changes() {
    ! git -C "$REPO_ROOT" diff --cached --quiet 2>/dev/null && return 0
    ! git -C "$REPO_ROOT" diff --quiet 2>/dev/null && return 0
    [[ -n "$(git -C "$REPO_ROOT" ls-files --others --exclude-standard 2>/dev/null)" ]] && return 0
    return 1
}

# Stash uncommitted changes
stash_if_needed() {
    if has_changes; then
        STASH_MSG="update.sh-$$-$(date +%s)"
        info "Stashing local changes..."
        if ! git -C "$REPO_ROOT" stash push --include-untracked -m "$STASH_MSG" -q; then
            die "Failed to stash. Commit or stash manually first."
        fi
        STASHED=1
    fi
}

# Restore stash (called on exit)
restore_stash() {
    if [[ $STASHED -eq 1 && -n "$STASH_MSG" ]]; then
        local ref
        ref="$(git -C "$REPO_ROOT" stash list | grep -F "$STASH_MSG" | head -1 | cut -d: -f1)"
        if [[ -n "$ref" ]]; then
            info "Restoring stashed changes..."
            git -C "$REPO_ROOT" stash pop "$ref" -q 2>/dev/null || \
                warn "Could not restore stash. Run 'git stash pop' manually."
        fi
        STASHED=0
    fi
}

trap restore_stash EXIT

# Validate git repo
git -C "$REPO_ROOT" rev-parse --git-dir &>/dev/null || die "Not a git repository: $REPO_ROOT"

# Read libs.txt into LIB_ENTRIES array (repo,path,branch per line)
# Must be called before stashing since file might be untracked
read_libs_file() {
    LIB_ENTRIES=()
    [[ ! -f "$LIBS_FILE" ]] && return 1
    
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines and comments
        line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [[ -z "$line" || "$line" == \#* ]] && continue
        
        LIB_ENTRIES+=("$line")
    done < "$LIBS_FILE"
    return 0
}

# Parse a libs.txt line into REPO, LOCAL_PATH, BRANCH globals
parse_lib_entry() {
    local line="$1"
    IFS=',' read -r REPO LOCAL_PATH BRANCH <<< "$line"
    REPO="$(echo "$REPO" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    LOCAL_PATH="$(echo "$LOCAL_PATH" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    BRANCH="$(echo "$BRANCH" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    BRANCH="${BRANCH:-main}"
}

# Check if path matches filter list. Empty list = match all.
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
        
        # Exact match
        [[ "$target" == "$filter" ]] && return 0
        # Target is under filter (filter is parent)
        [[ "$target" == "$filter"/* ]] && return 0
    done
    return 1
}

do_pull() {
    read_libs_file || die "libs.txt not found at $LIBS_FILE"
    [[ ${#LIB_ENTRIES[@]} -eq 0 ]] && { info "No entries in libs.txt"; return 0; }
    
    stash_if_needed
    
    local failed=0
    for entry in "${LIB_ENTRIES[@]}"; do
        parse_lib_entry "$entry"
        
        [[ -z "$REPO" || -z "$LOCAL_PATH" ]] && continue
        match_filter "$LOCAL_PATH" "$@" || continue
        
        if [[ -d "$REPO_ROOT/$LOCAL_PATH" ]]; then
            info "Pulling $LOCAL_PATH from $BRANCH..."
            if ! git -C "$REPO_ROOT" subtree pull --prefix="$LOCAL_PATH" "$REPO" "$BRANCH" --squash -m "Update subtree $LOCAL_PATH"; then
                warn "Failed to pull $LOCAL_PATH"
                ((failed++))
            fi
        else
            info "Adding $LOCAL_PATH from $REPO ($BRANCH)..."
            if ! git -C "$REPO_ROOT" subtree add --prefix="$LOCAL_PATH" "$REPO" "$BRANCH" --squash; then
                warn "Failed to add $LOCAL_PATH"
                ((failed++))
            fi
        fi
    done
    
    [[ $failed -gt 0 ]] && return 1
    info "Done."
}

do_push() {
    read_libs_file || die "libs.txt not found at $LIBS_FILE"
    [[ ${#LIB_ENTRIES[@]} -eq 0 ]] && { info "No entries in libs.txt"; return 0; }
    
    stash_if_needed
    
    local failed=0
    for entry in "${LIB_ENTRIES[@]}"; do
        parse_lib_entry "$entry"
        
        [[ -z "$REPO" || -z "$LOCAL_PATH" ]] && continue
        match_filter "$LOCAL_PATH" "$@" || continue
        
        if [[ ! -d "$REPO_ROOT/$LOCAL_PATH" ]]; then
            warn "$LOCAL_PATH does not exist, skipping"
            continue
        fi
        
        info "Pushing $LOCAL_PATH to $REPO ($BRANCH)..."
        if ! git -C "$REPO_ROOT" subtree push --prefix="$LOCAL_PATH" "$REPO" "$BRANCH"; then
            warn "Failed to push $LOCAL_PATH"
            ((failed++))
        fi
    done
    
    [[ $failed -gt 0 ]] && return 1
    info "Done."
}

do_remove() {
    local -a paths=("$@")
    
    [[ ${#paths[@]} -eq 0 ]] && die "remove requires at least one path"
    
    stash_if_needed
    
    local failed=0
    for path in "${paths[@]}"; do
        path="$(normalize "$path")"
        local full="$REPO_ROOT/$path"
        
        if [[ ! -d "$full" ]]; then
            warn "$path does not exist"
            ((failed++))
            continue
        fi
        
        info "Removing $path..."
        rm -rf "$full"
        git -C "$REPO_ROOT" add -A "$path" 2>/dev/null || true
        
        if git -C "$REPO_ROOT" diff --cached --quiet; then
            info "  (already removed)"
        else
            git -C "$REPO_ROOT" commit -m "Remove subtree $path" && info "  Committed."
        fi
    done
    
    [[ $failed -gt 0 ]] && return 1
    info "Done."
}

do_list() {
    if [[ ! -f "$LIBS_FILE" ]]; then
        info "No libs.txt at $LIBS_FILE"
        return 0
    fi
    
    read_libs_file
    [[ ${#LIB_ENTRIES[@]} -eq 0 ]] && { info "No entries in libs.txt"; return 0; }
    
    info "Subtrees configured in libs.txt:"
    for entry in "${LIB_ENTRIES[@]}"; do
        parse_lib_entry "$entry"
        local status="[missing]"
        [[ -d "$REPO_ROOT/$LOCAL_PATH" ]] && status="[present]"
        printf "  %-40s %-10s %s (%s)\n" "$LOCAL_PATH" "$status" "$REPO" "$BRANCH"
    done
}

ACTION="${1:-pull}"
shift || true

case "$ACTION" in
    pull)           do_pull "$@" ;;
    push)           do_push "$@" ;;
    remove|rm)      do_remove "$@" ;;
    list|ls)        do_list ;;
    help|-h|--help) head -20 "$0" | tail -n +3 | sed 's/^# *//' ;;
    *)              die "Unknown action: $ACTION (use: pull, push, remove, list, help)" ;;
esac
