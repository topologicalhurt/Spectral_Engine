#!/usr/bin/env bash

# Subtree management script
# 
# Usage:
#   ./update.sh                     Pull updates from all upstream repos
#   ./update.sh pull                Same as above
#   ./update.sh pull lib/a lib/b    Pull only specific subtrees
#   ./update.sh push                Push all subtrees back to upstream
#   ./update.sh push lib/a lib/b    Push only specific subtrees
#   ./update.sh remove path/to/sub  Remove subtree(s) by local path
#
# libs.txt format:
#   <repo_url>,<path_in_remote_repo>,<branch>
#   https://github.com/user/repo,src/lib,main
#
# All subtrees are placed under submodules/ directory:
#   submodules/<path_in_remote_repo>

set -euo pipefail

# Get absolute paths BEFORE anything can change
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )
INFILE="$SCRIPT_DIR/libs.txt"

# Stash tracking - use unique message per invocation
STASH_MSG="update.sh-auto-stash-$$"
STASHED=0
EXIT_CODE=0
SCRIPT_DIR_WAS_UNTRACKED=0

# Cleanup function - ALWAYS runs on exit to restore stash
cleanup() {
    if [[ $STASHED -eq 1 ]]; then
        # Remove recreated script dir before popping stash to avoid conflicts
        if [[ $SCRIPT_DIR_WAS_UNTRACKED -eq 1 ]] && [[ -d "$SCRIPT_DIR" ]]; then
            rm -rf "$SCRIPT_DIR"
        fi
        
        local stash_ref
        stash_ref=$(git -C "$REPO_ROOT" stash list 2>/dev/null | grep "$STASH_MSG" | head -1 | cut -d: -f1)
        if [[ -n "$stash_ref" ]]; then
            echo "Restoring stashed changes..."
            if ! git -C "$REPO_ROOT" stash pop "$stash_ref" -q 2>/dev/null; then
                echo "Warning: Could not auto-restore stash."
                echo "Run: git stash list   # to see stashes"
                echo "Run: git stash pop    # to restore manually"
            fi
        fi
    fi
    exit $EXIT_CODE
}
trap cleanup EXIT

# Validate environment BEFORE stashing anything
if [[ ! -f "$INFILE" ]]; then
    echo "Error: $INFILE not found"
    EXIT_CODE=1
    exit 1
fi

if ! git -C "$REPO_ROOT" rev-parse --git-dir &>/dev/null; then
    echo "Error: $REPO_ROOT is not a git repository"
    EXIT_CODE=1
    exit 1
fi

# Read libs.txt into array BEFORE stashing (file might be untracked and get stashed)
mapfile -t LIB_LINES < "$INFILE"

ACTION="${1:-pull}"
shift || true

# Remaining args are filters (if any)
FILTERS=("$@")

# Check for uncommitted changes
has_changes() {
    ! git -C "$REPO_ROOT" diff --cached --quiet 2>/dev/null && return 0
    ! git -C "$REPO_ROOT" diff --quiet 2>/dev/null && return 0
    [[ -n $(git -C "$REPO_ROOT" ls-files --others --exclude-standard 2>/dev/null) ]] && return 0
    return 1
}

# Check if script directory is untracked (will disappear when stashed)
is_script_dir_untracked() {
    local rel_path
    rel_path=$(realpath --relative-to="$REPO_ROOT" "$SCRIPT_DIR" 2>/dev/null || echo "${SCRIPT_DIR#$REPO_ROOT/}")
    # Check if any file in script dir is untracked
    git -C "$REPO_ROOT" ls-files --others --exclude-standard "$rel_path" 2>/dev/null | grep -q .
}

# Stash if needed
if has_changes; then
    # Check if our script directory will disappear
    if is_script_dir_untracked; then
        SCRIPT_DIR_WAS_UNTRACKED=1
    fi
    
    echo "Stashing local changes..."
    if git -C "$REPO_ROOT" stash push --include-untracked -m "$STASH_MSG" -q 2>/dev/null; then
        STASHED=1
        
        # Recreate script directory if it was untracked and stashed away
        if [[ $SCRIPT_DIR_WAS_UNTRACKED -eq 1 ]] && [[ ! -d "$SCRIPT_DIR" ]]; then
            mkdir -p "$SCRIPT_DIR"
        fi
    else
        echo "Error: Failed to stash changes. Aborting."
        EXIT_CODE=1
        exit 1
    fi
fi

# Check if a subdir matches any filter (or no filters = match all)
# For remove action, use prefix matching; for others, exact or prefix match
matches_filter() {
    local subdir="$1"
    [[ ${#FILTERS[@]} -eq 0 ]] && return 0
    for f in "${FILTERS[@]}"; do
        # Normalize: strip leading ./ and trailing /
        local nf="${f#./}"
        nf="${nf%/}"
        local ns="${subdir#./}"
        ns="${ns%/}"
        # Exact match
        [[ "$ns" == "$nf" ]] && return 0
        # Prefix match (filter is parent of subdir)
        [[ "$ns" == "$nf"/* ]] && return 0
        # For remove: subdir is parent of filter
        [[ "$ACTION" == "remove" ]] && [[ "$nf" == "$ns"/* ]] && return 0
    done
    return 1
}

# Process each subtree (from array read before stashing)
for line in "${LIB_LINES[@]}"; do
    # Parse CSV line: repo_url, path_in_remote, branch
    IFS=',' read -r REPO_URL REMOTE_PATH BRANCH <<< "$line"
    
    # Skip empty lines and comments
    [[ -z "$REPO_URL" || "$REPO_URL" == \#* ]] && continue
    
    # Trim whitespace from fields
    REPO_URL=$(echo "$REPO_URL" | xargs)
    REMOTE_PATH=$(echo "$REMOTE_PATH" | xargs)
    BRANCH=$(echo "$BRANCH" | xargs)
    
    # Default branch to main if not specified
    BRANCH="${BRANCH:-main}"
    
    # Validate required fields
    if [[ -z "$REMOTE_PATH" ]]; then
        echo "Warning: Skipping line with missing path: $REPO_URL"
        continue
    fi
    
    # Local path is always under submodules/
    LOCAL_PATH="submodules/$REMOTE_PATH"
    
    # Skip if doesn't match filter
    matches_filter "$LOCAL_PATH" || continue
    
    case "$ACTION" in
        pull)
            if [[ -d "$REPO_ROOT/$LOCAL_PATH" ]]; then
                echo "Pulling updates for $LOCAL_PATH from $BRANCH..."
                if ! git -C "$REPO_ROOT" subtree pull --prefix="$LOCAL_PATH" "$REPO_URL" "$BRANCH" --squash; then
                    echo "Warning: Failed to pull $LOCAL_PATH"
                    EXIT_CODE=1
                fi
            else
                echo "Adding new subtree $LOCAL_PATH from $REPO_URL ($BRANCH)..."
                if ! git -C "$REPO_ROOT" subtree add --prefix="$LOCAL_PATH" "$REPO_URL" "$BRANCH" --squash; then
                    echo "Warning: Failed to add $LOCAL_PATH"
                    EXIT_CODE=1
                fi
            fi
            ;;
        push)
            if [[ -d "$REPO_ROOT/$LOCAL_PATH" ]]; then
                echo "Pushing local changes from $LOCAL_PATH to $REPO_URL ($BRANCH)..."
                if ! git -C "$REPO_ROOT" subtree push --prefix="$LOCAL_PATH" "$REPO_URL" "$BRANCH"; then
                    echo "Warning: Failed to push $LOCAL_PATH"
                    EXIT_CODE=1
                fi
            else
                echo "Warning: $LOCAL_PATH does not exist, skipping push"
            fi
            ;;
        remove)
            if [[ -d "$REPO_ROOT/$LOCAL_PATH" ]]; then
                echo "Removing subtree $LOCAL_PATH..."
                rm -rf "$REPO_ROOT/$LOCAL_PATH"
                git -C "$REPO_ROOT" add -A "$LOCAL_PATH" 2>/dev/null || true
                if git -C "$REPO_ROOT" diff --cached --quiet 2>/dev/null; then
                    echo "  Nothing to commit (already removed)"
                else
                    git -C "$REPO_ROOT" commit -m "Remove subtree $LOCAL_PATH"
                    echo "  Removed and committed."
                fi
            else
                echo "Warning: $LOCAL_PATH does not exist, nothing to remove"
            fi
            ;;
        *)
            echo "Unknown action: $ACTION"
            echo "Usage: $0 [pull|push|remove] [subtree-path ...]"
            EXIT_CODE=1
            exit 1
            ;;
    esac
done

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Done."
fi
