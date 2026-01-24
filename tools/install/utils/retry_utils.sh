#!/bin/bash
# Retry utilities for network-dependent operations
# Extracted from .github/workflows/scripts/retry_functions.sh

# Source utils for logging functions
_RETRY_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_RETRY_UTILS_DIR/utils.sh"

# Retry a single command with a specified number of attempts
# Usage: retry <retry_count> <command>
retry() {
    local retries=$1
    shift
    local cmd="$*"
    local count=0

    until eval "$cmd"; do
        count=$((count + 1))
        if [ $count -ge $retries ]; then
            log_error "Command failed after $retries retries: $cmd"
            return 1
        fi
        log_warn "Command failed (attempt $count/$retries), retrying in 5 seconds..."
        sleep 5
    done

    if [ $count -gt 0 ]; then
        log_success "Command succeeded after $count retries: $cmd"
    fi
    return 0
}

# Retry a batch of commands sequentially
# Usage: retry_commands <retry_count> <command1> <command2> ...
retry_commands() {
    local retries=$1
    shift
    local -a cmds=("$@")

    log_info "Retry config: max retries = $retries"
    log_info "Total commands to execute: ${#cmds[@]}"

    for cmd in "${cmds[@]}"; do
        log_info "Executing command: $cmd"
        retry $retries "$cmd"
        local cmd_exit_code=$?
        if [ $cmd_exit_code -ne 0 ]; then
            log_error "Batch commands failed at: $cmd"
            return $cmd_exit_code
        fi
    done

    log_success "All batch commands executed successfully!"
    return 0
}

# Retry pip install with a requirements file
# Usage: retry_pip_install <requirements_file> [retry_count]
retry_pip_install() {
    local requirements_file=$1
    local retries=${2:-3}

    if [ ! -f "$requirements_file" ]; then
        log_error "Requirements file not found: $requirements_file"
        return 1
    fi

    log_info "Installing from $requirements_file with $retries retries"
    retry $retries "pip install -r '$requirements_file'"
}

# Retry git clone operation
# Usage: retry_git_clone <repo_url> <target_dir> [retry_count]
retry_git_clone() {
    local repo_url=$1
    local target_dir=$2
    local retries=${3:-3}

    log_info "Cloning $repo_url to $target_dir with $retries retries"
    retry $retries "rm -rf '$target_dir' && git clone '$repo_url' '$target_dir'"
}
