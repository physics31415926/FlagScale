#!/bin/bash
# Base dependency installation script for CUDA platform
# Installs: common.txt + cuda/base.txt
#
# This script is called by install.sh and inherits its environment.
# It can also be run standalone for testing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

# Use inherited values or defaults for standalone execution
PROJECT_ROOT="${PROJECT_ROOT:-$(get_project_root)}"
PLATFORM="${PLATFORM:-cuda}"
RETRY_COUNT="${RETRY_COUNT:-3}"

main() {
    log_step "Installing base dependencies for $PLATFORM"

    # Install platform-agnostic common requirements
    local common_file="$PROJECT_ROOT/requirements/common.txt"
    if [ -f "$common_file" ]; then
        log_info "Installing common requirements"
        retry_pip_install "$common_file" "$RETRY_COUNT"
    fi

    # Install platform-specific base requirements
    local base_file="$PROJECT_ROOT/requirements/$PLATFORM/base.txt"
    if [ -f "$base_file" ]; then
        log_info "Installing $PLATFORM base requirements"
        retry_pip_install "$base_file" "$RETRY_COUNT"
    fi

    log_success "Base dependencies installed"
}

main "$@"
