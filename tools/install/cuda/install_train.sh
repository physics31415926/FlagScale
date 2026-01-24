#!/bin/bash
# Source dependencies for train task (CUDA platform)
# Installs: Megatron-LM-FL from git
#
# This script is called by install.sh after base and pip requirements.
# It only handles source dependencies (git repos, etc.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../utils/utils.sh"
source "$SCRIPT_DIR/../utils/retry_utils.sh"

# Use inherited values or defaults for standalone execution
PROJECT_ROOT="${PROJECT_ROOT:-$(get_project_root)}"
RETRY_COUNT="${RETRY_COUNT:-3}"

install_megatron_lm() {
    local megatron_dir="$PROJECT_ROOT/Megatron-LM-FL"
    local megatron_url="https://github.com/flagos-ai/Megatron-LM-FL.git"

    log_info "Installing Megatron-LM-FL"

    # Clone repository
    retry_git_clone "$megatron_url" "$megatron_dir" "$RETRY_COUNT"

    # Install from source
    cd "$megatron_dir"
    retry "$RETRY_COUNT" "pip install --no-build-isolation . -vvv"
    cd "$PROJECT_ROOT"

    log_success "Megatron-LM-FL installed"
}

main() {
    log_step "Installing source dependencies for train task"
    install_megatron_lm
}

main "$@"
