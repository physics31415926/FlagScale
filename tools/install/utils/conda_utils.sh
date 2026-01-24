#!/bin/bash
# Conda environment management utilities

# Source utils for logging
_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_UTILS_DIR/utils.sh"

# Advanced conda activation with auto-detection of installation locations
# This function tries multiple methods to find and activate conda
# Usage: activate_conda <env_name> [conda_custom_path]
# Returns: 0 on success, 1 on failure
#
# Priority order:
#   0. Use explicitly provided conda path (if provided)
#   1. Check if conda is already in PATH
#   2. Search common conda installation locations
#   3. Use 'which' to find conda dynamically
activate_conda() {
    local env_name=$1
    local conda_custom_path=${2:-""}

    # Method 0: Use explicitly provided conda path if available
    if [ -n "$conda_custom_path" ]; then
        if [ -f "$conda_custom_path/bin/activate" ]; then
            echo "🐍 Using provided conda path: $conda_custom_path"
            source "$conda_custom_path/bin/activate" "$env_name"
            if [ $? -eq 0 ]; then
                echo "✅ Successfully activated conda environment: $env_name"
                return 0
            fi
        else
            echo "⚠️  Provided conda path not valid: $conda_custom_path"
            echo "Falling back to auto-detection..."
        fi
    fi

    # Method 1: Check if conda command is already available
    if command -v conda &> /dev/null; then
        echo "🐍 Found conda in PATH, activating environment: $env_name"
        eval "$(conda shell.bash hook)"
        conda activate "$env_name"
        if [ $? -eq 0 ]; then
            echo "✅ Successfully activated conda environment: $env_name"
            return 0
        fi
    fi

    # Method 2: Check common conda installation locations
    local conda_paths=(
        "/root/miniconda3"
        "/root/anaconda3"
        "$HOME/miniconda3"
        "$HOME/anaconda3"
        "/opt/conda"
        "/usr/local/miniconda3"
        "/usr/local/anaconda3"
    )

    for conda_path in "${conda_paths[@]}"; do
        if [ -f "$conda_path/bin/activate" ]; then
            echo "🐍 Found conda at $conda_path, activating environment: $env_name"
            source "$conda_path/bin/activate" "$env_name"
            if [ $? -eq 0 ]; then
                echo "✅ Successfully activated conda environment: $env_name"
                return 0
            fi
        fi
    done

    # Method 3: Try to find conda using which
    local conda_exe conda_base
    if conda_exe=$(which conda 2>/dev/null); then
        conda_base=$(dirname "$(dirname "$conda_exe")")
        if [ -f "$conda_base/bin/activate" ]; then
            echo "🐍 Found conda via which at $conda_base, activating environment: $env_name"
            source "$conda_base/bin/activate" "$env_name"
            if [ $? -eq 0 ]; then
                echo "✅ Successfully activated conda environment: $env_name"
                return 0
            fi
        fi
    fi

    echo "❌ Failed to find and activate conda environment: $env_name"
    return 1
}

# Display conda and Python environment information
# Usage: display_python_info
display_python_info() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Python Environment Information"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if command -v python &> /dev/null; then
        echo "Python location: $(which python)"
        echo "Python version: $(python --version 2>&1)"
    else
        echo "⚠️  Python not found in PATH"
    fi

    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        echo "Conda environment: $CONDA_DEFAULT_ENV"
        if command -v conda &> /dev/null; then
            local conda_prefix=$(conda info --base 2>/dev/null)
            if [ -n "$conda_prefix" ]; then
                echo "Conda prefix: $conda_prefix"
            fi
        fi
    else
        echo "Conda environment: none"
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Create a new conda environment
# Usage: create_conda_env <env_name> [python_version]
create_conda_env() {
    local env_name=$1
    local python_version=${2:-3.12}

    if ! command_exists conda; then
        log_error "Conda not found in PATH"
        return 1
    fi

    # Check if environment already exists
    if conda env list | grep -q "^${env_name} "; then
        log_info "Conda environment '$env_name' already exists"
        return 0
    fi

    log_step "Creating conda environment: $env_name (Python $python_version)"
    if conda create -n "$env_name" python="$python_version" -y; then
        log_success "Conda environment '$env_name' created successfully"
        return 0
    else
        log_error "Failed to create conda environment '$env_name'"
        return 1
    fi
}

# Activate a conda environment (legacy function, kept for backwards compatibility)
# Usage: activate_conda_env <env_name>
# Note: Use activate_conda for better auto-detection capabilities
activate_conda_env() {
    local env_name=$1

    if ! command_exists conda; then
        log_error "Conda not found in PATH"
        return 1
    fi

    # Get conda base directory
    local conda_base=$(conda info --base)

    if [ ! -f "$conda_base/bin/activate" ]; then
        log_error "Conda activate script not found at $conda_base/bin/activate"
        return 1
    fi

    log_step "Activating conda environment: $env_name"
    source "$conda_base/bin/activate" "$env_name"

    if [ $? -eq 0 ]; then
        log_success "Conda environment '$env_name' activated"
        log_info "Current environment: $(get_conda_env)"
        return 0
    else
        log_error "Failed to activate conda environment '$env_name'"
        return 1
    fi
}

# Check if a conda environment exists
# Usage: conda_env_exists <env_name>
conda_env_exists() {
    local env_name=$1

    if ! command_exists conda; then
        return 1
    fi

    if conda env list | grep -q "^${env_name} "; then
        return 0
    else
        return 1
    fi
}

# List all conda environments
list_conda_envs() {
    if ! command_exists conda; then
        log_error "Conda not found in PATH"
        return 1
    fi

    log_info "Available conda environments:"
    conda env list
}
