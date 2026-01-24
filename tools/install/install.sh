#!/bin/bash
# Master installation orchestrator script
# Delegates to task-specific install scripts
#
# Task Discovery:
#   Valid tasks are discovered from platform configuration files
#   (tests/test_utils/config/platforms/*.yaml) which define supported
#   tasks under the functional tests section. Install scripts serve
#   as a fallback to ensure all tasks with implementations are recognized.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils/utils.sh"
source "$SCRIPT_DIR/utils/retry_utils.sh"
source "$SCRIPT_DIR/utils/conda_utils.sh"

# Get project root
PROJECT_ROOT=$(get_project_root)

# Default values
TASK=""
PLATFORM="cuda"  # Default to CUDA platform
RETRY_COUNT="3"
CONDA_ENV=""      # Optional: conda environment to activate
CONDA_PATH=""     # Optional: custom conda installation path
DEV_MODE="false"  # Install development dependencies (build, lint, test)

# Dynamically discover valid tasks from platform configuration
discover_valid_tasks() {
    local tasks=()
    local parse_config="$PROJECT_ROOT/tests/test_utils/runners/parse_config.py"

    # Primary method: Get tasks from platform configuration
    # This is the source of truth for which tasks are supported on the platform
    if [ -f "$parse_config" ] && command -v python >/dev/null 2>&1; then
        # Use parse_config.py to get functional tests from platform config
        # Extract task names (top-level keys) from the JSON output
        while IFS= read -r task; do
            if [ -n "$task" ]; then
                tasks+=("$task")
            fi
        done < <(python "$parse_config" --platform "$PLATFORM" --type functional 2>/dev/null | \
                 python -c "import sys, json; data = json.load(sys.stdin); print('\\n'.join(data.keys()))" 2>/dev/null || true)
    fi

    # Fallback method: Get tasks from install scripts that exist
    # This ensures tasks with install scripts but no tests yet are still valid
    if [ -d "$SCRIPT_DIR/$PLATFORM" ]; then
        for script in "$SCRIPT_DIR/$PLATFORM"/install_*.sh; do
            if [ -f "$script" ]; then
                task=$(basename "$script" | sed 's/^install_//' | sed 's/\.sh$//')
                if [ "$task" != "base" ]; then
                    # Add task if not already in array
                    if [[ ! " ${tasks[@]} " =~ " ${task} " ]]; then
                        tasks+=("$task")
                    fi
                fi
            fi
        done
    fi

    # Always add 'all' as a valid task for installing all dependencies
    tasks+=("all")

    # Return space-separated list
    echo "${tasks[@]}"
}

# Dynamically discover valid platforms from test config
discover_valid_platforms() {
    local platforms=()

    # Get platforms from test_utils/config/platforms/*.yaml files
    local config_dir="$PROJECT_ROOT/tests/test_utils/config/platforms"
    if [ -d "$config_dir" ]; then
        for config_file in "$config_dir"/*.yaml; do
            if [ -f "$config_file" ]; then
                platform=$(basename "$config_file" .yaml)
                # Skip template files
                if [ "$platform" != "template" ]; then
                    platforms+=("$platform")
                fi
            fi
        done
    fi

    # Return space-separated list
    echo "${platforms[@]}"
}

# Arrays to hold valid tasks and platforms (populated after parse_args)
VALID_TASKS=()
VALID_PLATFORMS=()

# Initialize valid platforms (can be done early as it doesn't depend on user input)
init_valid_platforms() {
    VALID_PLATFORMS=($(discover_valid_platforms))
}

# Initialize valid tasks (must be done after PLATFORM is known)
init_valid_tasks() {
    VALID_TASKS=($(discover_valid_tasks))
}

usage() {
    # Ensure platforms are discovered for help display
    if [ ${#VALID_PLATFORMS[@]} -eq 0 ]; then
        init_valid_platforms
    fi
    # Discover tasks for the current platform (default or specified)
    if [ ${#VALID_TASKS[@]} -eq 0 ]; then
        init_valid_tasks
    fi

    cat << EOF
Usage: $0 [OPTIONS]

Master installation script for FlagScale dependencies.

OPTIONS:
    --task TASK              Task type (required, see discovered tasks below)
    --platform PLATFORM      Platform: ${VALID_PLATFORMS[*]} (default: cuda)
    --retry-count N          Number of retry attempts (default: 3)
    --conda-env ENV          Optional: activate conda environment before install
    --conda-path PATH        Optional: custom conda installation path
    --dev                    Install development dependencies (build, lint, test)
    --help                   Show this help message

EXAMPLES:
    # Install training dependencies for CUDA platform
    $0 --task train --platform cuda

    # Install hetero_train dependencies (defaults to CUDA)
    $0 --task hetero_train

    # Install all task dependencies
    $0 --task all --platform cuda

    # Install with development dependencies (includes build, lint, test)
    $0 --task train --dev

    # Install with custom retry count
    $0 --task train --retry-count 5

TASK DISCOVERY:
    Tasks are discovered from platform configuration files:
      - Primary: tests/test_utils/config/platforms/\${PLATFORM}.yaml
      - Fallback: install/\${PLATFORM}/install_*.sh scripts

DISCOVERED VALID TASKS (for platform: $PLATFORM):
$(printf '    %s\n' "${VALID_TASKS[@]}")

DISCOVERED VALID PLATFORMS:
$(printf '    %s\n' "${VALID_PLATFORMS[@]}")

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --task)
                TASK="$2"
                shift 2
                ;;
            --platform)
                PLATFORM="$2"
                shift 2
                ;;
            --retry-count)
                RETRY_COUNT="$2"
                shift 2
                ;;
            --conda-env)
                CONDA_ENV="$2"
                shift 2
                ;;
            --conda-path)
                CONDA_PATH="$2"
                shift 2
                ;;
            --dev)
                DEV_MODE="true"
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

validate_inputs() {
    # Initialize valid platforms first
    init_valid_platforms

    # Check if platform is valid (must validate platform before discovering tasks)
    local valid=false
    for valid_platform in "${VALID_PLATFORMS[@]}"; do
        if [ "$PLATFORM" = "$valid_platform" ]; then
            valid=true
            break
        fi
    done

    if [ "$valid" = "false" ]; then
        log_error "Invalid platform: $PLATFORM"
        log_error "Valid platforms: ${VALID_PLATFORMS[*]}"
        exit 1
    fi

    # Now discover valid tasks for the specified platform
    init_valid_tasks

    # Check if task is specified
    if [ -z "$TASK" ]; then
        log_error "Task not specified. Use --task to specify a task."
        usage
        exit 1
    fi

    # Check if task is valid
    valid=false
    for valid_task in "${VALID_TASKS[@]}"; do
        if [ "$TASK" = "$valid_task" ]; then
            valid=true
            break
        fi
    done

    if [ "$valid" = "false" ]; then
        log_error "Invalid task: $TASK"
        log_error "Valid tasks for platform '$PLATFORM': ${VALID_TASKS[*]}"
        exit 1
    fi

    # Validate retry count
    if ! [[ "$RETRY_COUNT" =~ ^[0-9]+$ ]] || [ "$RETRY_COUNT" -lt 1 ]; then
        log_error "Invalid retry count: $RETRY_COUNT (must be positive integer)"
        exit 1
    fi

    log_success "Input validation passed"
}

# Install platform-specific base dependencies
install_base_dependencies() {
    local base_script="$SCRIPT_DIR/$PLATFORM/install_base.sh"

    if [ ! -f "$base_script" ]; then
        log_warn "Base install script not found: $base_script (skipping)"
        return 0
    fi

    log_step "Installing base dependencies for platform: $PLATFORM"
    chmod +x "$base_script" 2>/dev/null || true
    "$base_script"
}

# Install task-specific pip requirements
install_task_requirements() {
    local task=$1
    local requirements_file

    # Use _dev.txt if --dev flag is set, otherwise use regular .txt
    if [ "$DEV_MODE" = "true" ]; then
        requirements_file="$PROJECT_ROOT/requirements/$PLATFORM/${task}_dev.txt"
        if [ ! -f "$requirements_file" ]; then
            log_warn "Dev requirements not found: $requirements_file, falling back to regular"
            requirements_file="$PROJECT_ROOT/requirements/$PLATFORM/${task}.txt"
        fi
    else
        requirements_file="$PROJECT_ROOT/requirements/$PLATFORM/${task}.txt"
    fi

    if [ ! -f "$requirements_file" ]; then
        log_info "No task requirements file: $requirements_file (skipping)"
        return 0
    fi

    log_step "Installing pip requirements for task: $task"
    if [ "$DEV_MODE" = "true" ]; then
        log_info "Installing development dependencies (includes build, lint, test)"
    fi
    retry_pip_install "$requirements_file" "$RETRY_COUNT"
}

# Install task-specific source dependencies (git repos, etc.)
install_source_dependencies() {
    local task=$1
    local source_script="$SCRIPT_DIR/$PLATFORM/install_${task}.sh"

    if [ ! -f "$source_script" ]; then
        log_info "No source dependency script for task: $task (skipping)"
        return 0
    fi

    log_step "Installing source dependencies for task: $task"
    chmod +x "$source_script" 2>/dev/null || true
    "$source_script"
}

# Install all dependencies for a task
install_task() {
    local task=$1

    print_header "Installing Dependencies for Task: $task ($PLATFORM)"

    # 1. Install base dependencies (platform-specific)
    install_base_dependencies

    # 2. Install task pip requirements
    install_task_requirements "$task"

    # 3. Install task source dependencies (git repos, etc.)
    install_source_dependencies "$task"

    log_success "Task '$task' installation complete"
}

main() {
    print_header "FlagScale Dependency Installation"

    # Parse command line arguments
    parse_args "$@"

    # Validate inputs
    validate_inputs

    # Optionally activate conda environment if specified
    if [ -n "$CONDA_ENV" ]; then
        log_step "Activating conda environment: $CONDA_ENV"
        if activate_conda "$CONDA_ENV" "$CONDA_PATH"; then
            : # Success message already displayed by activate_conda
        else
            log_warn "Conda activation failed, continuing with current environment"
        fi
    fi

    # Display current environment
    log_info "Current conda environment: $(get_conda_env)"
    check_python_version || log_warn "Python version check failed (continuing anyway)"

    # Display dev mode status
    if [ "$DEV_MODE" = "true" ]; then
        log_info "Development mode: ENABLED (will install build, lint, test deps)"
    fi

    # Install dependencies based on task
    if [ "$TASK" = "all" ]; then
        log_info "Installing dependencies for all tasks"
        # Install all valid tasks except 'all' itself
        for task in "${VALID_TASKS[@]}"; do
            if [ "$task" != "all" ]; then
                print_separator
                install_task "$task"
            fi
        done
    else
        install_task "$TASK"
    fi

    print_header "Installation Complete"
    log_success "All dependencies installed successfully for task: $TASK"
}

# Make all install scripts executable
chmod +x "$SCRIPT_DIR"/*/install_*.sh 2>/dev/null || true
chmod +x "$SCRIPT_DIR"/utils/*.sh 2>/dev/null || true

# Run main function
main "$@"
