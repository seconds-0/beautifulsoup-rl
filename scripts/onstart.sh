#!/bin/bash
#
# Vast.ai Auto-Start Script
#
# This script runs when a Vast.ai instance starts (or resumes from pause).
# It downloads the latest checkpoint from B2 and resumes training.
#
# Requirements:
#   - b2 CLI installed and authorized
#   - jq for JSON parsing
#   - Environment variables in /root/.env or passed via Vast.ai
#
# Usage:
#   Called automatically by Vast.ai onstart, or manually:
#   RUN_ID=my-run ./onstart.sh

set -euo pipefail

# Configuration
RUN_ID="${RUN_ID:-default}"
CKPT_DIR="${CKPT_DIR:-/root/checkpoints}"
CONFIG_PATH="${CONFIG_PATH:-/root/config.toml}"
B2_BUCKET="${B2_BUCKET:-beautifulsoup-rl}"
LOCK_FILE="/tmp/training.lock"
LOG_FILE="/tmp/training.log"

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Load environment from .env if it exists
load_env() {
    if [[ -f /root/.env ]]; then
        log "Loading environment from /root/.env"
        set -a
        source /root/.env
        set +a
    fi
}

# Check required tools
check_dependencies() {
    local missing=()

    if ! command -v b2 &>/dev/null; then
        missing+=("b2")
    fi

    if ! command -v jq &>/dev/null; then
        missing+=("jq")
    fi

    if ! command -v uv &>/dev/null; then
        missing+=("uv")
    fi

    if (( ${#missing[@]} > 0 )); then
        log "ERROR: Missing required tools: ${missing[*]}"
        log "Run pod_setup.sh first to install dependencies"
        exit 1
    fi
}

# Download latest.json pointer from B2
get_latest_pointer() {
    log "Checking for existing checkpoints in B2..."

    if b2 file download "b2://$B2_BUCKET/$RUN_ID/latest.json" /tmp/latest.json 2>/dev/null; then
        log "Found latest.json pointer"
        cat /tmp/latest.json
        return 0
    else
        log "No existing checkpoints found for run: $RUN_ID"
        return 1
    fi
}

# Download latest checkpoint from B2
download_checkpoint() {
    local latest_step="$1"

    log "Downloading checkpoint: $latest_step"

    mkdir -p "$CKPT_DIR/$latest_step"

    if b2 sync "b2://$B2_BUCKET/$RUN_ID/$latest_step/" "$CKPT_DIR/$latest_step/"; then
        log "Successfully downloaded checkpoint"

        # Verify checkpoint integrity
        if [[ -f "$CKPT_DIR/$latest_step/optimizer.pt" ]]; then
            log "Checkpoint verified (optimizer.pt present)"
            return 0
        else
            log "WARNING: Checkpoint may be incomplete (missing optimizer.pt)"
            return 1
        fi
    else
        log "ERROR: Failed to download checkpoint"
        return 1
    fi
}

# Download config from B2
download_config() {
    log "Downloading training config from B2..."

    if b2 file download "b2://$B2_BUCKET/$RUN_ID/config.toml" "$CONFIG_PATH" 2>/dev/null; then
        log "Successfully downloaded config.toml"
        return 0
    else
        log "WARNING: No config.toml found in B2"
        log "Using local config if available, or training will fail"
        return 1
    fi
}

# Start checkpoint sync daemon in background
start_checkpoint_sync() {
    log "Starting checkpoint sync daemon..."

    # Kill any existing sync process
    pkill -f "checkpoint_sync.sh" 2>/dev/null || true

    # Start new sync daemon
    nohup /root/prime-rl/scripts/checkpoint_sync.sh >> /tmp/checkpoint_sync.log 2>&1 &
    log "Checkpoint sync daemon started (PID: $!)"
}

# Start training
start_training() {
    local resume_step="${1:-}"

    log "Starting training..."

    cd /root/prime-rl || {
        log "ERROR: /root/prime-rl not found"
        exit 1
    }

    # Build command
    local cmd="uv run rl @ $CONFIG_PATH --ckpt --ckpt.interval 5 --ckpt.keep-last 3"

    if [[ -n "$resume_step" ]]; then
        cmd+=" --ckpt.resume-step $resume_step"
        log "Resuming from step $resume_step"
    else
        log "Starting fresh training"
    fi

    # Export WandB run ID for continuity
    if [[ -n "${WANDB_RUN_ID:-}" ]]; then
        export WANDB_RUN_ID
        log "Using WandB run ID: $WANDB_RUN_ID"
    fi

    # Use flock to prevent duplicate training processes
    log "Executing: $cmd"
    exec flock -n "$LOCK_FILE" $cmd
}

# Main entry point
main() {
    log "=========================================="
    log "Vast.ai Auto-Start Script"
    log "Run ID: $RUN_ID"
    log "=========================================="

    # Load environment
    load_env

    # Check dependencies
    check_dependencies

    # Check if training is already running
    if flock -n "$LOCK_FILE" true 2>/dev/null; then
        log "No existing training process found"
    else
        log "Training process already running, exiting"
        exit 0
    fi

    # Try to resume from checkpoint
    local resume_step=""

    if get_latest_pointer; then
        # Parse latest.json
        local latest_step
        latest_step=$(jq -r .step /tmp/latest.json)

        # Get WandB run ID for continuity
        export WANDB_RUN_ID
        WANDB_RUN_ID=$(jq -r '.wandb_run_id // empty' /tmp/latest.json)

        if [[ -n "$latest_step" ]] && [[ "$latest_step" != "null" ]]; then
            log "Latest checkpoint: $latest_step"

            # Download the checkpoint
            if download_checkpoint "$latest_step"; then
                resume_step="${latest_step#step_}"
            else
                log "WARNING: Failed to download checkpoint, will try to start fresh"
            fi
        fi

        # Download config
        download_config || true
    fi

    # Start checkpoint sync daemon
    start_checkpoint_sync

    # Start training
    start_training "$resume_step"
}

main "$@"
