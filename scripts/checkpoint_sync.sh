#!/bin/bash
#
# Checkpoint Sync to Backblaze B2
#
# Atomically syncs completed checkpoints to B2 and updates latest.json pointer.
# Run via systemd timer every 5 minutes, or manually.
#
# Requirements:
#   - b2 CLI installed and authorized
#   - Environment variables: B2_BUCKET, RUN_ID, WANDB_RUN_ID
#
# Usage:
#   ./checkpoint_sync.sh [--once]  # Run once (for manual/testing)
#   ./checkpoint_sync.sh           # Run in loop (for background daemon)

set -euo pipefail

# Configuration
CKPT_DIR="${CKPT_DIR:-/root/checkpoints}"
B2_BUCKET="${B2_BUCKET:-beautifulsoup-rl}"
RUN_ID="${RUN_ID:-default}"
SYNC_INTERVAL="${SYNC_INTERVAL:-300}"  # 5 minutes
LOCK_FILE="/tmp/checkpoint_sync.lock"

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Check if checkpoint is complete (optimizer.pt is written last by prime-rl)
is_checkpoint_complete() {
    local dir="$1"
    [[ -f "$dir/optimizer.pt" ]]
}

# Get list of already-synced checkpoints
get_synced_checkpoints() {
    b2 ls "$B2_BUCKET" "$RUN_ID/" 2>/dev/null | grep -oP 'step_\d+' | sort -u || true
}

# Sync a single checkpoint directory to B2
sync_checkpoint() {
    local dir="$1"
    local step_name
    step_name=$(basename "$dir")

    log "Syncing $step_name to B2..."

    # Use b2 sync with --skipNewer to avoid re-uploading unchanged files
    if b2 sync --skipNewer "$dir" "b2://$B2_BUCKET/$RUN_ID/$step_name/"; then
        log "Successfully synced $step_name"
        return 0
    else
        log "ERROR: Failed to sync $step_name"
        return 1
    fi
}

# Update latest.json pointer
update_latest_pointer() {
    local latest_step="$1"
    local timestamp
    timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

    # Create latest.json with all metadata needed for resume
    cat > /tmp/latest.json << EOF
{
    "step": "$latest_step",
    "run_id": "$RUN_ID",
    "wandb_run_id": "${WANDB_RUN_ID:-}",
    "timestamp": "$timestamp"
}
EOF

    log "Updating latest.json pointer to $latest_step"
    b2 file upload "$B2_BUCKET" /tmp/latest.json "$RUN_ID/latest.json"
}

# Main sync function
do_sync() {
    # Ensure checkpoint directory exists
    if [[ ! -d "$CKPT_DIR" ]]; then
        log "Checkpoint directory $CKPT_DIR does not exist, skipping"
        return 0
    fi

    # Get already-synced checkpoints
    local synced
    synced=$(get_synced_checkpoints)

    # Track the latest checkpoint we've synced
    local latest_step=""
    local synced_count=0

    # Find and sync completed checkpoints
    for dir in "$CKPT_DIR"/step_*/; do
        [[ -d "$dir" ]] || continue

        local step_name
        step_name=$(basename "$dir")

        # Skip if not complete
        if ! is_checkpoint_complete "$dir"; then
            log "Skipping incomplete checkpoint: $step_name"
            continue
        fi

        # Skip if already synced (check by presence in B2)
        if echo "$synced" | grep -q "^${step_name}$"; then
            log "Already synced: $step_name"
        else
            # Sync this checkpoint
            if sync_checkpoint "$dir"; then
                ((synced_count++))
            fi
        fi

        # Track latest (numerically)
        local step_num="${step_name#step_}"
        local latest_num="${latest_step#step_}"
        if [[ -z "$latest_step" ]] || (( step_num > latest_num )); then
            latest_step="$step_name"
        fi
    done

    # Update latest.json if we have any checkpoints
    if [[ -n "$latest_step" ]]; then
        update_latest_pointer "$latest_step"
    fi

    if (( synced_count > 0 )); then
        log "Synced $synced_count new checkpoint(s)"
    else
        log "No new checkpoints to sync"
    fi
}

# Signal handler for graceful shutdown
cleanup() {
    log "Received shutdown signal, performing final sync..."
    do_sync
    log "Final sync complete, exiting"
    rm -f "$LOCK_FILE"
    exit 0
}

# Main entry point
main() {
    # Check for required tools
    if ! command -v b2 &>/dev/null; then
        log "ERROR: b2 CLI not found. Install with: pip install b2"
        exit 1
    fi

    # Acquire lock to prevent overlapping syncs
    exec 200>"$LOCK_FILE"
    if ! flock -n 200; then
        log "Another sync is already running, exiting"
        exit 0
    fi

    # Set up signal handlers
    trap cleanup SIGTERM SIGINT SIGHUP

    log "Starting checkpoint sync for run: $RUN_ID"
    log "Checkpoint dir: $CKPT_DIR"
    log "B2 bucket: $B2_BUCKET"

    # One-shot mode (for manual runs or systemd timer)
    if [[ "${1:-}" == "--once" ]]; then
        do_sync
        exit 0
    fi

    # Daemon mode (continuous sync loop)
    while true; do
        do_sync
        log "Sleeping for $SYNC_INTERVAL seconds..."
        sleep "$SYNC_INTERVAL"
    done
}

main "$@"
