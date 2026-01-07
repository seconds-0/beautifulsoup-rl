#!/bin/bash
#
# Cleanup Local Checkpoints with Smart B2 Upload
#
# PROBLEM: prime-rl's --ckpt.keep-last N cleanup runs AFTER save completes.
# If disk is nearly full, the save fails BEFORE cleanup can run.
# Also, B2 uploads are slow (~15min for 13GB) so we can't block on them.
#
# SOLUTION:
#   - Delete old checkpoints immediately (keep 1 on disk)
#   - Upload to B2 in background (non-blocking)
#   - Only start new upload if: 10+ steps passed AND previous upload done
#
# Usage:
#   nohup bash scripts/cleanup_local_checkpoints.sh --daemon 3 > /tmp/cleanup.log 2>&1 &
#
# Options:
#   --daemon N       Run every N minutes (default: 3)
#   --keep N         Keep last N checkpoints on disk (default: 1)
#   --upload-gap N   Min steps between upload starts (default: 10)
#   --no-b2          Skip B2 sync entirely
#   --dry-run        Show what would be done
#

set -e

# Configuration
PRIME_RL_DIR="${PRIME_RL_DIR:-/root/prime-rl-official}"
CKPT_DIR="${PRIME_RL_DIR}/outputs/checkpoints"
STATE_DIR="/tmp/ckpt_cleanup_state"
KEEP_LAST=${KEEP_LAST:-1}
DAEMON_INTERVAL=${DAEMON_INTERVAL:-3}  # minutes
UPLOAD_GAP=${UPLOAD_GAP:-10}  # min steps between uploads
B2_BUCKET="${B2_BUCKET:-beautifulsoup-rl}"
RUN_ID="${RUN_ID:-default}"
DRY_RUN=false
DAEMON_MODE=false
B2_SYNC=true

# State files
mkdir -p "$STATE_DIR"
UPLOAD_PID_FILE="$STATE_DIR/upload.pid"
LAST_UPLOAD_STEP_FILE="$STATE_DIR/last_upload_step"
UPLOADING_STEP_FILE="$STATE_DIR/uploading_step"  # Track which step is currently being uploaded

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --daemon)
            DAEMON_MODE=true
            if [[ $2 =~ ^[0-9]+$ ]]; then
                DAEMON_INTERVAL=$2
                shift
            fi
            ;;
        --keep)
            KEEP_LAST=$2
            shift
            ;;
        --upload-gap)
            UPLOAD_GAP=$2
            shift
            ;;
        --no-b2)
            B2_SYNC=false
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        --run-id)
            RUN_ID=$2
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

# Check if B2 is available
check_b2() {
    if ! command -v b2 &> /dev/null; then
        echo "[$1] WARNING: b2 CLI not found, disabling B2 sync"
        B2_SYNC=false
        return 1
    fi
    if ! b2 account get &> /dev/null && ! b2 account info &> /dev/null; then
        echo "[$1] WARNING: b2 not authorized, disabling B2 sync"
        B2_SYNC=false
        return 1
    fi
    return 0
}

# Check if upload is currently running
is_upload_running() {
    if [ -f "$UPLOAD_PID_FILE" ]; then
        local pid=$(cat "$UPLOAD_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0  # Running
        else
            rm -f "$UPLOAD_PID_FILE"  # Stale PID file
        fi
    fi
    return 1  # Not running
}

# Get last uploaded step number
get_last_upload_step() {
    if [ -f "$LAST_UPLOAD_STEP_FILE" ]; then
        cat "$LAST_UPLOAD_STEP_FILE"
    else
        echo "0"
    fi
}

# Extract step number from path like /path/to/step_123
get_step_num() {
    basename "$1" | sed 's/step_//'
}

# Get currently uploading step number (0 if none)
get_uploading_step() {
    if [ -f "$UPLOADING_STEP_FILE" ]; then
        cat "$UPLOADING_STEP_FILE"
    else
        echo "0"
    fi
}

# Background upload function
do_background_upload() {
    local ckpt_path=$1
    local step_name=$(basename "$ckpt_path")
    local step_num=$(get_step_num "$ckpt_path")
    local b2_dest="b2://${B2_BUCKET}/${RUN_ID}/checkpoints/${step_name}"
    local now=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$now] Starting background upload: $step_name -> $b2_dest"

    # Track which step we're uploading (for deletion protection)
    echo "$step_num" > "$UPLOADING_STEP_FILE"

    # Run upload in background
    (
        if b2 sync --no-progress "$ckpt_path" "$b2_dest" 2>&1; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Upload complete: $step_name"
            echo "$step_num" > "$LAST_UPLOAD_STEP_FILE"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Upload FAILED: $step_name"
        fi
        rm -f "$UPLOAD_PID_FILE"
        rm -f "$UPLOADING_STEP_FILE"  # Clear uploading marker
    ) &

    local upload_pid=$!
    echo "$upload_pid" > "$UPLOAD_PID_FILE"
    echo "[$now] Upload started in background (PID: $upload_pid)"
}

# Maybe start an upload (smart logic)
maybe_start_upload() {
    local newest_ckpt=$1
    local now=$2

    if [ "$B2_SYNC" != true ]; then
        return
    fi

    # Check if upload already running
    if is_upload_running; then
        echo "[$now] Upload already in progress, skipping"
        return
    fi

    local newest_step=$(get_step_num "$newest_ckpt")
    local last_upload=$(get_last_upload_step)
    local step_diff=$((newest_step - last_upload))

    if [ "$step_diff" -ge "$UPLOAD_GAP" ]; then
        echo "[$now] $step_diff steps since last upload (>= $UPLOAD_GAP), starting upload"
        if [ "$DRY_RUN" = true ]; then
            echo "[$now] [dry-run] Would upload $newest_ckpt"
        else
            do_background_upload "$newest_ckpt"
        fi
    else
        echo "[$now] Only $step_diff steps since last upload (need $UPLOAD_GAP), skipping"
    fi
}

cleanup_checkpoints() {
    local now=$(date '+%Y-%m-%d %H:%M:%S')

    # Check disk usage
    local disk_used=$(df --output=pcent /root 2>/dev/null | tail -1 | tr -d '% ')
    if [ -n "$disk_used" ] && [ "$disk_used" -gt 85 ]; then
        echo "[$now] WARNING: Disk usage at ${disk_used}%"
    fi

    # Find checkpoints
    if [ ! -d "$CKPT_DIR" ]; then
        echo "[$now] Checkpoint directory not found: $CKPT_DIR"
        return
    fi

    # Get checkpoints sorted by step (newest first)
    local ckpts=($(ls -d ${CKPT_DIR}/step_* 2>/dev/null | sort -t_ -k2 -n -r))
    local count=${#ckpts[@]}

    if [ "$count" -eq 0 ]; then
        echo "[$now] No checkpoints found"
        return
    fi

    local newest_ckpt="${ckpts[0]}"
    echo "[$now] Found $count checkpoint(s), newest: $(basename $newest_ckpt)"

    # Maybe start background upload of newest checkpoint
    maybe_start_upload "$newest_ckpt" "$now"

    # Delete old checkpoints (don't wait for upload)
    if [ "$count" -le "$KEEP_LAST" ]; then
        echo "[$now] Keeping $KEEP_LAST - nothing to delete"
        return
    fi

    # Get currently uploading step to protect it from deletion
    local uploading_step=$(get_uploading_step)

    for ((i=KEEP_LAST; i<count; i++)); do
        local ckpt="${ckpts[$i]}"
        local size=$(du -sh "$ckpt" 2>/dev/null | cut -f1)
        local step=$(basename "$ckpt")
        local step_num=$(get_step_num "$ckpt")

        # DON'T delete checkpoint that's currently being uploaded
        if [ "$step_num" = "$uploading_step" ] && [ "$uploading_step" != "0" ]; then
            echo "[$now] SKIPPING $step - upload in progress"
            continue
        fi

        if [ "$DRY_RUN" = true ]; then
            echo "[$now] [dry-run] Would delete: $ckpt ($size)"
            continue
        fi

        echo "[$now] Deleting: $ckpt ($size)"
        rm -rf "$ckpt"

        # Also clean orchestrator checkpoint
        local orch_ckpt="${PRIME_RL_DIR}/outputs/run_default/checkpoints/${step}"
        if [ -d "$orch_ckpt" ]; then
            rm -rf "$orch_ckpt"
        fi
    done

    # Report disk usage
    local disk_after=$(df --output=pcent /root 2>/dev/null | tail -1 | tr -d '% ')
    if [ -n "$disk_after" ]; then
        echo "[$now] Disk usage: ${disk_after}%"
    fi
}

# Main
if [ "$DAEMON_MODE" = true ]; then
    echo "Starting smart checkpoint cleanup daemon"
    echo "  Interval: ${DAEMON_INTERVAL}m"
    echo "  Keep local: ${KEEP_LAST}"
    echo "  Upload gap: ${UPLOAD_GAP} steps"
    echo "  B2 sync: ${B2_SYNC}"
    echo "  B2 bucket: ${B2_BUCKET}"
    echo "  Run ID: ${RUN_ID}"
    echo "  PID: $$"
    echo ""

    if [ "$B2_SYNC" = true ]; then
        check_b2 "$(date '+%Y-%m-%d %H:%M:%S')"
    fi

    # Initialize last upload step if not set
    if [ ! -f "$LAST_UPLOAD_STEP_FILE" ]; then
        echo "0" > "$LAST_UPLOAD_STEP_FILE"
    fi

    while true; do
        cleanup_checkpoints
        sleep $((DAEMON_INTERVAL * 60))
    done
else
    if [ "$B2_SYNC" = true ]; then
        check_b2 "$(date '+%Y-%m-%d %H:%M:%S')"
    fi
    cleanup_checkpoints
fi
