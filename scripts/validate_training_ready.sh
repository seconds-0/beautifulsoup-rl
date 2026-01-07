#!/bin/bash
#
# Validate that the pod is ready for training
#
# Run this BEFORE starting training to catch common issues.
# Returns exit code 0 if ready, non-zero otherwise.
#
# Usage:
#   bash scripts/validate_training_ready.sh
#   # OR on pod:
#   curl -sSL https://raw.githubusercontent.com/seconds-0/beautifulsoup-rl/main/scripts/validate_training_ready.sh | bash
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

echo "================================================"
echo "Prime-RL Training Readiness Check"
echo "================================================"
echo ""

# 1. Check GPU availability and memory
echo "[1/8] Checking GPUs..."
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "  ${RED}FAIL: nvidia-smi not found - no GPU available${NC}"
    ((ERRORS++))
else
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "  Found $GPU_COUNT GPU(s)"

    # Check for stale processes using GPU memory
    for i in $(seq 0 $((GPU_COUNT-1))); do
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
        if [ "$GPU_MEM" -gt 1000 ]; then
            echo -e "  ${YELLOW}WARNING: GPU $i using ${GPU_MEM}MB - may have stale processes${NC}"
            echo "    Fix: Run 'bash scripts/clean_training_state.sh' first"
            ((WARNINGS++))
        else
            echo -e "  ${GREEN}GPU $i: ${GPU_MEM}MB used (OK)${NC}"
        fi
    done
fi

# 2. Check PATH includes uv
echo ""
echo "[2/8] Checking PATH..."
if ! command -v uv &> /dev/null; then
    echo -e "  ${RED}FAIL: 'uv' not in PATH${NC}"
    echo "    Fix: export PATH=\"/root/.local/bin:\$PATH\""
    ((ERRORS++))
else
    UV_PATH=$(which uv)
    echo -e "  ${GREEN}uv found at: $UV_PATH${NC}"
fi

# 3. Check WANDB_API_KEY
echo ""
echo "[3/8] Checking WANDB_API_KEY..."
if [ -z "$WANDB_API_KEY" ]; then
    # Check if it's in ~/.wandb_api_key
    if [ -f ~/.wandb_api_key ]; then
        echo -e "  ${YELLOW}WARNING: WANDB_API_KEY not set, but ~/.wandb_api_key exists${NC}"
        echo "    Fix: source ~/.wandb_api_key"
        ((WARNINGS++))
    else
        echo -e "  ${RED}FAIL: WANDB_API_KEY not set${NC}"
        echo "    Fix: export WANDB_API_KEY=<your-key>"
        ((ERRORS++))
    fi
else
    echo -e "  ${GREEN}WANDB_API_KEY is set${NC}"
fi

# 4. Check vLLM environment variables
echo ""
echo "[4/8] Checking vLLM environment..."
if [ "$VLLM_USE_V1" != "0" ]; then
    echo -e "  ${YELLOW}WARNING: VLLM_USE_V1 not set to 0${NC}"
    echo "    Fix: export VLLM_USE_V1=0"
    ((WARNINGS++))
else
    echo -e "  ${GREEN}VLLM_USE_V1=0 (OK)${NC}"
fi

if [ "$VLLM_WORKER_MULTIPROC_METHOD" != "spawn" ]; then
    echo -e "  ${YELLOW}WARNING: VLLM_WORKER_MULTIPROC_METHOD not set to spawn${NC}"
    echo "    Fix: export VLLM_WORKER_MULTIPROC_METHOD=spawn"
    ((WARNINGS++))
else
    echo -e "  ${GREEN}VLLM_WORKER_MULTIPROC_METHOD=spawn (OK)${NC}"
fi

# 5. Check for stale checkpoints
echo ""
echo "[5/8] Checking for stale state..."
PRIME_RL_DIR="${PRIME_RL_DIR:-/root/prime-rl-official}"

STALE_DIRS=""
if [ -d "$PRIME_RL_DIR/outputs/run_default/checkpoints" ] && [ "$(ls -A $PRIME_RL_DIR/outputs/run_default/checkpoints 2>/dev/null)" ]; then
    STALE_DIRS="$STALE_DIRS outputs/run_default/checkpoints"
fi
if [ -d "$PRIME_RL_DIR/outputs/checkpoints" ] && [ "$(ls -A $PRIME_RL_DIR/outputs/checkpoints 2>/dev/null)" ]; then
    STALE_DIRS="$STALE_DIRS outputs/checkpoints"
fi
if [ -d "$PRIME_RL_DIR/outputs/run_default/rollouts" ] && [ "$(ls -A $PRIME_RL_DIR/outputs/run_default/rollouts 2>/dev/null)" ]; then
    STALE_DIRS="$STALE_DIRS outputs/run_default/rollouts"
fi

if [ -n "$STALE_DIRS" ]; then
    echo -e "  ${YELLOW}WARNING: Found stale state in:${NC}"
    for dir in $STALE_DIRS; do
        echo "    - $dir"
    done
    echo "    Fix: Run 'bash scripts/clean_training_state.sh'"
    ((WARNINGS++))
else
    echo -e "  ${GREEN}No stale state found (OK)${NC}"
fi

# 6. Check for step counter desync (CRITICAL for resume)
echo ""
echo "[6/8] Checking for step counter desync risk..."

# Get max trainer checkpoint step
TRAINER_CKPT_DIR="$PRIME_RL_DIR/outputs/checkpoints"
ORCH_CKPT_DIR="$PRIME_RL_DIR/outputs/run_default/checkpoints"

TRAINER_MAX_STEP=0
ORCH_MAX_STEP=0

if [ -d "$TRAINER_CKPT_DIR" ]; then
    for step_dir in "$TRAINER_CKPT_DIR"/step_*; do
        if [ -d "$step_dir" ]; then
            step_num=$(basename "$step_dir" | sed 's/step_//')
            if [ "$step_num" -gt "$TRAINER_MAX_STEP" ] 2>/dev/null; then
                TRAINER_MAX_STEP=$step_num
            fi
        fi
    done
fi

if [ -d "$ORCH_CKPT_DIR" ]; then
    for step_dir in "$ORCH_CKPT_DIR"/step_*; do
        if [ -d "$step_dir" ]; then
            step_num=$(basename "$step_dir" | sed 's/step_//')
            if [ "$step_num" -gt "$ORCH_MAX_STEP" ] 2>/dev/null; then
                ORCH_MAX_STEP=$step_num
            fi
        fi
    done
fi

if [ "$TRAINER_MAX_STEP" -gt 0 ] && [ "$ORCH_MAX_STEP" -gt 0 ]; then
    if [ "$ORCH_MAX_STEP" -gt "$TRAINER_MAX_STEP" ]; then
        echo -e "  ${RED}FAIL: Step counter desync detected!${NC}"
        echo "    Trainer checkpoint: step_$TRAINER_MAX_STEP"
        echo "    Orchestrator checkpoint: step_$ORCH_MAX_STEP (AHEAD!)"
        echo ""
        echo "    This WILL cause training to hang after 1 step if resumed."
        echo "    The runs.progress will init to $ORCH_MAX_STEP but DataLoader to $TRAINER_MAX_STEP."
        echo ""
        echo "    Fix: Clean ALL state before resuming:"
        echo "      bash scripts/clean_training_state.sh"
        echo "    Or: Remove orchestrator checkpoints ahead of trainer:"
        echo "      rm -rf $ORCH_CKPT_DIR/step_$ORCH_MAX_STEP"
        ((ERRORS++))
    elif [ "$TRAINER_MAX_STEP" -ne "$ORCH_MAX_STEP" ]; then
        echo -e "  ${YELLOW}WARNING: Checkpoint steps differ (trainer=$TRAINER_MAX_STEP, orch=$ORCH_MAX_STEP)${NC}"
        echo "    This may cause issues. Consider cleaning state."
        ((WARNINGS++))
    else
        echo -e "  ${GREEN}Checkpoints aligned at step $TRAINER_MAX_STEP (OK)${NC}"
    fi
elif [ "$TRAINER_MAX_STEP" -gt 0 ] || [ "$ORCH_MAX_STEP" -gt 0 ]; then
    echo -e "  ${YELLOW}WARNING: Only partial checkpoints exist${NC}"
    echo "    Trainer: step_$TRAINER_MAX_STEP, Orchestrator: step_$ORCH_MAX_STEP"
    echo "    Consider cleaning state for fresh start."
    ((WARNINGS++))
else
    echo -e "  ${GREEN}No checkpoints (fresh start OK)${NC}"
fi

# 7. Check disk space (CRITICAL: checkpoints are ~13GB each!)
echo ""
echo "[7/8] Checking disk space..."
DISK_AVAIL=$(df -BG /root 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
DISK_TOTAL=$(df -BG /root 2>/dev/null | tail -1 | awk '{print $2}' | tr -d 'G')
if [ -n "$DISK_AVAIL" ]; then
    # Calculate existing checkpoint usage
    CKPT_SIZE=$(du -s "$PRIME_RL_DIR/outputs/checkpoints" 2>/dev/null | cut -f1)
    CKPT_SIZE_GB=$((CKPT_SIZE / 1024 / 1024))
    CKPT_COUNT=$(ls -d "$PRIME_RL_DIR/outputs/checkpoints"/step_* 2>/dev/null | wc -l)

    echo "  Disk: ${DISK_AVAIL}GB free of ${DISK_TOTAL}GB"
    if [ "$CKPT_COUNT" -gt 0 ]; then
        echo "  Checkpoints: ${CKPT_COUNT} saved (${CKPT_SIZE_GB}GB total)"
    fi

    # Each checkpoint is ~13GB, need room for 4 during rotation (keep-last 3 + new)
    MIN_FREE=52  # 4 * 13GB
    WARN_FREE=65  # 5 * 13GB

    if [ "$DISK_AVAIL" -lt "$MIN_FREE" ]; then
        echo -e "  ${RED}FAIL: Only ${DISK_AVAIL}GB free (need ${MIN_FREE}GB+ for checkpoints)${NC}"
        echo "    Each checkpoint is ~13GB. With keep-last 3, you need room for 4 during rotation."
        echo "    Fix: Clean old checkpoints or increase disk size"
        echo "    Run: bash scripts/cleanup_local_checkpoints.sh"
        ((ERRORS++))
    elif [ "$DISK_AVAIL" -lt "$WARN_FREE" ]; then
        echo -e "  ${YELLOW}WARNING: ${DISK_AVAIL}GB free (recommend ${WARN_FREE}GB+)${NC}"
        echo "    Consider running cleanup daemon during training:"
        echo "    nohup bash scripts/cleanup_local_checkpoints.sh --daemon 5 > /tmp/cleanup.log 2>&1 &"
        ((WARNINGS++))
    else
        echo -e "  ${GREEN}${DISK_AVAIL}GB available (OK)${NC}"
    fi
fi

# 8. Check environment is installed
echo ""
echo "[8/8] Checking environment installation..."
if python3 -c "from verifiers import load_environment; load_environment('seconds-0/beautiful-soup-env')" 2>/dev/null; then
    echo -e "  ${GREEN}Environment installed (OK)${NC}"
else
    echo -e "  ${YELLOW}WARNING: Environment may not be installed${NC}"
    echo "    Fix: prime env install seconds-0/beautiful-soup-env"
    ((WARNINGS++))
fi

# Summary
echo ""
echo "================================================"
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All checks passed! Ready for training.${NC}"
    echo "================================================"
    echo ""
    echo "Start training with:"
    echo "  cd $PRIME_RL_DIR"
    echo "  nohup uv run rl @ configs/<your-config>.toml \\"
    echo "    --ckpt --ckpt.interval 5 --ckpt.keep-last 3 \\"
    echo "    > /root/training.log 2>&1 &"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}Passed with $WARNINGS warning(s). Training may work but fix warnings for reliability.${NC}"
    echo "================================================"
    exit 0
else
    echo -e "${RED}FAILED: $ERRORS error(s), $WARNINGS warning(s)${NC}"
    echo "================================================"
    echo ""
    echo "Fix the errors above before starting training."
    exit 1
fi
