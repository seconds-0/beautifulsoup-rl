#!/bin/bash
#
# Clean ALL training state for a fresh start
#
# This script cleans ALL directories that can cause stale state issues,
# including the commonly-missed run_default/checkpoints directory.
#
# Usage:
#   bash scripts/clean_training_state.sh
#   # OR on pod:
#   curl -sSL https://raw.githubusercontent.com/seconds-0/beautifulsoup-rl/main/scripts/clean_training_state.sh | bash
#

set -e

# Default paths (can be overridden)
PRIME_RL_DIR="${PRIME_RL_DIR:-/root/prime-rl-official}"
OUTPUTS_DIR="${PRIME_RL_DIR}/outputs"

echo "================================================"
echo "Prime-RL Training State Cleanup"
echo "================================================"
echo ""
echo "PRIME_RL_DIR: $PRIME_RL_DIR"
echo "OUTPUTS_DIR: $OUTPUTS_DIR"
echo ""

# 1. Kill any running training processes
echo "[1/4] Killing training processes..."
pkill -9 -f "prime_rl" 2>/dev/null || true
pkill -9 -f "uv.*rl" 2>/dev/null || true
pkill -9 -f "torchrun" 2>/dev/null || true
pkill -9 -f "vllm" 2>/dev/null || true
pkill -9 -f "VLLM" 2>/dev/null || true
sleep 3

# Verify GPU memory is cleared
echo "  Checking GPU memory..."
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$GPU_MEM" -gt 1000 ]; then
        echo "  WARNING: GPU 0 still using ${GPU_MEM}MB - may have stale processes"
        echo "  Waiting 5 more seconds..."
        sleep 5
    else
        echo "  GPU memory cleared"
    fi
fi

# 2. Clean checkpoint directories (BOTH locations!)
echo ""
echo "[2/4] Cleaning checkpoint directories..."

# Main checkpoints (where new checkpoints go)
if [ -d "$OUTPUTS_DIR/checkpoints" ]; then
    rm -rf "$OUTPUTS_DIR/checkpoints"/*
    echo "  Cleaned: outputs/checkpoints/"
fi

# run_default checkpoints (COMMONLY MISSED - causes trainer-orchestrator deadlock!)
if [ -d "$OUTPUTS_DIR/run_default/checkpoints" ]; then
    rm -rf "$OUTPUTS_DIR/run_default/checkpoints"/*
    echo "  Cleaned: outputs/run_default/checkpoints/"
fi

# 3. Clean communication directories
echo ""
echo "[3/4] Cleaning communication directories..."

# Rollouts (orchestrator -> trainer data)
if [ -d "$OUTPUTS_DIR/run_default/rollouts" ]; then
    rm -rf "$OUTPUTS_DIR/run_default/rollouts"/*
    echo "  Cleaned: outputs/run_default/rollouts/"
fi

# Broadcasts (weight updates)
if [ -d "$OUTPUTS_DIR/run_default/broadcasts" ]; then
    rm -rf "$OUTPUTS_DIR/run_default/broadcasts"/*
    echo "  Cleaned: outputs/run_default/broadcasts/"
fi

# Weights directory
if [ -d "$OUTPUTS_DIR/weights" ]; then
    rm -rf "$OUTPUTS_DIR/weights"/*
    echo "  Cleaned: outputs/weights/"
fi

# 4. Clean logs (optional but recommended for fresh start)
echo ""
echo "[4/4] Cleaning log directories..."

if [ -d "$OUTPUTS_DIR/logs" ]; then
    rm -rf "$OUTPUTS_DIR/logs"/*
    echo "  Cleaned: outputs/logs/"
fi

if [ -d "$OUTPUTS_DIR/run_default/logs" ]; then
    rm -rf "$OUTPUTS_DIR/run_default/logs"/*
    echo "  Cleaned: outputs/run_default/logs/"
fi

# Summary
echo ""
echo "================================================"
echo "Cleanup Complete!"
echo "================================================"
echo ""
echo "Directories cleaned:"
echo "  - outputs/checkpoints/"
echo "  - outputs/run_default/checkpoints/  (commonly missed!)"
echo "  - outputs/run_default/rollouts/"
echo "  - outputs/run_default/broadcasts/"
echo "  - outputs/weights/"
echo "  - outputs/logs/"
echo "  - outputs/run_default/logs/"
echo ""
echo "Next steps:"
echo "  1. Verify GPU memory is cleared: nvidia-smi"
echo "  2. Set environment: export PATH=\"/root/.local/bin:\$PATH\""
echo "  3. Start training with: uv run rl @ config.toml --ckpt --ckpt.interval 5 --ckpt.keep-last 3"
echo ""
