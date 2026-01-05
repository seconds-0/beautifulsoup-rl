#!/bin/bash
#
# Pod Setup Script for Prime-RL Training
#
# Run this script on every new GPU pod to set up the training environment.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/seconds-0/beautifulsoup-rl/main/scripts/pod_setup.sh | bash
#   # OR copy-paste and run manually
#
# Prerequisites:
#   - GPU pod with CUDA
#   - uv installed (pip install uv)
#   - WANDB_API_KEY environment variable set

set -e  # Exit on error

echo "================================================"
echo "Prime-RL Training Pod Setup"
echo "================================================"

# 1. Critical vLLM environment variables and system limits
# These prevent CUDA segfaults, multiprocessing issues, and FD exhaustion
echo ""
echo "[1/6] Setting vLLM environment variables and system limits..."
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "  VLLM_USE_V1=0 (disable V1 engine - has issues with LoRA)"
echo "  VLLM_WORKER_MULTIPROC_METHOD=spawn (prevent CUDA context inheritance)"

# Increase file descriptor limit (vLLM warns at 32000)
ulimit -n 65536 2>/dev/null || echo "  (could not increase ulimit, may cause issues)"
echo "  ulimit -n 65536 (prevent 'Too many open files' errors)"

# Add to .bashrc for persistence
cat >> ~/.bashrc << 'EOF'
# Prime-RL vLLM fixes
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ulimit -n 65536 2>/dev/null
EOF

# 2. WandB configuration
echo ""
echo "[2/5] Checking WandB configuration..."
if [ -z "$WANDB_API_KEY" ]; then
    echo "  ⚠️  WANDB_API_KEY not set!"
    echo "  Set it with: export WANDB_API_KEY=<your-key>"
    echo "  Or add to ~/.bashrc for persistence"
else
    echo "  ✓ WANDB_API_KEY is set"
    # Configure .netrc for wandb
    echo -e "machine api.wandb.ai\n  login user\n  password $WANDB_API_KEY" > ~/.netrc
    chmod 600 ~/.netrc
    echo "  ✓ Configured ~/.netrc for WandB"
fi

# 3. Clone and install prime-rl
echo ""
echo "[3/5] Setting up prime-rl..."
if [ -d "/root/prime-rl" ] || [ -d "$HOME/prime-rl" ]; then
    echo "  prime-rl directory already exists, skipping clone"
    cd ~/prime-rl || cd /root/prime-rl
else
    cd ~
    git clone https://github.com/PrimeIntellect-ai/prime-rl.git
    cd prime-rl
fi

echo "  Installing dependencies..."
uv sync --all-extras
echo "  ✓ prime-rl installed"

# 4. Install BeautifulSoup environment
echo ""
echo "[4/5] Installing BeautifulSoup RL environment..."
prime env install seconds-0/beautiful-soup-env
echo "  ✓ Environment installed"

# 5. Verify installation
echo ""
echo "[5/5] Verifying installation..."
uv run python -c "import beautiful_soup_env; print('  ✓ Environment imports successfully')"

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Copy your training config to /tmp/config.toml"
echo ""
echo "2. Start training with checkpointing:"
echo "   tmux new -s training"
echo "   cd ~/prime-rl"
echo "   uv run rl @ /tmp/config.toml --ckpt --ckpt.interval 5 --ckpt.keep-last 3 2>&1 | tee /tmp/training.log"
echo ""
echo "3. Monitor training:"
echo "   # In another terminal:"
echo "   tail -f /tmp/training.log"
echo "   # Check GPU usage:"
echo "   nvidia-smi"
echo ""
echo "Single-GPU memory checklist (already in config if using qwen2.5-7b-h100.toml):"
echo "  - [inference] gpu_memory_utilization = 0.50"
echo "  - [trainer.model.ac] freq = 1"
echo "  - [orchestrator.sampling] max_tokens = 2000"
echo "  - Never use fsdp_cpu_offload with LoRA!"
echo ""
