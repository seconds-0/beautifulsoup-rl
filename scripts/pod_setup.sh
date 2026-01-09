#!/bin/bash
#
# Pod Setup Script for Prime-RL Training with Resilient Checkpoint Sync
#
# Run this script on every new GPU pod to set up the training environment.
# Includes B2 checkpoint sync for crash recovery.
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/seconds-0/beautifulsoup-rl/main/scripts/pod_setup.sh | bash
#   # OR copy-paste and run manually
#
# Prerequisites:
#   - GPU pod with CUDA
#   - uv installed (pip install uv)
#
# Environment variables (set before running or pass via Vast.ai):
#   - WANDB_API_KEY: WandB API key
#   - B2_APPLICATION_KEY_ID: Backblaze B2 key ID
#   - B2_APPLICATION_KEY: Backblaze B2 application key
#   - RUN_ID: Training run identifier (default: "default")

set -e  # Exit on error

# Configuration
REPO_COMMIT="${REPO_COMMIT:-main}"  # Pin to specific commit for reproducibility
RUN_ID="${RUN_ID:-default}"

echo "================================================"
echo "Prime-RL Training Pod Setup"
echo "================================================"

# 1. Critical environment variables and PATH setup
# These prevent CUDA segfaults, multiprocessing issues, and FD exhaustion
echo ""
echo "[1/6] Setting environment variables and PATH..."

# Add ~/.local/bin to PATH (where uv is installed)
export PATH="/root/.local/bin:$PATH"
echo "  PATH updated: /root/.local/bin added"

# vLLM environment variables
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "  VLLM_USE_V1=0 (disable V1 engine - has issues with LoRA)"
echo "  VLLM_WORKER_MULTIPROC_METHOD=spawn (prevent CUDA context inheritance)"

# Increase file descriptor limit (vLLM warns at 32000)
ulimit -n 65536 2>/dev/null || echo "  (could not increase ulimit, may cause issues)"
echo "  ulimit -n 65536 (prevent 'Too many open files' errors)"

# Add to .bashrc for persistence (CRITICAL: includes PATH!)
cat >> ~/.bashrc << 'EOF'
# Prime-RL environment setup
export PATH="/root/.local/bin:$PATH"
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ulimit -n 65536 2>/dev/null
EOF

# 2. Install B2 CLI and jq for checkpoint sync
echo ""
echo "[2/8] Installing B2 CLI and jq..."
pip install --quiet b2
apt-get update -qq && apt-get install -y -qq jq > /dev/null 2>&1 || {
    echo "  (apt-get failed, trying pip for jq alternative)"
}
echo "  ✓ b2 CLI installed"

# Configure B2 if credentials are set
if [ -n "$B2_APPLICATION_KEY_ID" ] && [ -n "$B2_APPLICATION_KEY" ]; then
    echo "  Authorizing B2..."
    b2 authorize-account "$B2_APPLICATION_KEY_ID" "$B2_APPLICATION_KEY" > /dev/null 2>&1
    echo "  ✓ B2 authorized"
else
    echo "  ⚠️  B2 credentials not set (checkpoint sync disabled)"
fi

# 3. WandB configuration (CRITICAL - training will crash without this!)
echo ""
echo "[3/8] Checking WandB configuration..."
if [ -z "$WANDB_API_KEY" ]; then
    echo ""
    echo "  =============================================="
    echo "  ERROR: WANDB_API_KEY not set!"
    echo "  =============================================="
    echo ""
    echo "  Training WILL CRASH without a WandB API key."
    echo ""
    echo "  Fix options:"
    echo "    1. Set via environment:"
    echo "       export WANDB_API_KEY=<your-key>"
    echo ""
    echo "    2. Or create ~/.wandb_api_key file:"
    echo "       echo 'export WANDB_API_KEY=<your-key>' > ~/.wandb_api_key"
    echo "       source ~/.wandb_api_key"
    echo ""
    echo "    3. Get your key from: https://wandb.ai/authorize"
    echo ""
    WANDB_CONFIGURED=false
else
    echo "  ✓ WANDB_API_KEY is set"
    # Configure .netrc for wandb
    echo -e "machine api.wandb.ai\n  login user\n  password $WANDB_API_KEY" > ~/.netrc
    chmod 600 ~/.netrc
    echo "  ✓ Configured ~/.netrc for WandB"

    # Also save to ~/.wandb_api_key for easy sourcing
    echo "export WANDB_API_KEY=\"$WANDB_API_KEY\"" > ~/.wandb_api_key
    chmod 600 ~/.wandb_api_key
    echo "  ✓ Saved to ~/.wandb_api_key for persistence"

    # Add to .bashrc
    echo 'source ~/.wandb_api_key 2>/dev/null || true' >> ~/.bashrc
    WANDB_CONFIGURED=true
fi

# 4. Clone and install prime-rl
echo ""
echo "[4/8] Setting up prime-rl..."
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

# 5. Install BeautifulSoup environment
echo ""
echo "[5/8] Installing BeautifulSoup RL environment..."
prime env install seconds-0/beautiful-soup-env
echo "  ✓ Environment installed"

# 6. Download training scripts
echo ""
echo "[6/8] Downloading training scripts..."
SCRIPTS_BASE="https://raw.githubusercontent.com/seconds-0/beautifulsoup-rl/${REPO_COMMIT}/scripts"
mkdir -p ~/prime-rl/scripts

for script in checkpoint_sync.sh onstart.sh; do
    curl -sSL "$SCRIPTS_BASE/$script" -o ~/prime-rl/scripts/$script
    chmod +x ~/prime-rl/scripts/$script
    echo "  ✓ Downloaded $script"
done

# Create symlink at /root/scripts for backwards compatibility with instance onstart commands
mkdir -p /root/scripts
ln -sf ~/prime-rl/scripts/onstart.sh /root/scripts/onstart.sh
echo "  ✓ Created /root/scripts/onstart.sh symlink"

# 7. Set up checkpoint sync (if B2 configured)
echo ""
echo "[7/8] Setting up checkpoint sync..."
if [ -n "$B2_APPLICATION_KEY_ID" ]; then
    # Create checkpoint directory
    mkdir -p /root/checkpoints

    # Save environment to .env for systemd
    cat > /root/.env << EOF
RUN_ID=$RUN_ID
B2_APPLICATION_KEY_ID=$B2_APPLICATION_KEY_ID
B2_APPLICATION_KEY=$B2_APPLICATION_KEY
B2_BUCKET=beautifulsoup-rl
WANDB_API_KEY=$WANDB_API_KEY
WANDB_RUN_ID=${WANDB_RUN_ID:-}
VLLM_USE_V1=0
VLLM_WORKER_MULTIPROC_METHOD=spawn
EOF
    chmod 600 /root/.env
    echo "  ✓ Environment saved to /root/.env"

    # Start checkpoint sync daemon in background
    nohup ~/prime-rl/scripts/checkpoint_sync.sh >> /tmp/checkpoint_sync.log 2>&1 &
    echo "  ✓ Checkpoint sync daemon started"
else
    echo "  ⚠️  Skipping (B2 not configured)"
fi

# 8. Verify installation
echo ""
echo "[8/8] Verifying installation..."
uv run python -c "import beautiful_soup_env; print('  ✓ Environment imports successfully')"
command -v b2 >/dev/null && echo "  ✓ b2 CLI available"
command -v jq >/dev/null && echo "  ✓ jq available"

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Run ID: $RUN_ID"
echo "Checkpoint sync: $([ -n "$B2_APPLICATION_KEY_ID" ] && echo "ENABLED" || echo "DISABLED")"
echo ""
echo "Next steps:"
echo ""
echo "1. Copy your training config:"
echo "   # Upload config to B2 for persistence across restarts"
echo "   b2 file upload beautifulsoup-rl /path/to/config.toml $RUN_ID/config.toml"
echo "   b2 file download b2://beautifulsoup-rl/$RUN_ID/config.toml /root/config.toml"
echo ""
echo "2. Start training with checkpointing:"
echo "   tmux new -s training"
echo "   cd ~/prime-rl"
echo "   export WANDB_RUN_ID=\$(uuidgen)  # Generate unique run ID"
echo "   uv run rl @ /root/config.toml --ckpt --ckpt.interval 5 --ckpt.keep-last 3 2>&1 | tee /tmp/training.log"
echo ""
echo "3. Monitor training:"
echo "   # Watch logs:"
echo "   tail -f /tmp/training.log"
echo "   # Check checkpoint sync:"
echo "   tail -f /tmp/checkpoint_sync.log"
echo "   # Check GPU usage:"
echo "   nvidia-smi"
echo ""
echo "4. If training crashes, it will auto-resume from checkpoint:"
echo "   # Checkpoints are synced to B2 every 5 minutes"
echo "   # On restart, onstart.sh will pull the latest checkpoint"
echo ""
echo "Single-GPU memory checklist:"
echo "  - [inference] gpu_memory_utilization = 0.50"
echo "  - [trainer.model.ac] freq = 1"
echo "  - [orchestrator.sampling] max_tokens = 2000"
echo "  - Never use fsdp_cpu_offload with LoRA!"
echo ""
