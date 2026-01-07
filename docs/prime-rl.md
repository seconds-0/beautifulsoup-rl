# Prime-RL Documentation

Scraped from docs.primeintellect.ai/prime-rl on 2026-01-04 for offline reference.

---

## Table of Contents

1. [Overview](#overview)
2. [Entrypoints](#entrypoints)
3. [Configuration](#configuration)
4. [Checkpointing](#checkpointing)
5. [Environments](#environments)
6. [Async Training](#async-training)
7. [Logging](#logging)
8. [Benchmarking](#benchmarking)
9. [Deployment](#deployment)
10. [Troubleshooting](#troubleshooting)
11. [Multi-Run Training](#multi-run-training)

---

## Overview

PRIME-RL is a reinforcement learning framework for training language models. The system is organized into:

- **Entrypoints** - Main components: orchestrator, trainer, and inference
- **Configs** - TOML files, CLI arguments, environment variables
- **Environments** - Integration with Environments Hub (verifiers)
- **Async Training** - Off-policy training mechanisms
- **Logging** - loguru, torchrun, Weights & Biases
- **Runs** - Multi-run training with concurrent LoRA adapters
- **Checkpointing** - Save/resume training state
- **Benchmarking** - Performance measurement (MFU, throughput)
- **Deployment** - Single-GPU, multi-GPU, multi-node, Kubernetes

---

## Entrypoints

### Core Architecture Components

**Orchestrator**: Lightweight CPU process managing data flow and scheduling. Collects rollouts from inference server, assembles batched packets, distributes to trainer. Relays updated weights back to inference. Uses verifiers environments for multi-turn rollout generation and scoring.

**Trainer**: Produces updated policy models from rollouts using FSDP2 backend. Supports any HuggingFace-compatible model. Training objectives: GRPO, GSPO, OPO, RLOO, CISPO. Leverages PyTorch's native parallelism (inspired by torchtitan).

**Inference**: Standard OpenAI-compatible server with vLLM backend. Custom endpoints:
- `update_weights` - Reload model weights from disk checkpoints
- `reload_weights` - Reset to base model
- Supports native data parallelism across multiple nodes

### CLI Entrypoints

**RL Training:**
```bash
uv run rl --trainer @ path/to/train.toml --orchestrator @ path/to/orch.toml --inference @ path/to/infer.toml
```

**SFT (Supervised Fine-Tuning):**
```bash
# Single GPU
uv run sft ...

# Multiple GPUs
uv run torchrun --nproc-per-node 8 src/prime_rl/trainer/sft/train.py ...
```

**Evals:**
```bash
uv run eval @ configs/debug/eval/single_env.toml
uv run eval @ configs/debug/eval/multi_env.toml
uv run eval @ configs/debug/eval/local_model.toml --client.base-url http://localhost:8000/v1
```

**Synthetic Data Generation:**
```bash
uv run synthesize
```
Supports single-turn, multi-turn, and tool-calling environments.

---

## Configuration

Prime-RL uses `pydantic-settings` with four configuration sources (in order of precedence):

### 1. Command-Line Arguments (Highest Priority)
```bash
--model.name <model-name>
```

### 2. Config Files
Use `@` prefix with space:
```bash
uv run inference @ path/to/config.toml
```

Example TOML:
```toml
[model]
name = "Qwen/Qwen3-8B"
```

### 3. Environment Variables
Prefix with `PRIME_`, use `__` for nesting:
```bash
export PRIME_MODEL__NAME=Qwen/Qwen3-4B
```

### 4. Default Values (Lowest Priority)

### Precedence Example
```bash
PRIME_MODEL__NAME=Qwen/Qwen3-4B uv run ... @ qwen8b.toml @ qwen14b.toml --model.name Qwen/Qwen3-32B
```

Resolution order:
1. CLI argument: `Qwen/Qwen3-32B` <- **Used**
2. Second config file: `Qwen/Qwen-14B`
3. First config file: `Qwen/Qwen3-8B`
4. Environment variable: `Qwen/Qwen3-4B`
5. Default value: `Qwen/Qwen3-0.6B`

### Recommended Usage
- **Config files**: Reproducible experiments
- **CLI arguments**: Override values for experiment variants
- **Environment variables**: Production deployments

---

## Checkpointing

**IMPORTANT: Checkpointing is DISABLED by default to save disk space!**

### Overview

Checkpointing is non-standard due to separation of trainer/orchestrator and natural asynchrony:

- **SFT+RL Trainer**: Saves FSDP model shards (via DCP), optimizer/scheduler state, progress metrics
- **Orchestrator**: Checkpoints progress (step, tokens, samples, problems)
- **Inference**: Stateless - no checkpointing needed. Orchestrator reloads correct weights on restart.

### Configuration

Default directory: `checkpoints/step_{step}`

**CLI Flags:**

| Flag | Purpose |
|------|---------|
| `--ckpt` | Enable checkpointing (OFF by default!) |
| `--ckpt.interval N` | Save every N steps |
| `--ckpt.save-async` | Non-blocking checkpoint saves |
| `--ckpt.keep-last N` | Keep only last N checkpoints |
| `--ckpt.keep-interval N` | Preserve checkpoints at every N steps permanently |
| `--ckpt.resume-step N` | Resume from specific step |

### SFT Example

Split 40-step training into two 20-step runs:

```bash
# Run 1 - Train and checkpoint
uv run sft ... --max-steps 20 --ckpt

# Run 2 - Resume from step 20
uv run sft ... --max-steps 40 --ckpt.resume-step 20
```

### RL Example

Split 20-step RL training into two 10-step runs:

```bash
# Start inference server (persists across restarts)
uv run inference ...

# Run 1 - Train and checkpoint
uv run rl \
  --trainer @ path/to/train.toml \
  --orchestrator @ path/to/orch.toml \
  --max-steps 10 \
  --ckpt

# Run 2 - Resume to complete remaining steps
uv run rl \
  --trainer @ path/to/train.toml \
  --orchestrator @ path/to/orch.toml \
  --max-steps 20 \
  --ckpt.resume-step 10
```

The inference server automatically receives the correct checkpoint when orchestrator resumes.

---

## Environments

PRIME-RL trains and evaluates in any environments from the `verifiers` library.

### Installation Methods

**Check installation options:**
```bash
prime env info <owner>/<name>
```

**Temporary installation:**
```bash
prime env install <owner>/<name>
# Alternative:
uv pip install <name> --extra-index-url https://hub.primeintellect.ai/<owner>/simple/
```

**Local environment setup:**
```bash
uv pip install -e path/to/env
```

**Verification:**
```bash
uv run python -c "import <name>"
```

---

## Async Training

### Overview

PRIME-RL implements asynchronous off-policy training (vs traditional synchronous on-policy). The system allows inference to generate rollouts from a stale policy up to k steps ahead of the trainer.

Default k=2 to accommodate weight broadcasts for decentralized training.

### Off-Policy Benefits

With k=1 and equal trainer/inference timing: no idle time on either component.
Default k=2: overlap capacity for distributed weight synchronization.

### Loss Objective

Token-level variant of AIPO training objective from Llama-RL, omitting entropy and KL loss terms:
1. Sample N prompts
2. Generate G rollouts per prompt
3. Assign verifier scores to each rollout
4. Optimize using importance sampling with clipping ratio δ

### Step Semantics

Global training steps (n=1,2,3,...) tag artifacts:
- **Trainer**: Produces policy πₙ from rollouts (xₙ, yₙ)
- **Inference**: Produces rollouts from policy π_max(0,n-k)

Off-policy gap bounded by k steps, enabling efficient asynchronous execution.

---

## Logging

### Loguru

Global logger instance. Each entrypoint calls `setup_logger` once at startup.
Logs stored in `{output_dir}/logs`.

For RL training, streaming logs into tmux panes recommended (via `tmux.sh` script).

### Torchrun for Multi-Node

All ranks log simultaneously. Filter to master rank only:
```bash
uv run torchrun \
  --local-ranks-filter 0 \
  --nproc-per-node 8 \
  ...
```

Redirect console output:
```bash
uv run torchrun \
  --log-dir outputs/torchrun \
  --redirects 3 \
  --tee 3 \
  ...
```

Logs written to `outputs/torchrun/{rdzv_id}/attempt_0/{rank}/{stdout,stderr}.log`

### Weights & Biases Integration

**SFT Training:**
```bash
uv run sft ... --wandb
uv run sft ... --wandb.project my-project --wandb.name my-run
```

**RL Training:**
Both trainer and orchestrator log as separate W&B runs:
```bash
uv run rl ... --wandb
```

Trainer run appends `-trainer`, orchestrator appends `-orchestrator`.

Configure sample/distribution logging:
```bash
uv run rl ... \
  --no-trainer.wandb.log-extras.distributions \
  --orchestrator.wandb.log-extras.interval 50
```

**Setup:**
```bash
uv run wandb login
# or
export WANDB_API_KEY=...
```

---

## Benchmarking

Use `--bench` flag to benchmark performance (MFU, throughput).

### SFT Benchmarking

```bash
# Default fake data
uv run sft ... --data.type fake --bench

# Custom batch configurations
uv run sft ... --data.seq-len 4096 --data.batch-size 64 --data.micro-batch-size 2 --bench
```

Variable-length fake datasets more closely simulate real data.

### RL Training Benchmarking

**Trainer:** Fake data loader benchmarking available. Cannot benchmark against real data in isolation.

**Inference:** Run server separately, then orchestrator with `--bench` flag.

**Combined:**
```bash
uv run rl --trainer @ path/to/train.toml --orchestrator @ path/to/orch.toml --inference @ path/to/infer.toml --bench
```

---

## Deployment

### SFT Deployment

**Single-GPU:**
```bash
uv run sft ...
```

**Multi-GPU:**
```bash
uv run torchrun --nproc-per-node 8 --local-rank-filter 0 ...
```

**Multi-Node:**
Configure nodes via environment variables:
- `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`
- `MASTER_ADDR`, `MASTER_PORT`

Launch head node with `--node-rank 0`, subsequent nodes with `--node-rank 1`, etc.

**SLURM:** TBD

### Inference Deployment

vLLM multi-node data parallelism:
- Set `DATA_PARALLEL_ADDRESS` to head node IP
- Set `DATA_PARALLEL_RPC_PORT`
- Different nodes get distinct `--data-parallel-start-rank`
- Secondary nodes in headless mode

### RL Deployment

**Single-GPU:**
Run inference with reduced GPU memory (`--gpu-memory-utilization 0.5`) alongside trainer on same device.

**Multi-GPU:**
Deploy inference across GPUs 0-5, trainer on GPUs 6-7:
```bash
--inference-gpu-ids 0,1,2,3,4,5 --trainer-gpu-ids 6,7
```

**Parallel Experiments:**
Split GPUs, assign unique server ports and orchestrator endpoints.

**Multi-Node:**
Requires shared filesystem. Set:
- `OUTPUT_DIR`
- `INFERENCE_SERVER_IP`
- `INFERENCE_SERVER_API_KEY`

Launch single orchestrator with trainer distributed via `torchrun`.

**Kubernetes:**
Helm chart manages orchestrator, trainer, inference with automatic GPU scheduling.

---

## Troubleshooting

### API Timeout Issues

**Problem:** API connections timing out.

**Solution:** Increase file descriptor limit:
```bash
ulimit -n 32000
```

### CUDA Out of Memory

**Problem:** GPU OOM during training.

**Solutions:**
- Enable activation checkpointing: `--model.ac`
- Reduce micro batch size: `--data.micro-batch-size`
- Decrease sequence length: `--data.seq-len`
- Try context parallelism (experimental): `--model.cp`

### TOML Configuration Issues

**Problem:** Config file not recognized.

**Solution:** Ensure proper spacing:
```bash
# Correct
uv run ... @ path/to/config.toml

# Wrong
uv run ... @path/to/config.toml
```

Verify TOML matches required schema. Pydantic errors indicate mismatches.

---

## Multi-Run Training

### Core Concept

The `Runs` object is a singleton managing multiple concurrent training runs within a single trainer process. When `max_concurrent_runs > 1`, trainer executes multiple runs in parallel.

### Key Responsibilities

- **Discovery**: Scans for `run_*` directories, loads configurations
- **Mapping**: Bidirectional run ID <-> index mapping
- **Progress Tracking**: Per-run metrics (step, tokens, samples)
- **Synchronization**: Consistency across distributed ranks
- **Hooks**: Lazy initialization of per-run resources
- **LoRA Management**: Multi-adapter parameter access
- **State Access**: Per-run parameters and state dictionaries

### Per-Run Isolation

Each concurrent run has:
- Individual LoRA adapter weights
- Separate optimizer and scheduler
- Independent training progress
- Unique orchestrator configuration

### Design Philosophy

Enables efficient multi-tenant training: single trainer serves multiple experiments with independent adapter weights, optimizers, and learning rate schedules.

---

## Quick Reference: Common Commands

### Start RL Training
```bash
uv run rl @ config.toml --wandb
```

### With Checkpointing
```bash
uv run rl @ config.toml --ckpt --ckpt.interval 10 --ckpt.keep-last 3
```

### Resume from Checkpoint
```bash
uv run rl @ config.toml --ckpt.resume-step 50 --max-steps 100
```

### Run Evaluation
```bash
uv run eval @ eval_config.toml
```

### Install Environment
```bash
prime env install <owner>/<name>
```

---

## Quick Reference: Single-GPU RL Training

For running trainer + vLLM inference on one GPU (e.g., H100 80GB):

```toml
inference_gpu_ids = [0]
trainer_gpu_ids = [0]

[inference]
gpu_memory_utilization = 0.50  # Leave room for trainer!

[trainer.model.ac]
freq = 1  # Activation checkpointing

[orchestrator]
batch_size = 8
rollouts_per_example = 2

[orchestrator.sampling]
max_tokens = 2000  # Must fit within max_model_len - input_tokens
```

Launch:
```bash
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
uv run rl @ config.toml --ckpt --ckpt.interval 5 --ckpt.keep-last 3
```
