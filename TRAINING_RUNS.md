# Training Runs Log

Track all RL training experiments for BeautifulSoup environment.

## Active Runs

### Run: bs4-rl-qwen3-8b-2xh100-v4-resilient (2026-01-06) - RUNNING ✅

- **Model**: Qwen/Qwen3-8B (8.2B params)
- **Config**: /root/config.toml (2x H100 Prime pod)
- **Pod**: Prime Intellect 2x H100 80GB (86.38.238.54:1234)
- **Status**: RUNNING ✅
- **W&B Project**: beautiful-soup-env
- **Step Time**: ~3-5 minutes
- **Current Step**: 80 (as of 09:20 UTC on 2026-01-06)
- **Rewards**: Training progressing well

#### Incidents and Recovery

**1. vLLM Crash After Step 6 (03:00 UTC)**
- Inference server on GPU 0 died silently
- GPU 0 showed 0 MiB memory
- Trainer stuck waiting for batches

**Recovery:**
```bash
# Kill all processes
pkill -9 python

# Restart from checkpoint step 5
uv run rl @ /root/config.toml --ckpt --ckpt.resume-step 5 --ckpt.interval 5 --ckpt.keep-last 2
```

Training resumed successfully from step 5 at 03:15 UTC.

**2. Disk Space Management (RESOLVED)**
- Checkpoints are ~47GB each (31GB checkpoint + 16GB weights)
- `--ckpt.keep-last 2` does NOT auto-delete old checkpoints

**Solution deployed (2026-01-06 09:22 UTC):**
1. **Disk cleanup daemon** (`/root/disk_cleanup.sh`): Automatically deletes oldest checkpoint when disk > 85%
2. **B2 sync daemon** (`/root/periodic_sync.sh`): Syncs latest checkpoint to Backblaze B2 every 10 min
3. **Updated `scripts/checkpoint_sync.sh`**: Now cleans up old local checkpoints after successful B2 sync

**Daemons running:**
```bash
# Verify with:
ps aux | grep -E "(cleanup|sync)" | grep -v grep
```

#### Config (v4 - Resilient)

```toml
inference_gpu_ids = [0]
trainer_gpu_ids = [1]
max_steps = 1000

[model]
name = "Qwen/Qwen3-8B"

[wandb]
project = "beautiful-soup-env"
name = "bs4-rl-qwen3-8b-2xh100-v4-resilient"

[trainer.optim]
lr = 1e-5
weight_decay = 0.0

[trainer.model]
seq_len = 4096

[trainer.model.lora]
rank = 8
alpha = 32
dropout = 0.0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

[orchestrator]
batch_size = 128
rollouts_per_example = 8
seq_len = 4096
oversampling_factor = 2.0
lora_name = "qwen3-8b-bs4-lora"

[orchestrator.sampling]
max_tokens = 4096
temperature = 0.7

[orchestrator.buffer]
online_difficulty_filtering = true

[[orchestrator.env]]
id = "seconds-0/beautiful-soup-env"

[orchestrator.env.args]
split = "train"
mode = "tiered"
difficulty = "mixed"
seed = 42
executor_backend = "local"
network_access = true
timeout_s = 30.0
max_output_chars = 10000

[inference]
gpu_memory_utilization = 0.90

[inference.model]
enable_auto_tool_choice = true
tool_call_parser = "hermes"
enforce_eager = true
```

#### Lessons Learned

1. **vLLM can crash silently** - Monitor GPU memory, not just process count
2. **Checkpoint cleanup doesn't work** - `--ckpt.keep-last N` doesn't delete old checkpoints
3. **47GB per checkpoint** - Plan for manual cleanup every 5 steps with 65GB free disk
4. **Resume works well** - Training successfully resumes from checkpoint with full continuity

---

### Run: bs4-rl-qwen3-8b-2xh100-v3 (2026-01-05) - RUNNING ✅

- **Model**: Qwen/Qwen3-8B (8.2B params)
- **Config**: configs/prime-rl/qwen3-8b-2xh100.toml:v3
- **Pod**: 2x H100 80GB (bs4-llama-2xh100)
- **Status**: RUNNING ✅
- **W&B Run**: https://wandb.ai/seconds-0-domus-magna-inc/beautiful-soup-env/runs/r1hgf2ie
- **Step Time**: ~3-4 minutes (down from 27 min with v2!)
- **Speedup**: **7-8x** achieved

#### v3 Fix: Local Executor (CRITICAL!)

**Root cause found:** Training was LATENCY-BOUND, not throughput-bound!

| Setting | v2 | v3 | Impact |
|---------|----|----|--------|
| `executor_backend` | `"prime"` | `"local"` | **10-15x speedup** |
| `rollouts_per_example` | 4 | 8 | Restored (local is fast) |
| `oversampling_factor` | 1.0 | 2.0 | Restored (local is fast) |

#### Step Time Comparison

| Config | Step Time | Per-Rollout | Bottleneck |
|--------|-----------|-------------|------------|
| v2 (prime executor) | ~27 min | 3.2s | Network latency |
| **v3 (local executor)** | **~3-4 min** | ~0.2s | GPU-bound |

#### Incident: Inference Server Crash (Step 5)

**What happened:**
- Training crashed after step 5 completed
- Inference server (GPU 0) died silently - 0% utilization, 0 memory
- Orchestrator kept retrying but got `APIConnectionError('Connection error.')` for all 128 rollouts
- Training appeared "stuck" for ~3 hours with no progress

**Diagnosis:**
- Original logs were overwritten when training was restarted
- Found warning in logs: `Found ulimit of 32000 and failed to automatically increase`
- vLLM recommends 65536 file descriptors; 32000 may cause "Too many open files" errors
- Most likely causes: FD exhaustion, GPU memory fragmentation, or CUDA context corruption

**Recovery:**
```bash
# Kill stuck processes
ps aux | grep -E "(rl|vllm|ray)" | grep -v grep | awk '{print $2}' | xargs -r kill -9

# Resume from checkpoint
uv run rl @ /tmp/config.toml --ckpt --ckpt.resume-step 5 --ckpt.interval 5 --ckpt.keep-last 3
```

**Fix Applied:**
- Added `ulimit -n 65536` to `scripts/pod_setup.sh`
- Added to `.bashrc` for persistence

---

### Run: bs4-rl-qwen3-8b-2xh100-v2 (2026-01-05) - POSTMORTEM ⚠️

- **Model**: Qwen/Qwen3-8B (8.2B params)
- **Config**: configs/prime-rl/qwen3-8b-2xh100.toml:v2
- **Pod**: 2x H100 80GB
- **Status**: POSTMORTEM ⚠️ (optimizations failed, root cause found)
- **Step Time**: ~27 min (only 10% improvement from v1!)
- **Expected**: ~4-6 min (5-8x speedup)
- **Actual**: ~27 min (10% speedup)

#### v2 Postmortem: Wrong Bottleneck Identified

**What we tried:**
- Reduced max_tokens (10k → 4k) - No impact
- Reduced rollouts (8 → 4) - Made per-rollout time WORSE
- Added prefix caching - Not supported by prime-rl
- Added max_num_seqs, max_num_batched_tokens - Not supported

**Why it failed:**
Multi-turn tool-calling RL is **LATENCY-BOUND**, not throughput-bound!

- Remote sandbox (`executor_backend = "prime"`) adds ~1.5s per tool call
- 1024 rollouts × 1.5s = ~25 minutes of waiting (GPUs idle!)
- Reducing rollouts to 512 didn't reduce latency, just did less work

**The fix:** Switch to `executor_backend = "local"` (see v3)

#### Lessons Learned

1. **Profile before optimizing** - We optimized inference params, but tool execution was the bottleneck
2. **Latency vs throughput** - Multi-turn interactions are latency-bound
3. **Reducing work doesn't reduce fixed latency** - 512 rollouts took same time as 1024
4. **Question defaults** - `executor_backend="prime"` is safe but slow

---

### Run: bs4-rl-qwen2.5-7b-h100-v6 (2026-01-05) - RUNNING ✅

- **Model**: Qwen/Qwen2.5-7B-Instruct (7B params)
- **Config**: /tmp/config.toml (remote) - see below
- **Pod**: 1x H100 80GB SXM5 (DataCrunch spot, $0.99/hr)
- **Pod ID**: fd6a88cad0f04019841355376d088f5b
- **Start**: 2026-01-05 00:16 UTC
- **Status**: RUNNING ✅
- **W&B Run**: https://wandb.ai/seconds-0-domus-magna-inc/beautiful-soup-env/runs/189rkyj5
- **GPU Usage**: 72GB/81GB (89%) with activation checkpointing
- **Baseline**: N/A (OpenRouter doesn't support tool calling for Qwen2.5-7B)
- **Env Version**: 0.1.1 (fixed reward function crash on malformed JSON)

---

### Run: bs4-rl-qwen2.5-7b-h100-v5 (2026-01-05) - ABANDONED ⚠️

- **Model**: Qwen/Qwen2.5-7B-Instruct (7B params)
- **Config**: /tmp/config.toml (remote)
- **Pod**: 1x H100 80GB SXM5 (DataCrunch spot, $0.99/hr)
- **Pod ID**: fd6a88cad0f04019841355376d088f5b
- **Start**: 2026-01-04 23:50 UTC
- **End**: 2026-01-05 00:14 UTC (killed for v6 restart)
- **Status**: ABANDONED ⚠️ (reward function bug caused errors, restarted with v0.1.1)
- **W&B Run**: https://wandb.ai/seconds-0-domus-magna-inc/beautiful-soup-env/runs/t6agrtyb
- **GPU Usage**: 74GB/81GB (91%) with activation checkpointing
- **Baseline**: N/A (OpenRouter doesn't support tool calling for Qwen2.5-7B)
- **Progress**: 1/1000 steps completed (reward 0.0238, 287s, 50 tokens/s)
- **Issue**: Reward function crashed on malformed JSON: `'list' object has no attribute 'get'`
- **Fix**: Published env v0.1.1 with guard against malformed JSON arguments

#### Working Config (H100 Single GPU - Final v5)

```toml
# Key settings for single H100 with shared trainer/inference
# v5: Fixed max_tokens to fit within max_model_len
inference_gpu_ids = [0]
trainer_gpu_ids = [0]
max_steps = 1000

[model]
name = "Qwen/Qwen2.5-7B-Instruct"

[trainer.model]
seq_len = 2048

[trainer.model.ac]
freq = 1                  # CRITICAL: Full activation checkpointing saves ~20GB

[trainer.model.lora]
rank = 8
alpha = 32

[orchestrator]
batch_size = 8            # Minimal for single GPU
rollouts_per_example = 2  # Minimal for single GPU
seq_len = 2048

[orchestrator.sampling]
max_tokens = 2000         # CRITICAL: Must fit within max_model_len - input_tokens

[inference]
gpu_memory_utilization = 0.50

[inference.model]
enable_auto_tool_choice = true
tool_call_parser = "hermes"
enforce_eager = true
max_model_len = 4096
```

#### Memory Optimization Journey

1. **First attempt (OOM)**: gpu_memory_utilization = 0.45, no checkpointing → 80GB total, OOM
2. **Second attempt (device mismatch)**: fsdp_cpu_offload = true → RuntimeError: device mismatch CPU/CUDA
3. **Third attempt (config error)**: ac = "selective" → ValidationError: expects config object
4. **Fourth attempt (v4 - stuck)**: ac = {freq = 1} → 72GB used, but max_tokens=4000 > available context
5. **Fifth attempt (v5 - SUCCESS)**: max_tokens = 2000 → Training running!

#### Model Selection (2026-01-05)

**Winner: Qwen/Qwen2.5-7B-Instruct** (Ministral failed to load)

| Model | Tool Calls | Baseline | vLLM | Status |
|-------|------------|----------|------|--------|
| Ministral-3-8B | ✅ 1148 | 63.2% | ❌ KeyError | transformers doesn't support |
| **Qwen2.5-7B** | N/A | N/A | ✅ Works | **SELECTED** |
| Qwen3-8B | ❌ 0 calls | 39.6% | ? | Doesn't use tools |
| gpt-oss-20b | N/A | 49.2% | ❌ vLLM bug | Blocked |

**Key fixes for single-GPU training:**
1. `[inference] gpu_memory_utilization = 0.50` - leave room for trainer
2. `[trainer.model.ac] freq = 1` - activation checkpointing (NOT fsdp_cpu_offload)
3. Reduced batch_size (8) and rollouts_per_example (2)

---

### Run: bs4-rl-qwen2.5-7b-1000steps (2026-01-04) - ABANDONED

- **Model**: Qwen/Qwen2.5-7B-Instruct (7B params)
- **Config**: configs/prime-rl/beautiful-soup-env.toml
- **Pod**: bs4-rl-training (2x A6000 48GB)
- **Start**: 2026-01-04 21:03 UTC
- **Status**: ABANDONED ❌ (switched to Ministral-8B)
- **W&B Project**: beautiful-soup-env
- **W&B Run**: https://wandb.ai/seconds-0-domus-magna-inc/beautiful-soup-env/runs/aochk8k1
- **Baseline**: Could not benchmark (OpenRouter doesn't support tool calling for this model)

#### Config (v8 - Qwen2.5-7B)
```toml
max_steps = 1000
inference_gpu_ids = [0]
trainer_gpu_ids = [1]

[model]
name = "Qwen/Qwen2.5-7B-Instruct"  # Widely supported, good tool calling

[trainer.model.experimental.lora]
rank = 8
alpha = 32

[orchestrator]
batch_size = 128        # 7B model allows larger batches
rollouts_per_example = 8
seq_len = 8192
```

#### Model Selection Journey
1. **gpt-oss-20b** - vLLM weight reload bug (`default_weight_loader() got unexpected keyword argument 'weight_name'`)
2. **qwen/qwen3-vl-8b-instruct** - VL (Vision-Language) model, not supported by AutoModelForCausalLM
3. **mistralai/mistral-small-3.2-24b-instruct** - Not on HuggingFace Hub
4. **Qwen/Qwen2.5-7B-Instruct** - ✅ Current choice (7B, text-only, widely supported)

#### Issues Encountered

1. **Config format errors** (attempt 1):
   - `seq_len` must be under `[orchestrator]`, not top-level
   - `top_p` not allowed in `[orchestrator.sampling]`
   - `[orchestrator.buffer]` uses `online_difficulty_filtering = true`, not `type = "online-difficulty"`

2. **WandB API key missing** (attempt 2):
   - Error: `No API key configured`
   - Fix: Set `offline = true` in config, or configure `/root/.netrc`

3. **CUDA OOM with gpt-oss-20b** (attempt 3):
   - Error: `CUDA out of memory. Tried to allocate 2.16 GiB`
   - Root cause: 20B model weights (~40GB BF16) + inference server using 42.83GB on GPU 0
   - Fix: Enable `fsdp_cpu_offload = true`, reduce batch_size to 32, seq_len to 8192

4. **vLLM weight reload bug with gpt-oss-20b** (attempt 4):
   - Error: `TypeError: default_weight_loader() got an unexpected keyword argument 'weight_name'`
   - Root cause: `gpt-oss-20b` model loader incompatible with vLLM `/reload_weights` endpoint
   - Fix: Switch to different model

5. **qwen3-vl-8b is Vision-Language model** (attempt 5):
   - Error: `Unrecognized configuration class Qwen3VLConfig for AutoModelForCausalLM`
   - Root cause: VL models require different model class (AutoModelForVision2Seq)
   - Fix: Switch to text-only model

6. **Mistral model not on HuggingFace** (attempt 6):
   - Error: `mistralai/mistral-small-3.2-24b-instruct is not a valid model identifier`
   - Root cause: Model only available via Mistral API, not on HF Hub
   - Fix: Switch to Qwen/Qwen2.5-7B-Instruct

7. **Final model selection** (attempt 7-8):
   - Qwen/Qwen2.5-7B-Instruct: 7B params, text-only, widely supported, good tool calling
   - No CPU offloading needed on 2x A6000 (7B fits comfortably)

8. **vLLM V1 engine CUDA segfault** (attempt 9-10):
   - Error: `Segfault encountered` in `THCPModule_initExtension` / `cuCtxGetDevice`
   - Root cause: vLLM V1 engine spawns child processes that inherit invalid CUDA context
   - Fix: Set environment variables before launching:
     ```bash
     export VLLM_USE_V1=0
     export VLLM_WORKER_MULTIPROC_METHOD=spawn
     ```
   - Also added `enforce_eager = true` in config to disable CUDA graphs

#### WandB Setup on Pod
```bash
# Configure API key
echo -e 'machine api.wandb.ai\n  login user\n  password YOUR_KEY' > /root/.netrc
chmod 600 /root/.netrc
```

---

## Completed Runs

| Run ID | Model | Start | Duration | Final Reward | Notes |
|--------|-------|-------|----------|--------------|-------|
| *none yet* | | | | | |

---

## Run Template

```markdown
### Run: [run-id]
- **Model**: openai/gpt-oss-20b
- **Config**: configs/prime-rl/beautiful-soup-env.toml
- **Start**: YYYY-MM-DD HH:MM
- **End**: YYYY-MM-DD HH:MM
- **Duration**: X hours
- **W&B Link**: https://wandb.ai/...

#### Metrics
- Baseline: X%
- Final: Y%
- Improvement: +Z%

#### Config Changes
- max_steps: 50 (smoke test)
- batch_size: 256
- rollouts_per_example: 8

#### Notes
- Any issues encountered
- Observations
```

---

## Configuration Reference

### Current Config: `configs/prime-rl/beautiful-soup-env.toml`

Key settings:
- **Model**: `openai/gpt-oss-20b` (21B params, 3.6B active, Apache 2.0)
- **LoRA**: rank=8, alpha=32
- **Batch**: 256 rollouts, 8 per example
- **Context**: seq_len=4096, max_tokens=10000
- **Environment**: seconds-0/beautiful-soup-env

### Launch Command

```bash
# On GPU pod with 2+ GPUs:
uv run rl @ configs/prime-rl/beautiful-soup-env.toml \
  --wandb.project beautiful-soup-env \
  --wandb.name bs4-rl-gpt-oss-20b-run1
```

### Monitoring

```bash
# tmux session for wandb
tmux attach -t wandb-monitor

# View W&B dashboard
open https://wandb.ai/YOUR_USERNAME/beautiful-soup-env
```

---

## Model Baselines (Pre-Training)

| Model | Baseline (n=680) | Cost | Training Priority |
|-------|------------------|------|-------------------|
| openai/gpt-oss-20b | **49.2%** | $0.07/$0.30 | **PRIMARY TARGET** |
| prime-intellect/intellect-3 | ~78% | $0.20/$1.10 | Good but high baseline |
| qwen/qwen3-235b-a22b-instruct-2507 | 71.2% | $0.22/$0.88 | Good candidate |
| z-ai/glm-4.5-air | 66.8% | $0.20/$1.10 | Room for improvement |

## Lessons Learned

### Executor Backend: Local vs Prime (CRITICAL!)

**Multi-turn tool-calling RL is LATENCY-BOUND, not throughput-bound!**

| Backend | Tool Call Latency | 1024 Rollouts | Use Case |
|---------|-------------------|---------------|----------|
| `"prime"` | ~1.5 seconds | ~25 minutes | Production (secure) |
| `"local"` | ~0.05 seconds | **~1 minute** | Training on dedicated pods |

```toml
[orchestrator.env.args]
executor_backend = "local"    # CRITICAL: 10-15x speedup for training!
```

**Why this matters:**
- Remote sandbox adds network latency per tool call
- GPUs sit idle waiting for tool execution
- Reducing rollouts doesn't reduce fixed latency
- Always use `"local"` for training on dedicated/ephemeral pods

**Safe to use local because:**
- BeautifulSoup tasks only parse HTML (no file/network access)
- Training pods are ephemeral (no persistent data at risk)

### 2-GPU Setup (2x A6000 48GB)

1. **Memory**: 20B model (BF16) needs ~40GB. Split inference/trainer across GPUs.
2. **CPU Offload**: Required for FSDP with limited GPU memory. Add `fsdp_cpu_offload = true` under `[model]`.
3. **Batch Size**: Start small (32) and increase if memory allows.
4. **seq_len**: 8192 is safe; 16384 may cause OOM with large models.

### Config Format (prime-rl)

- `seq_len` → under `[orchestrator]`, NOT top-level
- `top_p` → NOT allowed in sampling config
- `[orchestrator.buffer]` → use `online_difficulty_filtering = true`
- `[[orchestrator.env]]` → double brackets, no `name` field needed

### WandB on Pod

```bash
# Option 1: .netrc file
echo -e 'machine api.wandb.ai\n  login user\n  password YOUR_KEY' > /root/.netrc
chmod 600 /root/.netrc

# Option 2: Environment variable
export WANDB_API_KEY=YOUR_KEY
```

### vLLM CUDA Segfault Fix

If you see `THCPModule_initExtension` or `cuCtxGetDevice` segfault in vLLM:

```bash
# Set before launching training
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Also add to config
[inference.model]
enforce_eager = true
```

This disables V1 engine and uses spawn instead of fork for child processes.

### Single-GPU Memory Sharing (Trainer + Inference)

When trainer and inference share the same GPU (e.g., single H100):

```toml
# CRITICAL: Limit vLLM memory so trainer has room
[inference]
gpu_memory_utilization = 0.45  # Default 0.9 will fail

[orchestrator]
batch_size = 32           # Reduce from 64+
rollouts_per_example = 4  # Reduce from 8
```

**Error when this is wrong:**
```
ValueError: Free memory on device (48.97/79.18 GiB) on startup is less than
desired GPU memory utilization (0.9, 71.26 GiB).
```

The trainer loads first (~30GB for 7B model), then vLLM tries to claim 90% = OOM.

### Ministral Model Compatibility

Ministral-3-8B-Instruct-2512 causes `KeyError: 'ministral3'` in transformers:
- Root cause: Model type not recognized by AutoModelForCausalLM
- Workaround: Use Qwen2.5-7B-Instruct instead (similar size, vLLM compatible)

### Activation Checkpointing (Recommended for Single-GPU)

Use activation checkpointing instead of CPU offload for single-GPU training:

```toml
[trainer.model.ac]
freq = 1  # Full checkpointing, saves ~20GB for 7B model
```

**Do NOT use `fsdp_cpu_offload = true`** with LoRA - it causes device mismatch errors:
```
RuntimeError: Attempted to set the storage of a tensor on device "cpu"
to a storage on different device "cuda:0"
```

### Single-GPU RL Training Checklist

For running trainer + vLLM inference on one GPU (e.g., H100 80GB):

1. `[inference] gpu_memory_utilization = 0.50` (or less)
2. `[trainer.model.ac] freq = 1` (activation checkpointing)
3. `[trainer.model] seq_len = 2048` (reduce from 4096)
4. `[orchestrator] batch_size = 8, rollouts_per_example = 2`
5. `[inference.model] enforce_eager = true, max_model_len = 4096`
6. `[orchestrator.sampling] max_tokens = 2000` (must fit within context!)

### max_tokens vs max_model_len Constraint

`max_tokens` (in `[orchestrator.sampling]`) must leave room for input tokens:

```
max_tokens + input_tokens <= max_model_len
```

With `max_model_len = 4096` and typical inputs of ~1300-1500 tokens, use `max_tokens = 2000`.

**Error when this is wrong:**
```
ValueError: 'max_tokens' or 'max_completion_tokens' is too large: 4000.
This model's maximum context length is 4096 tokens and your request has 1297 input tokens
(4000 > 4096 - 1297).
```

**Symptom:** Rollout generation hangs indefinitely at "0/8 rollouts" with no errors in main log.
The actual error appears in the vLLM process logs. Check `/tmp/vllm.log` if rollouts stall.

### Spot Instance Risks

Spot instances (e.g., DataCrunch $0.99/hr H100) can be reclaimed at any time:
- Training may stop without warning after 1-2 steps
- WandB shows "running" even after pod is terminated (no graceful shutdown)
- SSH connection fails when pod is reclaimed

**Detection:** If WandB runtime stops increasing and last update is 10+ minutes old, the pod is dead.

**Mitigation options:**
1. **Use on-demand instances** - More expensive but guaranteed availability
2. **Implement checkpointing** - Save LoRA adapters frequently, resume from last checkpoint (see below)
3. **Use Prime's pods** - More stable than cheap spot providers
4. **Monitor aggressively** - Check WandB every 5 minutes, restart quickly if dead

### Checkpointing (CRITICAL for Spot Instances)

**Checkpointing is OFF by default in prime-rl!** You must explicitly enable it via CLI flags.

| Flag | Purpose |
|------|---------|
| `--ckpt` | Enable checkpointing (OFF by default!) |
| `--ckpt.interval N` | Save every N steps |
| `--ckpt.keep-last N` | Keep only last N checkpoints |
| `--ckpt.save-async` | Non-blocking saves |
| `--ckpt.resume-step N` | Resume from checkpoint |

**Checkpoints saved to:** `checkpoints/step_N/`

**Example: 25-step validation run with checkpointing**
```bash
uv run rl @ configs/prime-rl/qwen2.5-7b-h100.toml \
  --ckpt --ckpt.interval 5 --ckpt.keep-last 3 \
  --max-steps 25
```

**Resume from checkpoint:**
```bash
uv run rl @ configs/prime-rl/qwen2.5-7b-h100.toml \
  --ckpt.resume-step 20 --max-steps 50
```

**What gets checkpointed:**
- **Trainer**: FSDP model shards, optimizer/scheduler state, progress metrics
- **Orchestrator**: Progress (step, tokens, samples)
- **Inference**: Nothing (stateless) - orchestrator reloads correct weights automatically

**WARNING: `--ckpt.keep-last N` does NOT auto-delete old checkpoints!**

As of 2026-01-06, the `--ckpt.keep-last` flag does not actually clean up old checkpoints. Both checkpoints and weights directories accumulate:
- Each checkpoint: ~31GB (optimizer state, model shards)
- Each weight save: ~16GB (model weights)
- Total per step: ~47GB

**Manual cleanup required:**
```bash
# After step 15 checkpoint is saved, delete step 10
rm -rf /app/outputs/checkpoints/step_10 /app/outputs/weights/step_10
```

Plan for disk to fill to ~90% at each checkpoint save, then manually clean up to recover space.

---

## Resilient Training System

Automated checkpoint sync and recovery for spot instance training.

### Architecture

```
+-------------------------------------------------------------------+
|                         TRAINING POD                              |
|  +-------------+    +-------------+    +-----------------------+  |
|  |  prime-rl   |--->| checkpoints |--->| checkpoint_sync.sh    |  |
|  |  training   |    |   /ckpt/    |    | (every 5 min)         |  |
|  +-------------+    +-------------+    +-----------+-----------+  |
|                                                    |               |
|                                                    v               |
|                                          +-----------------------+ |
|                                          |   Backblaze B2        | |
|                                          |  beautifulsoup-rl     | |
|                                          +-----------------------+ |
+-------------------------------------------------------------------+

+-------------------------------------------------------------------+
|                   GITHUB ACTIONS (Always-On)                      |
|  +-----------------------+    +-------------------------------+   |
|  |  training-monitor.yml |--->|  training_controller.py       |   |
|  |  (every 10 min)       |    |  Check WandB, auto-provision  |   |
|  +-----------------------+    +-------------------------------+   |
+-------------------------------------------------------------------+
```

### Components

| Script | Purpose |
|--------|---------|
| `scripts/checkpoint_sync.sh` | Atomic sync of checkpoints to B2 with latest.json pointer |
| `scripts/onstart.sh` | Vast.ai auto-start: pull checkpoint + resume |
| `scripts/training.service` | Systemd unit for process supervision |
| `scripts/training_controller.py` | GitHub Actions: monitor WandB + auto-provision |
| `scripts/provision_vast.py` | Manual Vast.ai provisioning |
| `scripts/wandb_monitor.py` | WandB health check with stall detection |

### Quick Start

```bash
# 1. Set up environment (on pod)
curl -sSL https://raw.githubusercontent.com/seconds-0/beautifulsoup-rl/main/scripts/pod_setup.sh | bash

# 2. Start training (checkpoints auto-sync to B2)
tmux new -s training
uv run rl @ /tmp/config.toml --ckpt --ckpt.interval 5 --ckpt.keep-last 3
```

### After Spot Interruption (Automatic)

1. Vast.ai pauses instance when outbid
2. When price drops, instance resumes
3. `onstart.sh` runs automatically:
   - Pulls latest checkpoint from B2
   - Resumes training from that step
4. Training continues seamlessly

### GitHub Actions Controller

The `.github/workflows/training-monitor.yml` workflow runs every 10 minutes:
- Checks WandB for training status (healthy/failed/stalled)
- Auto-provisions new Vast.ai instance if training died
- Idempotent (safe to run repeatedly)

**Required GitHub Secrets:**
- `WANDB_API_KEY`
- `VAST_API_KEY`
- `B2_APPLICATION_KEY_ID`
- `B2_APPLICATION_KEY`

### Manual Operations

```bash
# Check training status
python scripts/wandb_monitor.py --run-id bs4-qwen3-8b

# Search for Vast.ai instances
python scripts/provision_vast.py search --gpu H100 --count 2 --max-price 2.50
python scripts/provision_vast.py search --gpu 4090 --count 2 --max-price 0.50  # Much cheaper!

# Provision new instance
python scripts/provision_vast.py create --run-id bs4-qwen3-8b --gpu H100 --count 2
python scripts/provision_vast.py create --run-id bs4-qwen3-8b --gpu 4090 --count 2 --max-price 0.50

# Terminate instances
python scripts/provision_vast.py terminate --run-id bs4-qwen3-8b --force
```

### Deployment Checklist

Before deploying a training pod:

1. **Vast.ai Account**
   - [ ] Account has credit (check: `vastai show user --raw | jq .balance`)
   - [ ] API key is set: `export VAST_API_KEY=...`

2. **GitHub Secrets** (for auto-recovery)
   - [ ] `WANDB_API_KEY` - WandB logging
   - [ ] `VAST_API_KEY` - Auto-provisioning
   - [ ] `B2_APPLICATION_KEY_ID` - Checkpoint storage
   - [ ] `B2_APPLICATION_KEY` - Checkpoint storage

3. **B2 Bucket**
   - [ ] Bucket exists: `beautifulsoup-rl`
   - [ ] B2 CLI authorized: `b2 authorize-account`

4. **Config**
   - [ ] Upload config.toml to B2: `b2 file upload beautifulsoup-rl config.toml $RUN_ID/config.toml`

### GPU Pricing (2026-01-05)

| GPU | Count | Price/hr | VRAM | Availability |
|-----|-------|----------|------|--------------|
| RTX 4090 | 2 | $0.45-0.72 | 48GB | Abundant |
| H100 PCIE | 2 | $5.73 | 160GB | Limited |
| H100 SXM5 | 2 | ~$4-6 | 160GB | Rare |

**Recommendation:** Use 2x RTX 4090 for 7-8B models - 10x cheaper than H100 and sufficient VRAM.

### Checkpoint Flow

1. **Training writes**: `checkpoints/step_N/` (includes optimizer.pt when complete)
2. **Sync script detects**: Only syncs directories with `optimizer.pt` (atomic)
3. **Upload to B2**: `b2://beautifulsoup-rl/{RUN_ID}/step_N/`
4. **Update pointer**: `latest.json` with step + WANDB_RUN_ID
5. **On resume**: Download `latest.json`, then only the latest checkpoint

### WandB Run Continuity

The same WandB run ID is preserved across restarts:
- `latest.json` includes `wandb_run_id` field
- `onstart.sh` exports `WANDB_RUN_ID` before resuming
- Metrics continue in the same WandB run (no fragmentation)

### TODO: Multi-Provider Fallback

If Vast.ai has no availability, fall back to RunPod:
- [ ] Add RunPod API integration to `provision_vast.py`
- [ ] Modify `training_controller.py` to try providers in order
- [ ] Handle RunPod's 5s termination warning (SIGTERM → checkpoint → die)
