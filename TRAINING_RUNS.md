# Training Runs Log

Track all RL training experiments for BeautifulSoup environment.

## Active Runs

### Run: bs4-rl-qwen2.5-7b-h100-v5 (2026-01-05) - RUNNING ✅

- **Model**: Qwen/Qwen2.5-7B-Instruct (7B params)
- **Config**: /tmp/config.toml (remote) - see below
- **Pod**: 1x H100 80GB SXM5 (DataCrunch spot, $0.99/hr)
- **Pod ID**: fd6a88cad0f04019841355376d088f5b
- **Start**: 2026-01-04 23:50 UTC
- **Status**: RUNNING ✅
- **W&B Run**: https://wandb.ai/seconds-0-domus-magna-inc/beautiful-soup-env/runs/t6agrtyb
- **GPU Usage**: 74GB/81GB (91%) with activation checkpointing
- **Baseline**: N/A (OpenRouter doesn't support tool calling for Qwen2.5-7B)
- **First Step**: Reward 0.0238, 287s, 50 tokens/s

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
