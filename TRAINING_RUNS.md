# Training Runs Log

Track all RL training experiments for BeautifulSoup environment.

## Active Runs

### Run: bs4-rl-qwen2.5-7b-1000steps (2026-01-04)

- **Model**: Qwen/Qwen2.5-7B-Instruct (7B params)
- **Config**: configs/prime-rl/beautiful-soup-env.toml
- **Pod**: bs4-rl-training (2x A6000 48GB)
- **Start**: 2026-01-04 ~21:00 UTC
- **Status**: LAUNCHING (attempt 8)
- **W&B Project**: beautiful-soup-env
- **Baseline**: TBD (need to run smoke test)

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
