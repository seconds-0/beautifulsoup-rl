# BeautifulSoup RL Benchmark & Training Skill

Run benchmarks, evaluations, and RL training on the BeautifulSoup RL environment. Includes comprehensive Prime model registry.

## ALWAYS Look Up Documentation First

**Verifiers and Prime-RL APIs change frequently.** Before running any eval or training:

1. **Resolve Context7 library IDs:**
   - `mcp__context7__resolve-library-id` with `libraryName="verifiers"`
   - `mcp__context7__resolve-library-id` with `libraryName="prime-rl"`

2. **Query current docs:**
   - `/websites/verifiers_readthedocs_io_en` (benchmark 93.5, preferred)
   - `/primeintellect-ai/verifiers` (benchmark 67.8)
   - `/ia03/prime-environments` (benchmark 61.3)

3. **Example queries:**
   - "How to configure orchestrator for training"
   - "Environment evaluation parameters"
   - "prime env eval command options"

**Why this matters:** Training config params, CLI flags, and API signatures evolve. Outdated assumptions cause silent failures or wasted compute.

## Trigger Phrases

- "run benchmark", "benchmark model", "eval on", "test model"
- "run on prime", "prime eval", "prime evaluation"
- "train model", "rl training", "prime rl", "start training"
- "check model performance"
- "prime models", "trainable models", "which model for RL"
- "model sizes", "model parameters", "model pricing"
- "training progress", "check training", "monitor training", "step time"
- "checkpoint", "evaluate checkpoint", "benchmark checkpoint"
- "stop training", "continue training", "when to stop", "training metrics"
- "reward curve", "entropy", "KL divergence", "training health"

---

## Model Registry (Prime Cloud)

### Best RL Training Targets (Smoke Test n=10, 2026-01-04)
| Model | Baseline | Params | Cost ($/1M in/out) | Notes |
|-------|----------|--------|-------------------|-------|
| `openai/gpt-oss-20b` | **53%** | 20B | $0.07/$0.30 | **Best target** - cheap, room to grow |
| `qwen/qwen3-vl-8b-instruct` | **50%** | 8B | $0.18/$0.70 | Good small target |
| `mistralai/mistral-small-3.2-24b-instruct` | **36.3%** | 24B | $0.20/$0.50 | Needs improvement |

### Validation/Ceiling Models (Smoke Test n=10, 2026-01-04)
| Model | Baseline | Params | Cost ($/1M in/out) | Notes |
|-------|----------|--------|-------------------|-------|
| `qwen/qwen3-235b-a22b-instruct-2507` | **100%** | 235B (22B) | $0.22/$0.88 | Perfect on smoke test |
| `prime-intellect/intellect-3` | **92.5%** | 106B (12B) | $0.20/$1.10 | MoE, GLM-based, **PI's model** |
| `z-ai/glm-4.5-air` | **79.7%** | 106B (12B) | $0.20/$1.10 | MoE, frontier baseline |
| `openai/gpt-5-nano` | **98.3%** | ? | $0.05/$0.40 | Ceiling (closed API, not trainable) |

### Models with Issues (avoid for RL)
| Model | Baseline | Issue |
|-------|----------|-------|
| `arcee-ai/trinity-mini` | 21% | JSON truncation despite max_tokens=10000 |
| `qwen/qwen3-30b-a3b-instruct-2507` | ~20% | JSON truncation (3B active too small) |
| `qwen/qwen3-30b-a3b-thinking-2507` | ? | Likely truncation (same architecture) |
| `meta-llama/llama-4-maverick` | 3.3% | Outputs code blocks, not tool calls |
| `allenai/olmo-3-7b-instruct` | 0% | Outputs markdown, not tool calls |

### Unavailable on Prime (404)
- `google/gemma-3-*` - Not available (except gemma-3-27b-it)
- `allenai/olmo-3-7b-think` - Listed in API but returns 404 at runtime

---

## Complete Prime Model Registry (from `prime inference models`)

*Last verified: 2026-01-04*

### Open-Weight / Trainable Models

#### Nano/Small (<10B active) - Best RL Targets
| Model ID | Params | Price ($/M in/out) | Notes |
|----------|--------|-------------------|-------|
| `arcee-ai/trinity-mini` | 3B | $0.045/$0.15 | Cheapest, JSON truncation issues |
| `qwen/qwen3-30b-a3b-instruct-2507` | 30B (3B active) | $0.20/$0.80 | MoE, instruct |
| `qwen/qwen3-30b-a3b-thinking-2507` | 30B (3B active) | $0.20/$2.40 | MoE, reasoning |
| `allenai/olmo-3-7b-instruct` | 7B | $0.10/$0.20 | No tool calls (outputs markdown) |
| `allenai/olmo-3-7b-think` | 7B | $0.12/$0.20 | Reasoning variant |
| `qwen/qwen3-vl-8b-instruct` | 8B | $0.18/$0.70 | Vision-language |

#### Medium (10-35B active)
| Model ID | Params | Price ($/M in/out) | Notes |
|----------|--------|-------------------|-------|
| `prime-intellect/intellect-3` | 106B (12B active) | $0.20/$1.10 | MoE, GLM-based, **PI's model** |
| `z-ai/glm-4.5-air` | 106B (12B active) | $0.20/$1.10 | MoE, strong baseline |
| `mistralai/mistral-nemo` | 12B | $0.10/$0.25 | Dense |
| `meta-llama/llama-4-maverick` | 400B (17B active) | $0.27/$0.88 | MoE |
| `openai/gpt-oss-20b` | 20B | $0.07/$0.30 | **Cheapest medium** |
| `qwen/qwen3-235b-a22b-2507` | 235B (22B active) | $0.22/$0.88 | MoE base |
| `qwen/qwen3-235b-a22b-instruct-2507` | 235B (22B active) | $0.22/$0.88 | MoE instruct |
| `qwen/qwen3-235b-a22b-thinking-2507` | 235B (22B active) | $0.65/$3.00 | MoE reasoning |
| `mistralai/mistral-small-24b-instruct-2501` | 24B | $0.80/$0.80 | Dense |
| `mistralai/mistral-small-3.2-24b-instruct` | 24B | $0.20/$0.50 | Dense, newer |
| `google/gemma-3-27b-it` | 27B | $0.119/$0.30 | Dense |
| `allenai/olmo-3-32b-think` | 32B | $0.30/$0.55 | Dense, reasoning |
| `moonshotai/kimi-k2-thinking` | ~1T (32B active) | $0.60/$2.50 | MoE |

#### Large (35-100B+ active)
| Model ID | Params | Price ($/M in/out) | Notes |
|----------|--------|-------------------|-------|
| `deepseek/deepseek-v3.2` | 671B (37B active) | $0.28/$0.42 | MoE, good value |
| `mistralai/mistral-large-2512` | ~41B | $0.50/$1.50 | Dense |
| `mistralai/mixtral-8x7b-instruct` | 47B (13B active) | $0.60/$0.60 | MoE |
| `mistralai/mixtral-8x22b-instruct` | 176B (~45B active) | $2.00/$6.00 | MoE |
| `z-ai/glm-4.5` | 106B (full) | $0.60/$2.20 | Dense |
| `z-ai/glm-4.6` | 106B (full) | $0.60/$2.20 | Dense, newer |
| `meta-llama/llama-3.1-70b-instruct` | 70B | $0.90/$0.90 | Dense |
| `meta-llama/llama-3.3-70b-instruct` | 70B | $0.90/$0.90 | Dense, newer |
| `qwen/qwen-2.5-72b-instruct` | 72B | $0.38/$0.40 | Dense |
| `openai/gpt-oss-120b` | 120B | $0.15/$0.60 | Dense |
| `moonshotai/kimi-k2-0905` | ~1T | $1.20/$5.00 | MoE |

#### Vision-Language Models
| Model ID | Params | Price ($/M in/out) | Notes |
|----------|--------|-------------------|-------|
| `qwen/qwen3-vl-8b-instruct` | 8B | $0.18/$0.70 | Small VL |
| `qwen/qwen3-vl-30b-a3b-instruct` | 30B (3B active) | $0.25/$1.00 | MoE VL |
| `qwen/qwen3-vl-30b-a3b-thinking` | 30B (3B active) | $0.16/$0.80 | MoE VL reasoning |
| `qwen/qwen3-vl-235b-a22b-instruct` | 235B (22B active) | $0.40/$1.90 | MoE VL large |
| `qwen/qwen3-vl-235b-a22b-thinking` | 235B (22B active) | $0.784/$3.16 | MoE VL reasoning |

### Closed APIs (Inference Only, NOT Trainable)

| Vendor | Models Available |
|--------|-----------------|
| `anthropic/` | claude-3.5-haiku, claude-3.5-sonnet, claude-3.7-sonnet, claude-3-opus, claude-haiku-4.5, claude-opus-4, claude-opus-4.1, claude-opus-4.5, claude-sonnet-4, claude-sonnet-4.5 |
| `openai/` | gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, gpt-5, gpt-5.1*, gpt-5.2*, gpt-5-chat, gpt-5-codex, gpt-5-mini, gpt-5-nano |
| `google/` | gemini-2.0-flash-001, gemini-2.0-flash-lite-001, gemini-2.5-flash*, gemini-2.5-pro, gemini-3-flash-preview, gemini-3-pro-preview |
| `x-ai/` | grok-3-mini, grok-4, grok-4-fast, grok-code-fast-1 |
| `deepseek/` | deepseek-chat, deepseek-chat-v3-0324, deepseek-chat-v3.1, deepseek-r1-0528, deepseek-v3.1-terminus, deepseek-v3.2-exp, deepseek-v3.2-speciale |
| `qwen/` | qwen3-coder, qwen3-max |

**Exception:** `openai/gpt-oss-20b` and `openai/gpt-oss-120b` ARE open-source and trainable.

---

## prime-rl Architecture Support

These architectures have optimized native implementations:

| Architecture | Models | Notes |
|--------------|--------|-------|
| **Llama** | `meta-llama/*` | All Llama variants |
| **Qwen3Moe** | `qwen/qwen3-*` | MoE variants |
| **Glm4Moe** | `z-ai/glm-*`, `prime-intellect/intellect-3` | GLM-based |
| **Afmoe** | `arcee-ai/trinity-*` | AFMoE architecture |

Other open-weight models (Mistral, OLMo, Gemma, DeepSeek) work via `impl: hf` fallback.

---

## Prime Evaluation Commands

### Quick Validation (50 examples)
```bash
prime env eval seconds-0/beautiful-soup-env \
  -a '{"split":"bench","mode":"mvp"}' \
  -m <model-name> \
  -n 50
```

### Full Benchmark (680 examples)
```bash
prime env eval seconds-0/beautiful-soup-env \
  -a '{"split":"bench","mode":"all"}' \
  -m <model-name> \
  -n 680
```

### With Custom max_tokens (for verbose models)
```bash
prime env eval seconds-0/beautiful-soup-env \
  -a '{"split":"bench","mode":"mvp"}' \
  -m <model-name> \
  -n 50 \
  -t 10000
```

### Check Available Models
```bash
prime inference models
```

---

## Diagnosing JSON Truncation Issues

**Symptom:** Eval logs show warnings like:
```
Malformed tool arguments from model: Unterminated string starting at: line 1 column 10
Could not repair JSON, using empty args
```

**Root Cause:** Model's tool call arguments are getting truncated before completion.

### Diagnostic Steps

1. **Check the error message pattern:**
   - "Unterminated string" = JSON string cut off mid-output
   - "Expecting value: line 1 column 1" = Raw code output instead of JSON
   - "Invalid \escape" = Broken escape sequences from truncation

2. **Look at the Args preview in logs:**
   ```
   Args preview: {"code": "import re\nfrom bs4 import BeautifulSoup\n\nsoup = make_soup()\n\n# Find all price elements...
   ```
   If the preview ends abruptly without closing `"}`, it's truncation.

3. **Test with higher max_tokens:**
   ```bash
   prime env eval ... -m <model> -n 10 -t 10000
   ```

4. **Compare results:** If higher max_tokens improves pass rate, that confirms truncation.

### Token Budget Guidelines

| Model Type | Recommended max_tokens | Notes |
|------------|----------------------|-------|
| Small/efficient (gpt-5-nano, glm-4.5) | 4000-6000 | Usually enough |
| Medium (llama-4-maverick, gpt-oss-20b) | 6000-10000 | Multi-turn needs headroom |
| Verbose (intellect-3, trinity-mini) | 10000+ | Long reasoning first |

### JSON Repair in Adapter

The environment includes automatic JSON repair in `bs4_env/adapters/verifiers_adapter.py`:
- Attempts to fix truncated JSON by adding missing braces
- Falls back to empty args `{}` if unfixable
- Logs warnings when repair is attempted

This prevents eval crashes but doesn't fix the underlying truncation. **Increasing max_tokens is the real fix.**

---

## Local Evaluation (Development)

Use OpenRouter for local development and debugging.

### Prerequisites
```bash
# Ensure .env has OPENROUTER_API_KEY
cat .env | grep OPENROUTER
```

### Run Single Model
```bash
source .env && uv run python -m bs4_env.scripts.eval_with_llm \
  --model <model_id> \
  --num 260 \
  --output results_<name>.json
```

### Run with Custom max_tokens
```bash
source .env && uv run python -m bs4_env.scripts.eval_with_llm \
  --model prime-intellect/intellect-3 \
  --num 50 \
  --max-tokens 10000 \
  --output results_intellect3.json
```

### OpenRouter Models (local testing)
| Model | Pass Rate | Cost | Notes |
|-------|-----------|------|-------|
| `openai/gpt-4o-mini` | ~95% | Cheap | Quick validation |
| `qwen/qwen3-8b` | 43-90% | $0.028/1M | Good RL candidate |
| `mistralai/ministral-8b-2512` | 68.4% | Cheap | Best 8B |

---

## Verifiers Tooling (vf-* commands)

The `verifiers` package provides CLI tools for evaluation and setup.

### Initial Setup (Clone prime-rl)

```bash
# Clone and install prime-rl with starter configs
uv run vf-setup
```

This clones the prime-rl repo and generates default training configs.

### Run Evaluation with vf-eval

```bash
# Generate outputs/ folder for PR submission
uv run vf-eval -s seconds-0/beautiful-soup-env -m gpt-4.1-mini -n 5

# With more examples and rollouts
uv run vf-eval -s seconds-0/beautiful-soup-env -m <model> -n 50 -r 3
```

**Options:**
- `-s` / `--slug`: Environment slug (owner/name)
- `-m` / `--model`: Model to evaluate
- `-n` / `--num`: Number of examples
- `-r` / `--rollouts`: Rollouts per example (default 3)

### Browse Results with vf-tui

```bash
# Interactive TUI to inspect eval results
uv run vf-tui
```

**Features:**
- Navigate through examples
- View rollouts side-by-side
- Inspect rewards and model outputs
- Filter by archetype or reward

### When to Use vf-* vs prime env eval

| Tool | Use Case |
|------|----------|
| `vf-eval` | Generate outputs/ for PR submission, local debugging |
| `vf-tui` | Inspect and debug individual rollouts |
| `prime env eval` | Production benchmarks on Prime Cloud |

---

## Prime RL Training

### Config File
`configs/prime-rl/beautiful-soup-env.toml`

### Architecture Overview

Prime RL runs 3 distributed components:
1. **Orchestrator** - CPU process managing data/scheduling
2. **Trainer** - GPU process for policy updates via FSDP2
3. **Inference** - vLLM-based rollout generation server

### Environment Installation (CRITICAL!)

**Environments MUST be installed via Prime CLI, NOT pip install:**

```bash
# Push env to Hub (from local machine with source)
prime env push

# Install on GPU pod (REQUIRED before training!)
prime env install seconds-0/beautiful-soup-env

# Verify installation
python -c "from verifiers import load_environment; load_environment('seconds-0/beautiful-soup-env')"
```

**Why this matters:** The orchestrator uses `verifiers.load_environment()` which only finds Hub-installed environments. `pip install -e .` does NOT work!

### Config Format (CRITICAL - Updated 2026-01-04)

Reference: [wiki_search example](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/wiki_search/rl.toml)

**CONFIG VALIDATION RULES:**
1. `trainer.model.seq_len` MUST be >= `orchestrator.seq_len`
2. LoRA config: `[trainer.model.lora]` **NOT** `[trainer.model.experimental.lora]`
3. `lora_name` REQUIRED under `[orchestrator]` when using LoRA
4. **NO** `top_p` in `[orchestrator.sampling]`
5. **NO** `mask_truncated_completions` or `zero_truncated_completions`
6. Buffer: `online_difficulty_filtering = true`, **NOT** `type = "online-difficulty"`
7. Launch with: `uv run rl @ path/to/config.toml` (not prime-rl)

### Recommended Default Config

```toml
# === GPU ASSIGNMENT ===
inference_gpu_ids = [0]
trainer_gpu_ids = [1]

# === TRAINING PARAMS ===
max_steps = 1000              # Production: 1000+

# === MODEL ===
[model]
name = "Qwen/Qwen2.5-7B-Instruct"  # 7B, text-only, works with vLLM

[wandb]
project = "beautiful-soup-env"
name = "bs4-rl-qwen2.5-7b-lora"

# === TRAINING ===
[trainer.optim]
lr = 1e-5                     # Conservative LR for LoRA
weight_decay = 0.0

[trainer.model]
seq_len = 4096                # MUST be >= orchestrator.seq_len

[trainer.model.lora]          # NOT experimental.lora!
rank = 8
alpha = 32
dropout = 0.0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# === ORCHESTRATOR ===
[orchestrator]
batch_size = 128              # Rollouts per batch
rollouts_per_example = 8
seq_len = 4096                # Must be <= trainer.model.seq_len
oversampling_factor = 2.0
lora_name = "qwen2.5-7b-bs4-lora"  # REQUIRED for LoRA

[orchestrator.sampling]
max_tokens = 10000            # CRITICAL for multi-turn tool calling

[orchestrator.buffer]
online_difficulty_filtering = true  # NOT type = "online-difficulty"

# === ENVIRONMENT ===
[[orchestrator.env]]
id = "seconds-0/beautiful-soup-env"  # Must be installed via prime env install

[orchestrator.env.args]
split = "train"
mode = "tiered"
difficulty = "mixed"
seed = 42
executor_backend = "prime"
network_access = true
timeout_s = 30.0
max_output_chars = 10000

[inference.model]
enable_auto_tool_choice = true
tool_call_parser = "hermes"
```

### Key Parameters Explained

| Parameter | Recommended | Why |
|-----------|-------------|-----|
| `max_tokens` | **10000** | Multi-turn tool calling needs headroom. Models generate reasoning + code + may retry. 768 caused truncation, 4000 was marginal. |
| `batch_size` | 256 | Balance between throughput and memory. Decrease if OOM. |
| `rollouts_per_example` | 8 | More rollouts = better reward signal, but more compute. 8 is good default. |
| `temperature` | 0.7 | Higher = more exploration. 0.7 balances creativity vs coherence. |
| `mode` | "tiered" | Overweights harder archetypes for faster learning signal. Use "all" for final training. |
| `lr` | 1e-5 | Conservative for LoRA. Can try 5e-5 if learning is slow. |
| `lora.rank` | 8 | Higher = more capacity but slower. 8 is good starting point. |

### Prime RL CLI Commands Reference

**From [prime-rl docs](https://github.com/PrimeIntellect-ai/prime-rl):**

```bash
# === ENVIRONMENT MANAGEMENT ===
prime env list                               # Browse available environments
prime env info owner/env-name                # View environment details
prime env install owner/env-name             # Install from Hub (REQUIRED!)
prime env install owner/env-name@1.0.0       # Install specific version
prime env push                               # Upload local env to Hub
prime env pull owner/env-name                # Download source code

# === TRAINING COMMANDS ===
uv run rl @ config.toml                      # Combined training (starts all 3)
uv run trainer @ train.toml                  # Trainer only
uv run orchestrator @ orch.toml              # Orchestrator only
uv run inference @ infer.toml                # Inference server only

# === OTHER ENTRYPOINTS ===
uv run sft @ sft.toml                        # Supervised fine-tuning
uv run eval @ eval.toml                      # Evaluation
uv run synthesize @ synth.toml               # Synthetic data generation

# === POD MANAGEMENT ===
prime pods list                              # View running pods
prime pods create --name my-pod              # Create new pod
prime pods status <pod-id>                   # Check pod state
prime pods ssh <pod-id>                      # SSH into pod
prime pods terminate <pod-id>                # Shutdown pod

# === GPU AVAILABILITY ===
prime availability list                      # All available GPUs
prime availability list --gpu-type H100_80GB # Filter by type
```

### Running Training (GPU Pod Workflow)

Training requires a GPU pod with the `prime_rl` image.

```bash
# 1. SSH into pod
prime pods ssh <pod-id>
# OR: ssh -i ~/.ssh/primeintellect_ed25519 root@<ip> -p <port>

# 2. Clone/update repo
cd /workspace
git clone https://github.com/seconds-0/beautifulsoup-rl.git
cd beautifulsoup-rl
git pull

# 3. CRITICAL: Install environment from Hub
prime env install seconds-0/beautiful-soup-env

# 4. Configure WandB
export WANDB_API_KEY=your-key
# OR create /root/.netrc:
# machine api.wandb.ai
#   login user
#   password YOUR_KEY

# 5. Start training in tmux WITH CHECKPOINTING
tmux new -s training
source /app/.venv/bin/activate
uv run rl @ configs/prime-rl/beautiful-soup-env.toml \
  --ckpt --ckpt.interval 5 --ckpt.keep-last 3 \
  2>&1 | tee /tmp/training.log

# 6. Monitor logs
tail -f /tmp/training.log
# OR check: outputs/logs/orchestrator.stdout, trainer/rank_0.log
```

### Checkpointing (CRITICAL for Spot Instances!)

**Checkpointing is OFF by default.** Always enable it:

```bash
uv run rl @ config.toml --ckpt --ckpt.interval 5 --ckpt.keep-last 3
```

| Flag | Purpose |
|------|---------|
| `--ckpt` | Enable checkpointing (required!) |
| `--ckpt.interval N` | Save every N steps |
| `--ckpt.keep-last N` | Keep only last N checkpoints |
| `--ckpt.save-async` | Non-blocking saves (optional) |
| `--ckpt.resume-step N` | Resume from step N |

**Resume from checkpoint:**
```bash
uv run rl @ config.toml --ckpt.resume-step 20 --max-steps 50
```

**Checkpoints saved to:** `checkpoints/step_N/`

### Important Notes

- **`rl @`** starts all 3 components (orchestrator, trainer, inference)
- **Environment MUST be installed via `prime env install`** - pip install doesn't work!
- **Always enable checkpointing** - spot instances can be reclaimed!
- Config file path is relative to current directory
- Use `tmux attach -t training` to reconnect
- Logs go to `outputs/logs/` and `/tmp/training.log`

**GPU Requirements**: Config uses `inference_gpu_ids = [0]` and `trainer_gpu_ids = [1]`, requiring at least 2 GPUs.

### Single-GPU Training Constraints

For single H100 80GB (inference + trainer sharing GPU 0):

| Setting | Value | Why |
|---------|-------|-----|
| `gpu_memory_utilization` | 0.50 | Leave room for trainer (~30GB for 7B) |
| `[trainer.model.ac] freq` | 1 | Full activation checkpointing saves ~20GB |
| `max_tokens` | 2000 | Must fit: max_tokens + input (~1500) < max_model_len |
| `fsdp_cpu_offload` | **NEVER with LoRA** | Causes device mismatch errors |

**Single-GPU config:** `configs/prime-rl/qwen2.5-7b-h100.toml`

### Pod Setup Script

Run on every new pod to set up vLLM environment variables and install deps:

```bash
# Download and run
curl -sSL https://raw.githubusercontent.com/seconds-0/beautifulsoup-rl/main/scripts/pod_setup.sh | bash

# Or manually set critical env vars:
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

### Model Selection for Training

When selecting models:
- Must be on HuggingFace Hub
- Must NOT be Vision-Language (VL) models
- Must be compatible with vLLM weight reloading

**Verify before training:**
```bash
python scripts/verify_model_for_training.py <model-name>
```

**Tested working:** Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3-4B-Instruct
**Known issues:** gpt-oss-20b (vLLM bug), qwen3-vl models (VL not supported), Mistral (not on HF)

### Monitoring Training

```bash
# Attach to tmux session with wandb monitor
tmux attach -t wandb-monitor

# Or view W&B dashboard
open https://wandb.ai/YOUR_USERNAME/beautiful-soup-env

# Track runs in TRAINING_RUNS.md
```

### Bootstrap Strategy (0% Models)
For models with 0% baseline, use staged training:

1. **Phase 1**: `mode="bootstrap"`, `difficulty="primer"` - 0% to 10%
2. **Phase 2**: `mode="bootstrap"` - 10% to 30%
3. **Phase 3**: `mode="tiered"` - 30% to 50%
4. **Phase 4**: `mode="all"` - 50% to 65%

### Auto-Curriculum (Recommended)

**Use `curriculum_enabled=true` for automatic phase progression without manual config switching.**

The environment tracks samples via `setup_state()` calls and automatically adjusts `difficulty_weights` based on training step.

#### Config Format

```toml
[orchestrator.env.args]
split = "train"
mode = "tiered"
difficulty = "mixed"
curriculum_enabled = true
samples_per_step = 256  # batch_size(64) * rollouts_per_example(4)

# Phase 1 (steps 0-150): Lock in tool format
[[orchestrator.env.args.curriculum_phases]]
until_step = 150
weights = { primer = 0.8, easy = 0.2, medium = 0.0, hard = 0.0 }

# Phase 2 (steps 150-400): Build confidence
[[orchestrator.env.args.curriculum_phases]]
until_step = 400
weights = { primer = 0.0, easy = 0.6, medium = 0.4, hard = 0.0 }

# Phase 3 (steps 400-750): Breadth + hard exposure
[[orchestrator.env.args.curriculum_phases]]
until_step = 750
weights = { primer = 0.1, easy = 0.35, medium = 0.35, hard = 0.2 }

# Phase 4 (steps 750-1000): Specialize on hard
[[orchestrator.env.args.curriculum_phases]]
until_step = 1000
weights = { primer = 0.0, easy = 0.15, medium = 0.35, hard = 0.5 }
```

#### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `curriculum_enabled` | Enable auto-curriculum (default: false) |
| `samples_per_step` | Samples per training step for step estimation |
| `curriculum_phases` | List of phases with `until_step` and `weights` |
| `weights` | Dict of difficulty weights: `primer`, `easy`, `medium`, `hard` |

#### Example Config

See `configs/prime-rl/qwen25-3b-2x5090-curriculum.toml` for a complete example.

---

## Pre-flight Checks

### Before Running on Prime
```bash
# 1. Check config alignment
cat configs/prime-rl/beautiful-soup-env.toml | grep max_tokens
# Should show: max_tokens = 10000

# 2. Verify environment is pushed
prime env push --dry-run

# 3. List available models
prime inference models

# 4. Verify sandbox dependencies
# Docker image must have: bs4, lxml, html5lib
```

---

## Key Files

| File | Purpose |
|------|---------|
| `bs4_env/scripts/eval_with_llm.py` | Local evaluation script |
| `configs/prime-rl/beautiful-soup-env.toml` | Prime RL training config |
| `TEST_RECORDS.md` | Historical benchmark results |
| `results_*.json` | Individual run outputs |
| `bs4_env/grading/rubric.py` | Reward computation |
| `bs4_env/prompt.py` | Model prompts |

---

## Analyzing Results

### Quick Summary
```bash
cat results_<name>.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'Pass rate: {data[\"pass_rate\"]*100:.1f}%')
print(f'Perfect rate: {data[\"perfect_rate\"]*100:.1f}%')
print(f'Avg reward: {data[\"avg_reward\"]:.3f}')
"
```

### Per-Archetype Breakdown
```bash
cat results_<name>.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for arch, avg in sorted(data['by_archetype'].items(), key=lambda x: x[1]):
    print(f'{arch}: {avg:.2f}')
"
```

---

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| 0 tool calls | max_tokens too low | Increase to 10000+ |
| Sandbox timeout | Code too slow | Check executor timeout |
| Model not found | Wrong model ID | Verify with `prime inference models` |
| Empty response | API error | Check retry logic, increase backoff |
| JSON truncation | Token limit | Use `-t 10000` flag |

---

## Common Training Failure Modes (Postmortem 2026-01-06)

These issues caused significant debugging time. Run `scripts/validate_training_ready.sh` before starting training!

### 1. WANDB_API_KEY Not Configured

**Symptom:** Training crashes immediately with `wandb.errors.errors.UsageError: api_key not configured (no-tty)`

**Cause:** WANDB_API_KEY environment variable not set.

**Fix:**
```bash
export WANDB_API_KEY="your-key-here"
# Or persist it:
echo 'export WANDB_API_KEY="your-key"' > ~/.wandb_api_key
source ~/.wandb_api_key
```

### 2. Stale vLLM Process Hogging GPU

**Symptom:** New training gets SIGTERM, GPU shows high memory usage before training starts.

**Cause:** Previous training run left vLLM inference server running.

**Fix:**
```bash
pkill -9 -f vllm
pkill -9 -f VLLM
nvidia-smi  # Verify GPU memory is ~1MB
```

### 3. `uv` Not in PATH

**Symptom:** Subprocesses fail with `FileNotFoundError: [Errno 2] No such file or directory: 'uv'`

**Cause:** PATH doesn't include `/root/.local/bin` where uv is installed.

**Fix:**
```bash
export PATH="/root/.local/bin:$PATH"
# Add to .bashrc for persistence
```

### 4. Stale Checkpoints Causing Deadlock (MOST COMMON!)

**Symptom:**
- Trainer shows "Starting training loop" but never completes any steps
- Orchestrator completes steps but trainer doesn't respond
- Orchestrator hits async barrier waiting for checkpoint that never comes

**Cause:** Old checkpoints in `outputs/run_default/checkpoints/` from previous run. The startup script cleans `outputs/checkpoints/` but NOT `outputs/run_default/checkpoints/`.

**Fix:**
```bash
# Run the comprehensive cleanup script:
bash scripts/clean_training_state.sh

# Or manually clean ALL checkpoint locations:
rm -rf outputs/checkpoints/*
rm -rf outputs/run_default/checkpoints/*  # CRITICAL - often missed!
rm -rf outputs/run_default/rollouts/*
rm -rf outputs/run_default/broadcasts/*
rm -rf outputs/weights/*
```

### 5. Missing `--ckpt.keep-last N`

**Symptom:** Disk fills up rapidly (each checkpoint is ~12GB for 3B model).

**Cause:** Checkpointing enabled but no cleanup configured - keeps ALL checkpoints.

**Fix:**
```bash
# Always include --ckpt.keep-last:
uv run rl @ config.toml --ckpt --ckpt.interval 5 --ckpt.keep-last 3
```

### Pre-Training Checklist

Run this before EVERY training start:

```bash
# 1. Validate readiness (catches all common issues)
bash scripts/validate_training_ready.sh

# 2. If validation fails, clean state:
bash scripts/clean_training_state.sh

# 3. Set environment
export PATH="/root/.local/bin:$PATH"
source ~/.wandb_api_key
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 4. Start with ALL required flags
nohup uv run rl @ config.toml \
  --ckpt --ckpt.interval 5 --ckpt.keep-last 3 \
  > /root/training.log 2>&1 &

# 5. Verify training is progressing (wait 5 min, check both logs)
tail outputs/run_default/logs/orchestrator.log | grep SUCCESS
tail outputs/logs/trainer/rank_0.log | grep SUCCESS
```

### Quick Diagnostic Commands

```bash
# Check GPU memory (should be ~1MB before starting)
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Check for stale processes
ps aux | grep -E "vllm|prime_rl|torchrun" | grep -v grep

# Check for stale checkpoints (should be empty before fresh start)
ls outputs/run_default/checkpoints/
ls outputs/checkpoints/

# Check trainer is actually processing (should show step completions)
tail -f outputs/logs/trainer/rank_0.log | grep SUCCESS
```

### 6. Step Counter Desynchronization on Resume (BUG - 2026-01-06)

**Symptom:**
- Training resumes from checkpoint (e.g., step 10)
- Trainer completes ONE step then hangs forever
- GPU utilization drops to 0%
- Orchestrator shows "Hit async barrier waiting for checkpoint N"

**Root Cause (prime-rl bug):**

When resuming training, two different step counters are initialized from DIFFERENT sources:

| Component | Source | Initialized To |
|-----------|--------|----------------|
| DataLoader (packer sender/receiver) | Trainer checkpoint (`--ckpt.resume-step`) | Resume step (e.g., 10) |
| `runs.progress` (packer's batch reader) | `max(orchestrator checkpoints)` | Highest checkpoint (e.g., 15) |

In `prime_rl/trainer/runs.py` lines 114-116:
```python
prev_ckpt_steps = set(
    int(i.stem.split("_")[-1]) for i in (self.get_run_dir(new_id) / "checkpoints").glob("step_*")
)
self.progress[new_id].step = max(prev_ckpt_steps)  # Uses ORCHESTRATOR checkpoints!
```

This causes a mismatch where:
- Packer's **sender** writes to `outputs/rollouts/step_10/rank_0.bin`
- Packer's **receiver** reads from `outputs/run_default/rollouts/step_15/rollouts.bin`
- After one iteration: receiver looks for `step_16/rollouts.bin` which DOESN'T EXIST
- Trainer waits forever for `step_11/rank_0.bin` that never gets written

**How to Detect:**

```bash
# Check if step counters are mismatched
ls outputs/run_default/checkpoints/  # Shows step_5, step_10, step_15
ls outputs/rollouts/                  # Shows only step_10 (one step processed)
ls outputs/run_default/rollouts/      # Shows step_10 through step_15 (orchestrator ahead)
```

If trainer processed only ONE step and is hanging, and orchestrator has more checkpoints than trainer's resume step, this is the bug.

**Fix (before restarting):**

```bash
# Option 1: Clean ALL state and start fresh (safest)
bash scripts/clean_training_state.sh
# Remove --ckpt.resume-step flag

# Option 2: Sync orchestrator checkpoints to trainer resume step
# Delete orchestrator checkpoints that are AHEAD of trainer checkpoint
rm -rf outputs/run_default/checkpoints/step_15  # Keep only step_10
rm -rf outputs/run_default/checkpoints/step_5   # Optional cleanup
# Now runs.progress will initialize to step_10 (matching trainer)
```

**Prevention:**

1. Always clean state before resuming: `bash scripts/clean_training_state.sh`
2. Don't mix checkpoint resume with stale orchestrator checkpoints
3. Long-term fix needed in prime-rl: `runs.progress` should accept start_step from checkpoint loading

---

## Performance Optimization & Debugging

### Understanding the Bottleneck: Latency vs Throughput

**Multi-turn tool-calling RL is LATENCY-BOUND, not throughput-bound.**

This is the most important insight for optimizing training speed. In each step:
1. Model generates a tool call
2. Tool executes (this is where time is spent!)
3. Result returns to model
4. Repeat 3-10+ times per rollout

The GPU sits idle waiting for tool execution results. Optimizing inference parameters (max_tokens, batch sizes) has minimal impact if tool execution is the bottleneck.

### The `executor_backend` Setting (CRITICAL)

| Backend | Tool Call Latency | 1024 Rollouts | Use Case |
|---------|-------------------|---------------|----------|
| `"prime"` | ~1.5 seconds | ~25 minutes | Production (secure sandboxing) |
| `"local"` | ~0.05 seconds | **~1 minute** | Training on dedicated pods |

**Default is `"prime"` (remote sandbox)** - safe but slow.

```toml
[orchestrator.env.args]
executor_backend = "prime"    # Remote sandbox - SLOW but safe
executor_backend = "local"    # Local subprocess - FAST, use for training
```

### When to Use Local Execution

Use `executor_backend = "local"` when:
- Training on dedicated ephemeral pod (no persistent data at risk)
- Tasks are sandboxed by nature (BS4 only parses HTML, no file/network access)
- Speed is critical (10-15x faster!)

Use `executor_backend = "prime"` when:
- Running on shared infrastructure
- Tasks involve file system or network access
- Security isolation is required

### Why Common "Optimizations" Don't Work

| "Optimization" | Expected Impact | Actual Impact | Why |
|----------------|-----------------|---------------|-----|
| Reduce `max_tokens` (10k→4k) | 2.5x faster | ~0% | Tool-calling models generate short responses, stop at tool call |
| Reduce `rollouts_per_example` (8→4) | 2x faster | **Worse per-rollout** | Fixed latency spread over fewer rollouts |
| Reduce `oversampling_factor` (2→1) | 2x faster | ~0% | Same latency, just less exploration |
| Enable prefix caching | 10-20% faster | **Not supported** | prime-rl doesn't support this parameter |
| Increase `max_num_seqs` | Better scheduling | **Not supported** | prime-rl doesn't support this parameter |

**The math:**
- v1: 1024 rollouts × 1.5s latency = ~25 min waiting
- v2: 512 rollouts × 1.5s latency = ~25 min waiting (same wall time, half the work!)
- Local: 1024 rollouts × 0.05s latency = **~50 seconds**

### Diagnosing Performance Issues

**Step 1: Check where time is spent**

Look at training logs for timing breakdown:
```
Step 0 completed in 27m 15s
  - Rollout generation: 26m 30s (97%)  ← Tool execution dominates
  - Reward computation: 30s (2%)
  - Policy update: 15s (1%)
```

**Step 2: Identify the bottleneck**

If rollout generation dominates (>80%), the bottleneck is tool execution latency.

**Step 3: Calculate per-rollout time**

```
per_rollout_time = step_time / (batch_size × rollouts_per_example)
```

If per-rollout time is >1 second, you're likely latency-bound.

**Step 4: Test with local executor**

Switch to `executor_backend = "local"` and observe speedup. If >5x faster, latency was the bottleneck.

### Recommended Training Config (Speed-Optimized)

```toml
[orchestrator]
batch_size = 128
rollouts_per_example = 8      # More rollouts = better signal (local is fast enough)
oversampling_factor = 2.0     # More exploration (local is fast enough)

[orchestrator.sampling]
max_tokens = 4096             # Match seq_len (multi-turn tool calls rarely need more)

[orchestrator.env.args]
executor_backend = "local"    # CRITICAL: 10-15x speedup
timeout_s = 30.0              # Local execution is fast, keep reasonable timeout
```

### Performance Expectations

| Config | Step Time | Per-Rollout | Notes |
|--------|-----------|-------------|-------|
| Prime executor, 512 rollouts | ~27 min | 3.2s | Latency-bound, GPU idle |
| Prime executor, 1024 rollouts | ~27 min | 1.6s | Same latency, better utilization |
| **Local executor, 1024 rollouts** | **~2-3 min** | ~0.1s | **GPU-bound, optimal** |

### Lessons Learned (2026-01-05 Postmortem)

1. **Profile before optimizing** - Always identify the actual bottleneck first
2. **Latency vs throughput** - Multi-turn interactions are latency-bound
3. **Question defaults** - `executor_backend="prime"` is safe but slow
4. **The bottleneck was environment execution, not inference**
5. **Reducing work doesn't reduce fixed latency** - Halving rollouts didn't halve time

---

## Training Monitoring & Evaluation

### Training Time Estimates

With local executor on 2x H100 (1024 rollouts/step):

| Milestone | Time from Start | Wall Clock Example |
|-----------|-----------------|-------------------|
| Step 100 | ~6 hours | +6h |
| Step 500 | ~30 hours | +1.25 days |
| Step 1000 | ~61 hours | +2.5 days |

**Step time breakdown:** ~220s average (3-4 minutes)

### Checkpoint Benchmarking Protocol

**When to benchmark:** Steps 100, 500, 1000 (or at any checkpoint)

**How to benchmark:**
```bash
# Load checkpoint weights into vLLM and eval
# (exact command depends on how weights are exported)
prime env eval seconds-0/beautiful-soup-env \
  -a '{"split":"bench","mode":"mvp"}' \
  -m <checkpoint-model-path> \
  -n 100
```

**Target improvements vs baseline:**

| Checkpoint | Expected Improvement | Interpretation |
|------------|---------------------|----------------|
| Step 100 | +5-10% | Early signal, learning started |
| Step 500 | +15-20% | Solid progress |
| Step 1000 | +25-30% | Strong training run |

### Key WandB Metrics to Monitor

| Metric | Healthy Range | Warning Sign | Action |
|--------|---------------|--------------|--------|
| **Reward** | Increasing | Plateau >200 steps | Consider stopping |
| **Policy entropy** | 0.05-0.15 | < 0.05 | Overfitting, stop |
| **KL divergence** | < 0.1 | > 0.5 | Unstable, reduce LR |
| **Grad norm** | < 1.0 | Spikes > 5.0 | Reduce LR or batch size |

### Stop Training Criteria

Stop training if ANY of these occur:

1. **Reward plateau** - No improvement for 300+ consecutive steps
2. **Benchmark stagnation** - < 5% improvement from step 500 → 1000
3. **Entropy collapse** - Policy entropy drops below 0.05 (overfit)
4. **KL explosion** - KL divergence spikes above 0.5 (training unstable)
5. **Gradient explosion** - Grad norm consistently > 5.0

### Continue Training Criteria

Continue training if:

1. Reward curve still trending upward
2. Entropy stable in 0.05-0.15 range
3. Benchmark shows improvement at each checkpoint
4. No signs of overfitting (train reward >> eval reward)

### Monitoring Checklist (Every 100 Steps)

- [ ] Check WandB reward curve - still increasing?
- [ ] Check entropy - not collapsed?
- [ ] Check KL divergence - stable?
- [ ] If at checkpoint: run benchmark eval
- [ ] Compare to previous checkpoint - improved?

---

## Dataset Splits

| Split | Size | Purpose |
|-------|------|---------|
| train | ~12,000 | RL training |
| eval | ~500 | Development testing |
| bench | 260-1040 | Final benchmarking |

---

## Config Version Tracking

### Config Metadata Header

All training configs should include a `[config]` section at the top:

```toml
[config]
version = "v2"
created = "2026-01-05"
env_version = "0.1.1"
based_on = "qwen3-8b-2xh100.toml:v1"
notes = "Speed optimizations: reduced max_tokens, rollouts, added prefix caching"
```

**Required fields:**
- `version`: Semantic version (v1, v2, etc.) - bump when changing any training param
- `created`: ISO date of this version
- `env_version`: BeautifulSoup env version (from pyproject.toml)
- `based_on`: Parent config:version (for tracking lineage)
- `notes`: Human-readable description of what changed and WHY

### Config Registry

`configs/registry.json` tracks all config versions:

```json
{
  "active_configs": {
    "qwen3-8b-2xh100": "configs/prime-rl/qwen3-8b-2xh100.toml:v2"
  },
  "configs": {
    "configs/prime-rl/qwen3-8b-2xh100.toml": {
      "model": "Qwen/Qwen3-8B",
      "hardware": "2x H100",
      "current_version": "v2",
      "versions": {
        "v1": {"created": "2026-01-04", "notes": "Initial config"},
        "v2": {"created": "2026-01-05", "notes": "Speed optimizations"}
      }
    }
  }
}
```

### Best Practices

1. **Version bump**: Change version when modifying ANY training parameter
2. **Document WHY**: Always explain reasoning in notes field
3. **Update registry**: Keep `configs/registry.json` in sync
4. **Atomic commits**: One config change = one commit
5. **Rollback via git**: `git show v1:configs/prime-rl/config.toml > config.toml`

### Key Config Parameters for Speed

| Setting | Default | Optimized | Impact |
|---------|---------|-----------|--------|
| `max_tokens` | 10000 | 4096 | 2.5x faster |
| `rollouts_per_example` | 8 | 4 | 2x faster |
| `oversampling_factor` | 2.0 | 1.0 | 2x faster |
| `enable_prefix_caching` | false | true | 10-20% faster |
| `max_num_seqs` | 256 | 512 | Better scheduling |
| `max_num_batched_tokens` | default | 8192 | Better throughput |

---

*Last updated: 2026-01-06 (added step counter desync bug documentation)*
