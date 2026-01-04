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
- "train model", "rl training", "prime rl"
- "check model performance"
- "prime models", "trainable models", "which model for RL"
- "model sizes", "model parameters", "model pricing"

---

## Model Registry (Prime Cloud)

### Best RL Training Targets
| Model | Baseline | Params | Cost ($/1M in/out) | Notes |
|-------|----------|--------|-------------------|-------|
| `openai/gpt-oss-20b` | **63.3%** | 20B | $0.07/$0.30 | **Best target** - cheap, room to grow |
| `meta-llama/llama-4-maverick` | **65.8%** | 400B (17B active) | $0.27/$0.88 | MoE, good backup |

### Validation/Ceiling Models
| Model | Baseline | Params | Cost ($/1M in/out) | Notes |
|-------|----------|--------|-------------------|-------|
| `openai/gpt-5-nano` | **98.3%** | ? | $0.05/$0.40 | Ceiling (too good for RL) |
| `z-ai/glm-4.5-air` | **86.9%** | 106B (12B) | $0.20/$1.10 | MoE, frontier baseline |
| `mistralai/mistral-small-3.2-24b-instruct` | **82.6%** | 24B | $0.10/$0.25 | Dense, validation |
| `prime-intellect/intellect-3` | **74%** | 106B (12B) | $0.20/$1.10 | MoE, GLM-based, **PI's model** |

### Models with Issues (avoid for RL)
| Model | Baseline | Issue |
|-------|----------|-------|
| `arcee-ai/trinity-mini` | 21-34% | Inconsistent - often skips tools entirely |
| `allenai/olmo-3-7b-instruct` | 0% | Outputs code as markdown, not tool calls |

### Unavailable on Prime (404)
- `google/gemma-3-*` - Not available
- `qwen/qwen3-30b-*` - Model ID not found

---

## Full Model Registry by Size

### Nano (<3B active)
| Model | Total | Active | Price | prime-rl |
|-------|-------|--------|-------|----------|
| `arcee-ai/trinity-mini` | 26B | 3B | $0.045/$0.15 | native (Afmoe) |

### Small (3-10B active)
| Model | Total | Active | Price | prime-rl |
|-------|-------|--------|-------|----------|
| `allenai/olmo-3-7b-instruct` | 7B | 7B | $0.10/$0.20 | hf |
| `allenai/olmo-3-7b-think` | 7B | 7B | $0.12/$0.20 | hf |
| `qwen/qwen3-vl-8b-instruct` | 8B | 8B | $0.18/$0.70 | native |

### Medium (10-35B active)
| Model | Total | Active | Price | prime-rl |
|-------|-------|--------|-------|----------|
| `prime-intellect/intellect-3` | 106B | 12B | $0.20/$1.10 | native (Glm4Moe) |
| `z-ai/glm-4.5-air` | 106B | 12B | $0.20/$1.10 | native (Glm4Moe) |
| `meta-llama/llama-4-maverick` | 400B | 17B | $0.27/$0.88 | native (Llama) |
| `openai/gpt-oss-20b` | 20B | 20B | $0.07/$0.30 | hf |
| `qwen/qwen3-235b-a22b-instruct-2507` | 235B | 22B | $0.22/$0.88 | native (Qwen3Moe) |
| `mistralai/mistral-small-3.2-24b-instruct` | 24B | 24B | $0.20/$0.50 | hf |
| `google/gemma-3-27b-it` | 27B | 27B | $0.119/$0.30 | hf |
| `moonshotai/kimi-k2-0905` | 1T | 32B | $1.20/$5.00 | hf |

### Large (35-100B active)
| Model | Total | Active | Price | prime-rl |
|-------|-------|--------|-------|----------|
| `deepseek/deepseek-v3.2` | 671B | 37B | $0.28/$0.42 | hf |
| `mistralai/mistral-large-2512` | 675B | 41B | $0.50/$1.50 | hf |
| `meta-llama/llama-3.1-70b-instruct` | 70B | 70B | $0.90/$0.90 | native |
| `qwen/qwen-2.5-72b-instruct` | 72B | 72B | $0.38/$0.40 | hf |

### Closed APIs (Inference Only, NOT Trainable)
| Vendor | Models | Notes |
|--------|--------|-------|
| `anthropic/` | Claude 3.5-4.5 | Haiku, Sonnet, Opus |
| `openai/` | GPT-4o to 5.2 | Except gpt-oss-* |
| `google/gemini-*` | Gemini 2.0-3 | Flash, Pro |
| `x-ai/` | Grok 3-4 | |

**Exception:** `openai/gpt-oss-20b` and `openai/gpt-oss-120b` are open-source and trainable.

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

## Prime RL Training

### Config File
`configs/prime-rl/beautiful-soup-env.toml`

### Recommended Default Config

```toml
# === MODEL ===
[model]
name = "openai/gpt-oss-20b"  # Best RL target: 63.3% baseline, cheap

# === TRAINING ===
max_steps = 50                # Smoke test. Production: 1000+

[trainer.optim]
lr = 1e-5                     # Conservative LR for LoRA
weight_decay = 0.0

[trainer.model.experimental.lora]
rank = 8                      # LoRA rank (8-16 typical)
alpha = 32                    # LoRA alpha (usually 2-4x rank)
dropout = 0.0

# === ORCHESTRATOR ===
[orchestrator]
batch_size = 256              # Rollouts per batch
rollouts_per_example = 8      # Completions per prompt
seq_len = 4096                # Context window

[orchestrator.sampling]
max_tokens = 10000            # CRITICAL: Must be high for multi-turn tool calling
temperature = 0.7             # Exploration vs exploitation
top_p = 0.95

[orchestrator.buffer]
type = "online-difficulty"    # Prioritize harder examples
oversampling_factor = 2.0

# === ENVIRONMENT ===
[[orchestrator.env]]
id = "seconds-0/beautiful-soup-env"

[orchestrator.env.args]
split = "train"
mode = "tiered"               # Overweights harder archetypes
difficulty = "mixed"
executor_backend = "prime"    # Sandboxed execution
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

### Running Training (Requires GPU Pod)

Training requires GPU infrastructure. Create a Prime Cloud pod and run there:

```bash
# 1. Check available GPU resources
prime availability list

# 2. Create a 2-GPU pod (need inference + trainer GPUs)
prime pods create
# Select: 2x A100 80GB or 2x H100 for best performance

# 3. SSH into the pod
prime pods ssh <pod-id>

# 4. Clone repo and install
git clone https://github.com/seconds-0/beautifulsoup-rl.git
cd beautifulsoup-rl
uv sync

# 5. Run training with verifiers rl command
uv run rl \
  --trainer @ configs/prime-rl/beautiful-soup-env.toml \
  --orchestrator @ configs/prime-rl/beautiful-soup-env.toml \
  --inference @ configs/prime-rl/beautiful-soup-env.toml \
  --trainer-gpus 1 --inference-gpus 1
```

**Note**: The config uses `inference_gpu_ids = [0]` and `trainer_gpu_ids = [1]`, requiring at least 2 GPUs.

### Bootstrap Strategy (0% Models)
For models with 0% baseline, use staged training:

1. **Phase 1**: `mode="bootstrap"`, `difficulty="primer"` - 0% to 10%
2. **Phase 2**: `mode="bootstrap"` - 10% to 30%
3. **Phase 3**: `mode="tiered"` - 30% to 50%
4. **Phase 4**: `mode="all"` - 50% to 65%

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

## Dataset Splits

| Split | Size | Purpose |
|-------|------|---------|
| train | ~12,000 | RL training |
| eval | ~500 | Development testing |
| bench | 260-1040 | Final benchmarking |

---

*Last updated: 2026-01-04*
