# BeautifulSoup RL Benchmark Skill

Run benchmarks, evaluations, and RL training on the BeautifulSoup RL environment.

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

## Preferred Models

### Prime RL Training Targets (0% baseline - maximum RL gain)
| Model | Baseline | Cost | Notes |
|-------|----------|------|-------|
| `allenai/olmo-3-7b-instruct` | 0% | Cheap | Perfect weak baseline on Prime |
| `meta-llama/llama-4-maverick` | 0% | Unknown | Weak baseline on Prime |

### Prime Validation Models (verify training works)
| Model | Pass Rate | Cost | Notes |
|-------|-----------|------|-------|
| `arcee-ai/trinity-mini` | 75.6% | $0.045/$0.15 | Best cheap RL candidate |
| `prime-intellect/intellect-3` | 67.3% | $0.20/$1.10 | Frontier (106B MoE), needs max_tokens=8000+ |
| `mistralai/mistral-small-3.2-24b-instruct` | 100% | $0.10/$0.30 | Ceiling validation |

### OpenRouter Evaluation (local testing)
| Model | Pass Rate | Cost | Notes |
|-------|-----------|------|-------|
| `openai/gpt-4o-mini` | ~95% | Cheap | Quick validation |
| `qwen/qwen3-8b` | 43-90% | $0.028/1M | Good RL candidate |
| `mistralai/ministral-8b-2512` | 68.4% | Cheap | Best 8B |
| `openai/gpt-5.2` | 75.6% | Medium | Frontier baseline |

### Blocked Models (avoid)
| Model | Issue |
|-------|-------|
| `google/gemma-3-*` | No function calling |
| `deepseek/deepseek-r1-*` | No function calling |
| `meta-llama/llama-3.2-1b-instruct` | No function calling |
| `qwen/qwen3-30b-a3b-*` | JSON decode errors |

**Model ID Differences:** Prime uses different IDs than OpenRouter. Always verify with `prime inference models` before running evals on Prime.

## Prime Evaluation (Preferred)

Use Prime's `env eval` for production-grade evaluation.

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

### Check Available Models
```bash
prime inference models
```

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
  --max-tokens 8000 \
  --output results_intellect3.json
```

### Run in Background
```bash
source .env && uv run python -m bs4_env.scripts.eval_with_llm \
  --model <model_id> \
  --num 260 \
  --output results_<name>.json 2>&1 &
```

### Run Multiple Models in Parallel
Run each in a separate background process. Monitor with:
```bash
tail -5 /tmp/claude/-Users-alexanderhuth-beautifulsoup-rl/tasks/<task_id>.output
```

## Prime RL Training

### Config File
`configs/prime-rl/beautiful-soup-env.toml`

Key settings:
| Setting | Value | Notes |
|---------|-------|-------|
| `max_tokens` | 4000 | Increased from 768 for verbose models |
| `seq_len` | 4096 | Context length |
| `batch_size` | 256 | Rollouts per batch |
| `rollouts_per_example` | 8 | Completions per prompt |
| `temperature` | 0.7 | Generation temperature |

### Running Training
```bash
uv run prime-rl @ configs/prime-rl/beautiful-soup-env.toml
```

### Bootstrap Strategy (0% Models)
For models with 0% baseline, use staged training:

1. **Phase 1**: `mode="bootstrap"`, `difficulty="primer"` - 0% to 10%
2. **Phase 2**: `mode="bootstrap"` - 10% to 30%
3. **Phase 3**: `mode="tiered"` - 30% to 50%
4. **Phase 4**: `mode="all"` - 50% to 65%

## Pre-flight Checks

### Before Running on Prime
```bash
# 1. Check config alignment
cat configs/prime-rl/beautiful-soup-env.toml | grep max_tokens
# Should show: max_tokens = 4000

# 2. Verify environment is pushed
prime env push --dry-run

# 3. List available models
prime inference models

# 4. Verify sandbox dependencies
# Docker image must have: bs4, lxml, html5lib
```

## Key Files

| File | Purpose |
|------|---------|
| `bs4_env/scripts/eval_with_llm.py` | Local evaluation script |
| `configs/prime-rl/beautiful-soup-env.toml` | Prime RL training config |
| `TEST_RECORDS.md` | Historical benchmark results |
| `results_*.json` | Individual run outputs |
| `bs4_env/grading/rubric.py` | Reward computation |
| `bs4_env/prompt.py` | Model prompts |

## Analyzing Results

### Quick Summary
```python
cat results_<name>.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'Pass rate: {data[\"pass_rate\"]*100:.1f}%')
print(f'Perfect rate: {data[\"perfect_rate\"]*100:.1f}%')
print(f'Avg reward: {data[\"avg_reward\"]:.3f}')
"
```

### Per-Archetype Breakdown
```python
cat results_<name>.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for arch, avg in sorted(data['by_archetype'].items(), key=lambda x: x[1]):
    print(f'{arch}: {avg:.2f}')
"
```

### Inspect Failures
```python
cat results_<name>.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
failures = [r for r in data['results'] if r['reward'] == 0][:5]
for r in failures:
    print(f'idx={r[\"idx\"]} arch={r[\"archetype_id\"]}')
    print(f'  final_output: {str(r.get(\"final_output\", \"\"))[:200]}')
    print(f'  metrics: {r[\"metrics\"]}')
    print()
"
```

### Check Tool History
```python
cat results_<name>.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
r = data['results'][0]  # Change index as needed
print(json.dumps(r.get('tool_history', []), indent=2)[:2000])
"
```

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| 0 tool calls | max_tokens too low | Increase to 4000+ (8000+ for intellect-3) |
| Sandbox timeout | Code too slow | Check executor timeout |
| Model not found | Wrong model ID | Verify with `prime inference models` |
| Empty response | API error | Check retry logic, increase backoff |

### "0 calls" Failures
Model didn't make any tool calls. Check:
1. Is the model ID correct?
2. Is `max_tokens` sufficient? (verbose models need 8000+)
3. Is the API working?
4. Check `final_output` for error messages

### `make_soup(HTML)` Error
Model is calling `make_soup(HTML)` instead of `make_soup()`.
- **Fix**: Updated prompt to clarify usage
- **File**: `bs4_env/prompt.py`

### Format/Schema Failures
Model returns correct value but wrong format.
- Check if coercion should be added to `normalize.py`
- Add aliases to `KEY_ALIASES` if key names differ

### Limit Detection Failures
Model fails on `limit_*` archetypes.
- Check if model understands when to claim limitation
- Verify evidence is actual HTML substring

## Prioritizing Fixes

1. **High priority**: Issues affecting >20% of examples
2. **Medium priority**: Issues affecting 5-20%
3. **Low priority**: Edge cases <5%

Focus on:
1. Prompt clarity issues (affect all models)
2. Grading strictness (affects scoring)
3. Model-specific issues (affects one model)

## Updating TEST_RECORDS.md

After each benchmark run, update `TEST_RECORDS.md` with:
- Model name and version
- Date
- Pass rate, perfect rate, avg reward
- Notable failures or issues
- Comparison to previous runs

## Benchmark Configs

| Setting | Default | Notes |
|---------|---------|-------|
| `--num` | 260 | Full bench split |
| `--split` | bench | Use eval for development |
| `--mode` | mvp | Phase 1 archetypes |
| `--max-tokens` | 4000 | Increase for verbose models |

## Dataset Splits

| Split | Size | Purpose |
|-------|------|---------|
| train | ~12,000 | RL training |
| eval | ~500 | Development testing |
| bench | 260-1040 | Final benchmarking |
