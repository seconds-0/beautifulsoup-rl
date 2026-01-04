# BeautifulSoup RL Benchmark Skill

Run and analyze LLM benchmarks on the BeautifulSoup RL environment.

## Trigger Phrases
- "run benchmark"
- "benchmark model"
- "eval on"
- "test model"
- "check model performance"

## Available Models (OpenRouter)

### Frontier Models
| Model | ID | Notes |
|-------|----|----|
| INTELLECT-3 | `prime-intellect/intellect-3` | **Primary ceiling** - 106B MoE, cheap ($0.20/$1.10) |
| GLM-4.7 | `z-ai/glm-4.7` | Good open-source, fast |
| Kimi K2 | `moonshotai/kimi-k2` | Strong but slow |

### Small Models (RL Training Targets)
| Model | ID | Notes |
|-------|----|----|
| Ministral 3B | `mistralai/ministral-3b` | Fast, good baseline |
| Qwen3-8B | `qwen/qwen3-8b` | Slower, struggles with limits |

## Running Benchmarks

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

## Key Files

| File | Purpose |
|------|---------|
| `bs4_env/scripts/eval_with_llm.py` | Main evaluation script |
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

### "0 calls" Failures
Model didn't make any tool calls. Check:
1. Is the model ID correct?
2. Is the API working?
3. Check `final_output` for error messages

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
| `--timeout` | 30s | Per-task timeout |

## Dataset Splits

| Split | Size | Purpose |
|-------|------|---------|
| train | ~12,000 | RL training |
| eval | ~500 | Development testing |
| bench | 260 | Final benchmarking |
