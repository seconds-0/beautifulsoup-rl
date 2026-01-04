# BeautifulSoup RL Environment

## Project Overview

This is an RL environment for Prime Intellect's Environments Hub bounty. It trains and evaluates agents on BeautifulSoup (BS4) parsing tasks - extracting data from messy HTML, avoiding API gotchas, and recognizing when static parsing is impossible.

**Target**: Prime Intellect Environments Hub Bounty (Software Library Evals: BeautifulSoup)

## Key Concepts

### Task Types
1. **Extraction tasks** (`solvable=True`): Model extracts data using BS4, graded against ground truth. Reward: +1.0 correct, 0.0 wrong.
2. **Limitation tasks** (`solvable=False`): Content is unparseable (JS-rendered, etc). Model must abstain with evidence. Reward: +0.5 with valid evidence, 0.0 otherwise.

### Anti-Hacking Rules
- If `solvable=True` and model claims "limit" → 0.0 (can't always-abstain for free points)
- If `solvable=False` and model claims "limit" → +0.5 ONLY if:
  - Reason is in allowed list for that task
  - Evidence is a literal substring found in the HTML

### Output Format
```json
{
  "status": "ok" | "limit",
  "answer": <matches task schema> | null,
  "limit": {"reason": "js_required", "evidence": "<script>renderContent()"} | null
}
```

## Architecture

```
beautiful_soup_env.py      # load_environment() entrypoint
bs4_env/
  config.py               # EnvConfig dataclass
  registry.py             # @register decorator, archetype specs
  dataset.py              # Build HF dataset with train/eval/bench splits
  lazy_dataset.py         # Memory-efficient LazyBS4Dataset with LRU caching
  auto_import.py          # Registry initialization (archetype loading)
  prompt.py               # Format prompts (no label leakage!)
  generators/
    base.py               # Generator protocol, TaskInstance, RNG utils
    mvp_*.py              # Archetype implementations
  grading/
    schema.py             # JSON schema validation
    normalize.py          # Deterministic normalization
    safety.py             # Credential/token detection
    rubric.py             # Reward computation, efficiency multiplier, process credit
  tools/
    executor.py           # LocalSubprocessExecutor, PrimeSandboxExecutor, PooledSubprocessExecutor
    harness.py            # Injected globals (HTML, QUERY, CONSTRAINTS)
  adapters/
    verifiers_adapter.py  # Wire to vf.Environment
  data/
    bench_manifest.json   # Fixed 52-archetype bench split for reproducibility
```

## Critical Rules

### 0. Prime-First Testing (Dogfood on Production)
**Test on Prime's actual infrastructure whenever possible.** Local testing is for rapid iteration only.

**Why Prime-first:**
- Local eval uses different settings than Prime (e.g., `max_tokens=4000` local vs `768` Prime)
- Local results may not predict actual Prime performance
- Subtle differences in tool calling, timeouts, and sandboxing can cause surprises
- We're building for Prime's RL pipeline, so test there

**When to use local eval:**
- Rapid iteration during development (syntax errors, basic logic)
- When Prime credits are limited
- Testing specific edge cases with verbose output

**When to use Prime eval (preferred):**
- Validating model performance (the real metric)
- Benchmarking before/after changes
- Final verification before merging

**Config alignment check:**
```toml
# configs/prime-rl/beautiful-soup-env.toml
[orchestrator.sampling]
max_tokens = 4000       # Matches local eval default
temperature = 0.7
```

If local eval shows different results than Prime, check config alignment first.

### 1. Never Derive Ground Truth by Parsing
Ground truth must come from structured data BEFORE HTML rendering. Never do this:
```python
# WRONG - label leakage
html = generate_html()
ground_truth = BeautifulSoup(html).find(...).text  # NO!
```

Do this instead:
```python
# CORRECT
data = {"title": "Example", "price": 99.99}
html = render_to_html(data)  # Render data to HTML
ground_truth = data["title"]  # Ground truth from source data
```

### 2. Deterministic Generation
All generators must be deterministic given a seed:
```python
def generate(self, seed: int) -> TaskInstance:
    rng = make_rng(self.archetype_id, seed)
    # Use rng for ALL randomness
```

Use `stable_int_seed()` (SHA-256 based), NOT Python's `hash()` which is salted.

### 3. Test Every Archetype
Each archetype needs:
- Determinism test (same seed → same output)
- Grading test (correct answer → +1.0, wrong answer → 0.0)

### 4. Local Development, Prime Validation
Use local executor for development, but validate on Prime:
```python
# Development (fast iteration)
config = EnvConfig(executor_backend="local")

# Validation (matches production)
prime env eval seconds-0/beautiful-soup-env -m <model> -n 50
```

**Local eval limitations:**
- Uses OpenRouter API (different from Prime's vLLM inference)
- Both now use `max_tokens=4000` (aligned)
- No sandboxing (security behaviors may differ)
- Token counting may differ from Prime's tokenizer

### 5. Normalization is Dangerous
Start with strict exact-match. Only add normalization when demonstrably needed. Over-normalization enables reward hacking.

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_grading.py

# Run with coverage
pytest --cov=bs4_env
```

## Running Evaluations

### Prime Evaluation (Preferred)
Use Prime for real performance metrics - this is what matters:
```bash
# Quick validation (50 examples)
prime env eval seconds-0/beautiful-soup-env \
  -a '{"split":"bench","mode":"mvp"}' \
  -m <model-name> \
  -n 50

# Full benchmark
prime env eval seconds-0/beautiful-soup-env \
  -a '{"split":"bench","mode":"all"}' \
  -m <model-name> \
  -n 680
```

### Local Evaluation (Development Only)
Use local eval for rapid iteration, not for final metrics:
```bash
# Preview dataset (sanity check)
python -m bs4_env.scripts.preview_dataset

# End-to-end local smoke test
python -m bs4_env.scripts.smoke_eval_local

# LLM evaluation via OpenRouter (NOTE: settings differ from Prime!)
uv run python -m bs4_env.scripts.eval_with_llm --model <model> --num 50

# Match Prime's max_tokens for closer alignment
uv run python -m bs4_env.scripts.eval_with_llm --model <model> --num 50 --max-tokens 768
```

**⚠️ Local vs Prime differences:**
- Prime uses `max_tokens=768`, local defaults to `4000`
- Prime uses vLLM inference, local uses OpenRouter
- Results may not correlate - always validate on Prime

## Test Records

**Always update `TEST_RECORDS.md` after running evaluations.**

This file tracks:
- Which models have been tested
- Pass rates by archetype
- Blocked models (API issues, rate limits)
- Environment changes that affect results

Target **small/weak models** for testing - they benefit most from RL training.

## File Ownership

| File | Purpose | When to Modify |
|------|---------|----------------|
| `config.py` | Environment configuration | Adding new config options |
| `registry.py` | Archetype registration | Rarely - stable interface |
| `generators/base.py` | Generator protocol | Rarely - stable interface |
| `generators/mvp_*.py` | Archetype implementations | Adding new archetypes |
| `grading/rubric.py` | Reward computation, efficiency, process credit | Adjusting reward logic (CAREFUL) |
| `grading/normalize.py` | Output normalization | Adding normalization rules (CAREFUL) |
| `adapters/verifiers_adapter.py` | Verifiers integration | Wiring to Prime |
| `tests/test_solvability.py` | Archetype solvability regression | Add tests for new archetypes |
| `tests/test_efficiency.py` | Efficiency multiplier tests | Adjust when efficiency rules change |
| `tests/test_process_partial_credit.py` | Process credit for 0% models | Adjust when credit tiers change |

## Common Gotchas

### BS4 Gotchas We Test
- `.string` returns `None` when element has multiple children → use `get_text()`
- `class` is reserved word → use `class_` or `attrs={"class": ...}`
- Missing elements return `None` → check before accessing `.text` or attributes
- Different parsers produce different DOM trees for malformed HTML

### Development Gotchas
- Python's `hash()` is not deterministic across runs (salted) → use SHA-256
- HuggingFace datasets are lazy → force evaluation in tests
- Subprocess execution needs timeout handling

### Grading Gotchas
- **Efficiency multiplier**: Penalizes excessive tool calls (>10 calls = 0.0 reward by default)
- **Weighted tool counts**: `navigate` costs 0.2x, `run_python` costs 1.0x for soft efficiency gradient
- **Hard cap uses raw count**: `tool_call_count_raw` for hard caps, `tool_call_count` for soft gradient
- **Archetype-aware limits**: Check `task_info["metadata"]["constraints"]["max_tool_calls"]`
- **Process partial credit**: Capped at 0.30, uses AST analysis of code samples
- **Sandbox dependency check**: `PrimeSandboxExecutor` verifies bs4/lxml/html5lib on first run

## Git Workflow

### Git Author (REQUIRED)
All commits must use the **seconds-0** GitHub account:
```bash
git config user.name "seconds-0"
git config user.email "36005888+seconds-0@users.noreply.github.com"
```

A pre-commit hook enforces this - commits will be rejected if the wrong author is configured.

**Do NOT use** the huth-stacks account for this repository.

### Atomic Commits (Mandatory)
Each commit should be ONE coherent change for reviewability. Commit after completing each logical unit, not at the end of a phase.

**Good examples:**
- "Add config loader utility" (one file, one purpose)
- "Add retry logic to executor" (single feature)
- "Fix token estimation for long messages" (one bug fix)

**Bad examples:**
- "Phase 2 - implement grading pipeline" (20 files, multiple features)
- "Add schema and normalize modules" (two separate concerns)
- "Various fixes and improvements" (meaningless batching)

**Rule of thumb:** If the commit message needs "and" or a list, split it.

### Other Git Rules
- Run tests before committing
- No force pushes to main
- **CRITICAL: No AI attribution in commits**
  - Do NOT add "Co-Authored-By: Claude" or any Anthropic attribution
  - Do NOT add "Generated with Claude Code" footers
  - Keep commit messages clean and professional
  - This is enforced by a commit-msg hook
