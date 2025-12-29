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
  prompt.py               # Format prompts (no label leakage!)
  generators/
    base.py               # Generator protocol, TaskInstance, RNG utils
    mvp_*.py              # Archetype implementations
  grading/
    schema.py             # JSON schema validation
    normalize.py          # Deterministic normalization
    safety.py             # Credential/token detection
    rubric.py             # Reward computation
  tools/
    executor.py           # LocalSubprocessExecutor, PrimeSandboxExecutor
    harness.py            # Injected globals (HTML, QUERY, CONSTRAINTS)
  adapters/
    verifiers_adapter.py  # Wire to vf.Environment
```

## Critical Rules

### 0. Test How You Measure (Production Quality)
**Local testing must match Prime's production flow exactly.** No shortcuts.

- If Prime uses function calling, our local eval uses function calling
- If Prime uses specific tool schemas, we use the same schemas
- Our local results should predict actual Prime performance

This ensures we're building production-quality tooling, not approximations that break in deployment.

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

### 4. Local-First Development
Everything must work with `LocalSubprocessExecutor` before testing with Prime:
```python
config = EnvConfig(executor_backend="local")
```

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

## Running Local Evaluation

```bash
# Preview dataset (sanity check)
python -m bs4_env.scripts.preview_dataset

# End-to-end local smoke test
python -m bs4_env.scripts.smoke_eval_local

# LLM evaluation via OpenRouter
uv run python -m bs4_env.scripts.eval_with_llm --model <model> --num 50
```

## Test Records

**Always update `TEST_RECORDS.md` after running evaluations.**

This file tracks:
- Which models have been tested
- Pass rates by archetype
- Blocked models (API issues, rate limits)
- Environment changes that affect results

Target **small/weak models** for testing - they benefit most from RL training.

## Running Prime Evaluation

```bash
# Baseline evaluation
prime env eval --env beautiful_soup_env \
  --env-args '{"split":"bench","mode":"mvp"}' \
  --model <model-name>

# Push to Hub
prime env push
```

## File Ownership

| File | Purpose | When to Modify |
|------|---------|----------------|
| `config.py` | Environment configuration | Adding new config options |
| `registry.py` | Archetype registration | Rarely - stable interface |
| `generators/base.py` | Generator protocol | Rarely - stable interface |
| `generators/mvp_*.py` | Archetype implementations | Adding new archetypes |
| `grading/rubric.py` | Reward computation | Adjusting reward logic (CAREFUL) |
| `grading/normalize.py` | Output normalization | Adding normalization rules (CAREFUL) |
| `adapters/verifiers_adapter.py` | Verifiers integration | Wiring to Prime |

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

## Dependencies

Core:
- `beautifulsoup4` - The library we're training on
- `lxml` - Fast parser backend
- `html5lib` - Lenient parser backend
- `verifiers` - Prime's environment framework
- `datasets` - HuggingFace datasets

Dev:
- `pytest` - Testing
- `pytest-cov` - Coverage

## Git Workflow

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
