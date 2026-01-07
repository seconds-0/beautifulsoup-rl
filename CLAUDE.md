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

### 0. Always Check .env First
**Before setting up credentials or environment variables, ALWAYS check `.env` in the project root.**

This file contains:
- API keys (PRIME_API_KEY, OPENROUTER_API_KEY, WANDB_API_KEY, VAST_API_KEY)
- Cloud storage credentials (B2_APPLICATION_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET)
- Training configuration (RUN_ID, VLLM_* settings)

Don't ask the user for credentials that are already in `.env`.

### 1. Prime-First Testing (Dogfood on Production)
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

### 2. Never Derive Ground Truth by Parsing
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

### 3. Deterministic Generation
All generators must be deterministic given a seed:
```python
def generate(self, seed: int) -> TaskInstance:
    rng = make_rng(self.archetype_id, seed)
    # Use rng for ALL randomness
```

Use `stable_int_seed()` (SHA-256 based), NOT Python's `hash()` which is salted.

### 4. Test Every Archetype
Each archetype needs:
- Determinism test (same seed → same output)
- Grading test (correct answer → +1.0, wrong answer → 0.0)

### 5. Local Development, Prime Validation
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

### 6. Normalization is Dangerous
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

## Prime RL Training

**See `.claude/skills/benchmark/SKILL.md` for full documentation.**

### Critical Rules

1. **Environment installation**: MUST use `prime env install seconds-0/beautiful-soup-env` on pod. `pip install -e .` does NOT work!

2. **Config validation**:
   - `trainer.model.seq_len` >= `orchestrator.seq_len`
   - LoRA: `[trainer.model.lora]` NOT `[trainer.model.experimental.lora]`
   - `lora_name` required under `[orchestrator]`
   - NO `top_p`, `mask_truncated_completions`, `zero_truncated_completions`
   - Buffer: `online_difficulty_filtering = true` NOT `type = "online-difficulty"`

3. **Model selection**: Must be on HuggingFace, NOT VL models, compatible with vLLM

4. **Commands**:
   ```bash
   prime env install seconds-0/beautiful-soup-env  # REQUIRED on pod
   uv run rl @ config.toml                         # Start training
   ```

### Known Model Compatibility Issues

| Model | Status | Issue |
|-------|--------|-------|
| gpt-oss-20b | BLOCKED | vLLM weight reload bug |
| qwen3-vl-* | BLOCKED | VL model, wrong class |
| Ministral-3-8B | BLOCKED | transformers KeyError |
| **Qwen2.5-7B-Instruct** | **WORKS** | Use with hermes parser |

### Required Environment Variables (on pod)

```bash
export VLLM_USE_V1=0                      # Disable V1 engine (has LoRA issues)
export VLLM_WORKER_MULTIPROC_METHOD=spawn # Prevent CUDA context inheritance
export WANDB_API_KEY=<key>                # For logging
```

**Finding API Keys**: Always check `.env` file in the repo root and `~/.netrc` for stored credentials before asking the user. Keys for this project:
- `WANDB_API_KEY` → `.env` and GitHub secrets
- `B2_APPLICATION_KEY_ID`, `B2_APPLICATION_KEY` → GitHub secrets
- `VAST_API_KEY` → GitHub secrets

### Single-GPU Training Constraints

| Setting | Value | Why |
|---------|-------|-----|
| `[inference] gpu_memory_utilization` | `0.50` | Leave room for trainer (default 0.9 will OOM) |
| `[trainer.model.ac] freq` | `1` | Activation checkpointing saves ~20GB |
| `[orchestrator.sampling] max_tokens` | `2000` | Must be < max_model_len - 1500 |
| `fsdp_cpu_offload` | **NEVER with LoRA** | Uses 1.65x MORE memory with LoRA! |

### Checkpointing (CRITICAL)

**Checkpointing is OFF by default in prime-rl!** Always enable it:

```bash
uv run rl @ config.toml --ckpt --ckpt.interval 5 --ckpt.keep-last 3
```

Resume from checkpoint:
```bash
uv run rl @ config.toml --ckpt.resume-step 20 --max-steps 50
```

### Resilient Training (Spot Instances)

**See `TRAINING_RUNS.md` > "Resilient Training System" for full docs.**

For spot instance training with auto-recovery:

| Script | Purpose |
|--------|---------|
| `scripts/pod_setup.sh` | Sets up pod with B2 CLI, checkpoint sync |
| `scripts/checkpoint_sync.sh` | Syncs checkpoints to Backblaze B2 every 5 min |
| `scripts/onstart.sh` | Auto-resumes from B2 checkpoint (Vast.ai) |
| `scripts/wandb_monitor.py` | Check training health from WandB |
| `scripts/provision_vast.py` | Provision/terminate Vast.ai instances |
| `.github/workflows/training-monitor.yml` | Auto-recovery (checks every 10 min) |

**Quick start:**
```bash
# On pod (sets up B2 sync automatically)
curl -sSL https://raw.githubusercontent.com/seconds-0/beautifulsoup-rl/main/scripts/pod_setup.sh | bash

# Start training with checkpointing
uv run rl @ config.toml --ckpt --ckpt.interval 5 --ckpt.keep-last 3
```

**GitHub Secrets required for auto-recovery:** `WANDB_API_KEY`, `VAST_API_KEY`, `B2_APPLICATION_KEY_ID`, `B2_APPLICATION_KEY`

### Config Version Tracking

**All training configs must be versioned.** This enables rollback, comparison, and reproducibility.

**Config metadata header** (required in all `.toml` configs):
```toml
[config]
version = "v2"
created = "2026-01-05"
env_version = "0.1.1"
based_on = "qwen3-8b-2xh100.toml:v1"
notes = "Speed optimizations: reduced max_tokens, added prefix caching"
```

**Registry file** (`configs/registry.json`):
- Tracks active config per model/hardware combo
- Version history with notes
- Update when changing active config

**Rules:**
1. Bump version when changing ANY training parameter
2. Document WHY in notes field
3. Update `registry.json` when switching active configs
4. Commit config changes atomically (one config = one commit)
5. Never delete old versions from registry (history is valuable)

**Key speed parameters** (tune these first):
| Parameter | Impact | Notes |
|-----------|--------|-------|
| `max_tokens` | High | Reduce to match actual needs (4k vs 10k = ~2.5x faster) |
| `rollouts_per_example` | High | Fewer rollouts = faster steps (8→4 = 2x faster) |
| `oversampling_factor` | Medium | 1.0 = no oversampling (fastest) |
| `enable_prefix_caching` | Medium | Reuses KV cache for shared prompts |

### Pre-Training Checklist

Run before creating a pod:
```bash
python scripts/verify_model_for_training.py <model-name>
```

On pod setup:
```bash
bash scripts/pod_setup.sh
```

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
