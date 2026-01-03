# BeautifulSoup RL Environment

An RL environment for training and evaluating agents on BeautifulSoup HTML parsing tasks. Built for [Prime Intellect's Environments Hub](https://docs.primeintellect.ai/verifiers/source/environments).

**Naming:**
- **Hub name**: `seconds-0/beautiful-soup-env` (use in `prime env eval`)
- **Python module**: `beautiful_soup_env` (use in `from beautiful_soup_env import ...`)

## Overview

This environment trains agents to:
- Extract data correctly from messy, malformed HTML using BeautifulSoup
- Avoid common BS4 API gotchas (`.string` returning None, reserved word `class`, etc.)
- Recognize when static parsing is impossible and abstain with evidence
- Respect safety boundaries (no credential extraction)

## Installation

```bash
# Basic installation
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# With all extras (dev, verifiers, prime, eval)
pip install -e ".[all]"
```

### Optional Dependencies

| Extra | Purpose | When to Use |
|-------|---------|-------------|
| `.[dev]` | pytest, ruff, mypy | Local development and testing |
| `.[eval]` | OpenAI client | Running LLM evaluations |
| `.[prime]` | Prime sandboxes | Production execution on Prime |
| `.[verifiers]` | Verifiers framework | Integration with Prime's RL pipeline |
| `.[all]` | All of the above | Full development environment |

## Quick Start

```python
from beautiful_soup_env import load_environment

# Load the environment (defaults to bench split for fast loading)
env = load_environment()

# Or with custom config
env = load_environment(
    split="train",       # Use "bench" for quick testing, "train" for RL training
    mode="mvp",
    difficulty="mixed",
    executor_backend="local"
)

# Get a task
task = env.get_example()
print(task["prompt"])
```

## Task Types

### Extraction Tasks
The agent extracts data from HTML using BeautifulSoup code. Graded against deterministic ground truth.

```json
{
  "status": "ok",
  "answer": "The extracted text"
}
```

### Limitation Tasks
Some content is intentionally unparseable (JS-rendered, image-based, etc.). The agent must recognize this and abstain with evidence.

```json
{
  "status": "limit",
  "limit": {
    "reason": "js_required",
    "evidence": "<script>renderContent()"
  }
}
```

## Reward Structure

| Outcome | Reward |
|---------|--------|
| Correct extraction | +1.0 |
| Correct limitation (with evidence) | +0.5 |
| Wrong answer | 0.0 |
| Safety violation | -0.5 |

## Model Training Results

Baseline evaluation results on 52 archetypes (1040 bench examples). See [TEST_RECORDS.md](TEST_RECORDS.md) for full details.

### Benchmark Calibration

| Tier | Model | Pass Rate | Perfect Rate | RL Target? |
|------|-------|-----------|--------------|------------|
| Small | Qwen3-4B | 0% | 0% | Ideal |
| Small | Llama 3.2-3B | 0% | 0% | Ideal |
| Medium | Ministral-3B | 50.6% | 19.7% | Good |
| Medium | Qwen3-8B | 43.1% | 26.6% | Good |
| Medium | Ministral-8B | 68.4% | 57.9% | Validation |
| Large | Kimi K2 | 72.8% | 45.0% | Ceiling |
| Large | GLM-4.7 | 74.7% | 45.1% | Ceiling |

**Key Insight:** 0% → 50% → 75% progression shows clear learning signal for RL training. Small models (3-4B) start at 0% but have function calling support, making them ideal RL training targets.

### Archetype Difficulty Distribution

| Category | Example Archetypes | 3B Pass Rate |
|----------|-------------------|--------------|
| Easy | `extract_images`, `extract_links`, `select_options` | 90%+ |
| Medium | `json_ld_extraction`, `table_column_by_header` | 50-80% |
| Hard | `semantic_decoy`, `aggregation_min_max` | 20-50% |
| Very Hard | Multi-step navigation, limitation detection | 0-20% |

## PRIME-RL Training Configuration

### Recommended Training Config

```python
from beautiful_soup_env import load_environment

# Training configuration for PRIME-RL
env = load_environment(
    split="train",              # 1000 seeds per archetype
    mode="all",                 # All 52 archetypes (phase 1 + phase 2)
    difficulty="mixed",         # Mix all difficulties
    executor_backend="prime",   # Use Prime's sandboxed executor
    network_access=False,       # Determinism + safety
    timeout_s=30.0,
    seed=42,                    # Reproducibility
)

# Alternative: Use mode="mvp" for 29 core phase-1 archetypes only
# Alternative: Use mode="tiered" for difficulty-weighted sampling
```

### Key Training Parameters

| Parameter | Training | Evaluation | Benchmark |
|-----------|----------|------------|-----------|
| `split` | `"train"` | `"eval"` | `"bench"` |
| Seed range | 0-100k | 100k-110k | 110k+ (fixed) |
| Examples/archetype | 1000 | 100 | 20 |

### Mode Options

| Mode | Archetypes | Description |
|------|------------|-------------|
| `"mvp"` | 29 | Phase 1 core archetypes (production-ready) |
| `"phase2"` | 23 | Phase 2 archetypes (advanced) |
| `"all"` | 52 | All archetypes, uniform sampling |
| `"tiered"` | 52 | All archetypes with difficulty-weighted sampling |
| `"hard_only"` | 18 | Only hard difficulty archetypes |

### Bench Split Behavior

The `bench` split uses a **fixed manifest** for reproducibility across runs:
- The `mode` parameter is **ignored** for bench - all 52 archetypes are included regardless of mode setting
- `archetype_filter` and `difficulty` filters still apply if specified
- Tasks are loaded from `bs4_env/data/bench_manifest.json` (20 seeds × 52 archetypes = 1040 tasks)
- This ensures benchmark scores are comparable across environment versions

## Running Evaluations

### Local Testing
```bash
# Preview dataset
python -m bs4_env.scripts.preview_dataset

# Smoke test
python -m bs4_env.scripts.smoke_eval_local
```

### Prime Evaluation
```bash
# Run benchmark evaluation
prime env eval seconds-0/beautiful-soup-env -m meta-llama/llama-3.1-70b-instruct -n 100

# With specific environment config
prime env eval seconds-0/beautiful-soup-env \
  -a '{"split":"bench","mode":"mvp"}' \
  -m <model-name> \
  -n 100
```

### Prime Sandbox Mode

When running on Prime's infrastructure, the environment uses sandboxed execution:

```python
env = load_environment(
    executor_backend="prime",  # Uses Prime's sandboxed executor
    network_access=True,       # Required to install bs4/lxml in default image
)
```

**Sandbox Configuration:**
- **Default image**: Uses `python:3.11-slim` with `network_access=True` to pip install dependencies at runtime
- **Prebuilt image (recommended for production)**: Use a custom Docker image with dependencies pre-installed and set `network_access=False`
- Code execution timeout defaults to 30 seconds
- A starter training config is available at `configs/prime-rl/beautiful-soup-env.toml`

### Prebuilt Image Strategy (for `network_access=False`)

For deterministic training and security-isolated execution, you need a prebuilt Docker image since runtime `pip install` requires network access.

**Required dependencies:**
```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir beautifulsoup4 lxml html5lib
```

**Using the prebuilt image:**
```python
env = load_environment(
    executor_backend="prime",
    network_access=False,      # No runtime network access
    docker_image="your-registry/bs4-prebuilt:latest",  # Your prebuilt image
)
```

**Benefits of prebuilt images:**
- **Determinism**: No network variability or version drift from pip installs
- **Speed**: No dependency installation overhead per sandbox
- **Security**: Complete network isolation during code execution
- **Reliability**: No failures from network issues or package availability

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=bs4_env

# Type checking
mypy bs4_env

# Linting
ruff check bs4_env
```

## Environment Card

### Skill Description

This environment trains agents to use BeautifulSoup (BS4), Python's most popular HTML parsing library. Tasks cover real-world web scraping scenarios including:

- **Core extraction**: Text, attributes, and structured data from HTML elements
- **Traversal**: Navigating DOM trees, sibling relationships, parent/child
- **Tables**: Complex tables with rowspan/colspan, nested tables
- **Forms**: Input fields, labels, validation attributes
- **Structured data**: JSON-LD, microdata, Open Graph metadata
- **Error handling**: Malformed HTML, missing elements, API gotchas
- **Internationalization**: Unicode, RTL text, CJK characters
- **Limitation detection**: Recognizing when static parsing is impossible

### Task Distribution

| Metric | Count |
|--------|-------|
| **Total Archetypes** | 52 |
| **Solvable Tasks** | 47 |
| **Limitation Tasks** | 5 |

**By Difficulty:**
| Easy | Medium | Hard |
|------|--------|------|
| 10 | 24 | 18 |

**By Category:**
| Category | Count | Description |
|----------|-------|-------------|
| hard | 10 | Complex extraction requiring multiple techniques |
| advanced | 8 | Advanced BS4 features and edge cases |
| forms | 6 | Form parsing and input extraction |
| core_extraction | 5 | Basic text and attribute extraction |
| limitations | 5 | Tasks requiring abstention with evidence |
| table_parsing | 4 | Table structure and cell extraction |
| i18n | 3 | Internationalization and encoding |
| bs4_gotchas | 3 | Common API pitfalls and mistakes |
| error_bait | 3 | Tasks designed to trigger common errors |
| traversal | 2 | DOM navigation and relationships |
| structured_data | 2 | JSON-LD, microdata extraction |
| output_normalization | 1 | Format normalization tasks |

### Benchmark Stability

The benchmark split uses a fixed manifest (`bs4_env/data/bench_manifest.json`) containing 1040 pre-selected (archetype, seed) pairs. This ensures:

- **Reproducibility**: Same tasks across runs and environment versions
- **Isolation**: Adding/removing archetypes doesn't affect existing benchmark tasks
- **Versioning**: Manifest version tracks benchmark changes (currently v1.2.0)

### Baseline Results

See [TEST_RECORDS.md](./TEST_RECORDS.md) for current model evaluation results.

## Architecture

See [CLAUDE.md](./CLAUDE.md) for detailed architecture documentation.

## License

MIT
