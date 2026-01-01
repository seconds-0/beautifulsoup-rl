# BeautifulSoup RL Environment

An RL environment for training and evaluating agents on BeautifulSoup HTML parsing tasks. Built for [Prime Intellect's Environments Hub](https://docs.primeintellect.ai/verifiers/environments).

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

# With Verifiers support
pip install -e ".[all]"
```

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
prime env eval beautiful_soup_env -m meta-llama/llama-3.1-70b-instruct -n 100

# With specific environment config
prime env eval beautiful_soup_env \
  -a '{"split":"bench","mode":"mvp"}' \
  -m <model-name> \
  -n 100
```

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
