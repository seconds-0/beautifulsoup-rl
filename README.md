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

# Load the environment
env = load_environment(
    split="train",
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
prime env eval --env beautiful_soup_env \
  --env-args '{"split":"bench","mode":"mvp"}' \
  --model <model-name>
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
| **Total Archetypes** | 53 |
| **Solvable Tasks** | 48 |
| **Limitation Tasks** | 5 |

**By Difficulty:**
| Easy | Medium | Hard |
|------|--------|------|
| 10 | 24 | 19 |

**By Category:**
| Category | Count | Description |
|----------|-------|-------------|
| hard | 11 | Complex extraction requiring multiple techniques |
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

The benchmark split uses a fixed manifest (`bs4_env/data/bench_manifest.json`) containing 1060 pre-selected (archetype, seed) pairs. This ensures:

- **Reproducibility**: Same tasks across runs and environment versions
- **Isolation**: Adding/removing archetypes doesn't affect existing benchmark tasks
- **Versioning**: Manifest version tracks benchmark changes (currently v1.1.0)

### Baseline Results

See [TEST_RECORDS.md](./TEST_RECORDS.md) for current model evaluation results.

## Architecture

See [CLAUDE.md](./CLAUDE.md) for detailed architecture documentation.

## License

MIT
