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

## Architecture

See [CLAUDE.md](./CLAUDE.md) for detailed architecture documentation.

## License

MIT
