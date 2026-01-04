# Training Runs Log

Track all RL training experiments for BeautifulSoup environment.

## Active Runs

*None currently running*

---

## Completed Runs

| Run ID | Model | Start | Duration | Final Reward | Notes |
|--------|-------|-------|----------|--------------|-------|
| *none yet* | | | | | |

---

## Run Template

```markdown
### Run: [run-id]
- **Model**: openai/gpt-oss-20b
- **Config**: configs/prime-rl/beautiful-soup-env.toml
- **Start**: YYYY-MM-DD HH:MM
- **End**: YYYY-MM-DD HH:MM
- **Duration**: X hours
- **W&B Link**: https://wandb.ai/...

#### Metrics
- Baseline: X%
- Final: Y%
- Improvement: +Z%

#### Config Changes
- max_steps: 50 (smoke test)
- batch_size: 256
- rollouts_per_example: 8

#### Notes
- Any issues encountered
- Observations
```

---

## Configuration Reference

### Current Config: `configs/prime-rl/beautiful-soup-env.toml`

Key settings:
- **Model**: `openai/gpt-oss-20b` (21B params, 3.6B active, Apache 2.0)
- **LoRA**: rank=8, alpha=32
- **Batch**: 256 rollouts, 8 per example
- **Context**: seq_len=4096, max_tokens=10000
- **Environment**: seconds-0/beautiful-soup-env

### Launch Command

```bash
# On GPU pod with 2+ GPUs:
uv run rl @ configs/prime-rl/beautiful-soup-env.toml \
  --wandb.project beautiful-soup-env \
  --wandb.name bs4-rl-gpt-oss-20b-run1
```

### Monitoring

```bash
# tmux session for wandb
tmux attach -t wandb-monitor

# View W&B dashboard
open https://wandb.ai/YOUR_USERNAME/beautiful-soup-env
```

---

## Model Baselines (Pre-Training)

| Model | Baseline | Cost | Training Priority |
|-------|----------|------|-------------------|
| openai/gpt-oss-20b | 63.3% | $0.07/$0.30 | **PRIMARY TARGET** |
| prime-intellect/intellect-3 | ~78% | $0.20/$1.10 | Good but high baseline |
| qwen/qwen3-235b-a22b-instruct-2507 | 71.2% | $0.22/$0.88 | Good candidate |
| z-ai/glm-4.5-air | 66.8% | $0.20/$1.10 | Room for improvement |
