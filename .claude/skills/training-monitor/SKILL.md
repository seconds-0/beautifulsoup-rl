# Training Monitor Skill

Monitor prime-rl training runs and detect common failure modes.

## Activation

Use this skill when the user says:
- "check training", "is training running", "monitor run"
- "is the pod alive", "check wandb", "training status"
- "why did training fail", "debug training"

## Checks to Perform

### 1. WandB Status

Check if the training run is still active:

```bash
# Via Python script (if available)
python scripts/check_training.py

# Or via wandb CLI
wandb status
```

**Red flags:**
- Run shows "running" but `_runtime` stopped increasing
- Last update was >10 minutes ago
- No new metrics logged

### 2. Pod Status

```bash
# List all pods
prime pods list

# Get details for specific pod
prime pods status <pod-id>
```

**Red flags:**
- Pod status is not "ACTIVE"
- SSH connection fails
- Pod age is very short (may have been reclaimed)

### 3. SSH Health Check

```bash
# Get SSH command from pod status, then:
ssh -i ~/.ssh/primeintellect_ed25519 -p <port> root@<host> "nvidia-smi && ps aux | grep python"
```

**What to look for:**
- GPU memory usage (should be ~70-90% if training)
- Python processes running (trainer, orchestrator, inference)
- No OOM errors in dmesg

### 4. Training Logs

```bash
# On the pod:
tail -100 /tmp/training.log

# Check vLLM logs if rollouts are stuck:
tail -100 /tmp/vllm.log
```

## Common Failure Patterns

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| WandB "running" but no updates for 10+ min | Pod reclaimed (spot instance) | Create new pod, resume from checkpoint |
| Rollouts stuck at "0/N" | max_tokens > max_model_len - input_tokens | Reduce max_tokens in config |
| OOM error | Memory config wrong | Check gpu_memory_utilization, enable AC |
| KeyError in transformers | Model not supported | Use verified model (Qwen2.5-7B) |
| Device mismatch CPU/CUDA | fsdp_cpu_offload with LoRA | Use activation checkpointing instead |
| Reward = 0 constantly | Reward function bug | Check verifiers_adapter.py, update env |

## Quick Diagnostic Commands

```bash
# 1. Check WandB run status
python -c "
import wandb
api = wandb.Api()
run = api.run('seconds-0-domus-magna-inc/beautiful-soup-env/<run-id>')
print(f'State: {run.state}')
print(f'Last update: {run.summary.get(\"_timestamp\", \"unknown\")}')
"

# 2. Check pod is alive
prime pods list | grep ACTIVE

# 3. SSH and check GPU
ssh ... "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv"

# 4. Check for OOM in dmesg
ssh ... "dmesg | grep -i oom | tail -5"

# 5. Count Python processes
ssh ... "pgrep -c python"
```

## Recovery Procedures

### Pod Reclaimed (Spot Instance)

1. Confirm pod is dead: `prime pods list`
2. Check last checkpoint: Look at WandB for last step
3. Create new pod: `prime pods create --name bs4-rl-v8 --gpu-type H100_80GB --gpu-count 1`
4. Set up pod: Run `scripts/pod_setup.sh`
5. Resume: `uv run rl @ config.toml --ckpt.resume-step <last-step> --max-steps <target>`

### OOM During Training

1. SSH to pod and check: `nvidia-smi`
2. Kill training: `pkill -9 -f python`
3. Fix config:
   - Reduce `gpu_memory_utilization` to 0.45
   - Ensure `[trainer.model.ac] freq = 1`
   - Reduce `batch_size` and `rollouts_per_example`
4. Restart training

### Rollouts Stuck

1. Check vLLM logs: `tail /tmp/vllm.log`
2. Look for "max_tokens" errors
3. Fix: Set `max_tokens = 2000` if using `max_model_len = 4096`
4. Restart training

## Environment Variables Checklist

Before starting training, verify:

```bash
echo $VLLM_USE_V1              # Should be: 0
echo $VLLM_WORKER_MULTIPROC_METHOD  # Should be: spawn
echo $WANDB_API_KEY            # Should be set
```

## Related Files

- `scripts/check_training.py` - WandB monitoring script
- `scripts/pod_setup.sh` - Pod initialization
- `configs/prime-rl/qwen2.5-7b-h100.toml` - Single-GPU config
- `TRAINING_RUNS.md` - Training history and lessons
