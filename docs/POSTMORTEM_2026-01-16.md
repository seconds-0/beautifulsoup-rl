# Postmortem: Post-Fix Regression (2026-01-16)

## Executive Summary

Post-fix runs crash EARLIER than pre-fix runs with identical configs.

| Run ID | Config | Crash Step | Status |
|--------|--------|------------|--------|
| `nib461c5a1o3bqxwyen3cset` | Conservative | Step 27 | Pre-fix baseline |
| `j2ycmvx1gijwmidjivojuprb` | Conservative | Step 6 | Post-fix, STOPPED |
| `lcpzp0pl3yo42idk8jn2hksb` | Conservative + buffer | Step 3 | Post-fix, STOPPED |

## Crash Pattern

All crashes follow the same pattern:
```
INFO - Hit async barrier for step N. Waiting for checkpoint N...
[logs stop - no error, no traceback]
```

## Configuration (Identical Across Runs)

```toml
model = "Qwen/Qwen3-4B-Instruct-2507"
max_steps = 1000
batch_size = 32
rollouts_per_example = 4
max_async_level = 1
oversampling_factor = 1.0

[sampling]
max_tokens = 4096

[[env]]
id = "seconds-0/beautiful-soup-env"
args = { split = "train", mode = "all", cpu_cores = 2, memory_gb = 32 }
```

## Expert Consultation

### Codex Assessment
- **Verdict**: NOT the same LoRA loading issue
- LoRA failures surface at init with explicit errors
- These runs progress through multiple steps then stall at async barrier
- **Key insight**: Disable checkpointing - if hang disappears, checkpoint path is culprit

### Gemini Assessment
- **Verdict**: Same root cause (inference pod death), worse timing
- Fix likely introduced memory overhead during weight-reload phase
- Earlier termination suggests regressive fix adding overhead

## Ablation Plan

### Round 1 (Running)
1. **No Checkpointing** (`qwen3-4b-ablation-nockpt.toml`)
   - Hypothesis: If survives, checkpoint path is the culprit
   - Run ID: `sxp18yy8ombrfdhtq2c59c3f` (retry)
   - Dashboard: https://app.primeintellect.ai/dashboard/training/sxp18yy8ombrfdhtq2c59c3f
   - Result: `TBD`

2. **Ultra-Conservative** (`qwen3-4b-ablation-ultra.toml`)
   - batch_size=16, rollouts=2
   - Hypothesis: If survives, load/memory overhead from fix
   - Run ID: `w8phobtfkq4ar7nqz33theon` (retry)
   - Dashboard: https://app.primeintellect.ai/dashboard/training/w8phobtfkq4ar7nqz33theon
   - Result: `TBD`

### Round 2 (Running)
3. **Small Model** (`llama-1b-ablation-small.toml`)
   - Llama-3.2-1B instead of Qwen3-4B (4x smaller)
   - Hypothesis: If survives longer, GPU memory pressure is the issue
   - Run ID: `el76b7o6m0le32h1kdz782qh`
   - Dashboard: https://app.primeintellect.ai/dashboard/training/el76b7o6m0le32h1kdz782qh
   - Result: `TBD`

### Round 3 (If Needed)
4. **Minimal Config** - Remove buffer section

## Ablation Results

| Ablation | Run ID | Last Step | Status | Conclusion |
|----------|--------|-----------|--------|------------|
| No checkpointing | sxp18yy8ombrfdhtq2c59c3f | Step 5 | **STOPPED** | Config doesn't disable weight sync |
| Ultra-conservative | w8phobtfkq4ar7nqz33theon | Step 7 | **HUNG** at barrier | Slightly better but same failure |
| Sync mode (async=0) | N/A | N/A | **REJECTED** | Lab Hosted requires async_level >= 1 |
| Small model (1B) | cje51pf89wmtjpuwbvzzf6an | Step 0 | **FAILED** | "No samples" - Llama 1B too weak for BS4 task |

### Key Findings from Ablations

1. **"No checkpoint" doesn't disable the checkpoint wait** - The async barrier for weight sync is INTERNAL to prime-rl, not configurable via config file. Run still hits "Waiting for checkpoint N".

2. **Ultra-conservative survived 1 step longer** (7 vs 6) but still hung at the same barrier pattern.

3. **All runs hang at async barrier** - This is consistent across:
   - Production (step 6)
   - Bootstrap (step 3)
   - No-ckpt ablation (step 5)
   - Ultra-conservative ablation (step 7)

4. **Conclusion: Platform-side issue confirmed** - The checkpoint/weight sync path is hanging regardless of our configuration. This is NOT our environment or config.

5. **`max_async_level = 0` not allowed** - Lab Hosted requires `max_async_level >= 1`. We cannot disable async mode to bypass the barrier. This means we're forced to use the async weight sync path that's hanging.

6. **Smaller model (Llama 1B) not viable** - Failed with "Step with no samples" - the model is too weak to generate valid BeautifulSoup code. Cannot test GPU memory hypothesis this way.

### Comparison Table

| Run | Config | Last Step | Wait Time at Barrier |
|-----|--------|-----------|---------------------|
| Pre-fix elimination | Conservative | Step 27 | Survived ~30 min |
| Post-fix production | Conservative | Step 6 | Crashed |
| Post-fix bootstrap | Conservative + buffer | Step 3 | Crashed |
| Ablation: no-ckpt | Conservative | Step 5 | **12+ min hung** |
| Ablation: ultra-cons | batch=16, rollouts=2 | Step 7 | **6+ min hung** |

## Timeline

- **2026-01-16 ~09:00**: Prime deploys LoRA loading fix
- **2026-01-16 ~10:00**: Started production run `j2ycmvx1gijwmidjivojuprb`
- **2026-01-16 ~10:00**: Started bootstrap run `lcpzp0pl3yo42idk8jn2hksb`
- **2026-01-16 ~11:00**: Production crashed at step 6
- **2026-01-16 ~11:00**: Bootstrap crashed at step 3
- **2026-01-16 ~12:00**: Stopped crashed runs, starting ablations
- **2026-01-16 ~12:30**: Started ablation runs (first attempt failed - missing WANDB key)
- **2026-01-16 ~11:30**: Restarted ablations with WANDB key
- **2026-01-16 ~11:40**: No-ckpt ablation reached step 5, hung at barrier
- **2026-01-16 ~11:46**: Ultra-conservative ablation reached step 7, hung at barrier
- **2026-01-16 ~11:52**: Both ablations confirmed hung (no progress 6-12 min)
- **2026-01-16 ~12:41**: Stopped no-ckpt ablation, tried sync mode (rejected - async_level >= 1 required)
- **2026-01-16 ~12:43**: Started small model ablation (Llama 1B)
- **2026-01-16 ~12:50**: Small model failed - "no samples" (model too weak for BS4 task)
- **2026-01-16 ~12:55**: Ultra-conservative still hung at step 7 for **1+ hour**

## Data Request for Prime

Need for runs:
- `j2ycmvx1gijwmidjivojuprb` (production, crashed step 6)
- `lcpzp0pl3yo42idk8jn2hksb` (bootstrap, crashed step 3)
- `sxp18yy8ombrfdhtq2c59c3f` (ablation no-ckpt, hung step 5)
- `w8phobtfkq4ar7nqz33theon` (ablation ultra-conservative, hung step 7)

1. Container exit codes (OOMKilled? SIGKILL? Evicted?)
2. K8s pod events around crash time
3. Inference pod logs - Did they OOM?
4. Resource metrics (GPU mem, CPU mem) at crash window
5. Exact image/commit SHA - Verify fix version deployed
6. Does the new LoRA fix increase peak memory during weight reload?

---
*Last updated: 2026-01-16*
