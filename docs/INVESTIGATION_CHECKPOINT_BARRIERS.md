# Investigation Notes: Lab Hosted Checkpoint Barrier Behavior

## Summary

Lab Hosted RL training runs are getting stuck at checkpoint barriers. This document records our experimental observations and possible explanations for further discussion with Prime.

---

## Experimental Runs (2026-01-17)

### Runs That Got Stuck

| Run ID | Config | Batch | Mode | Async | Observation |
|--------|--------|-------|------|-------|-------------|
| `heeh6kmc19fbh4z0ch7pq4x7` | v0.1.5-validation | 32 | all | 1 | Stuck at ckpt 1 (step 2) |
| `qhubcp7xqykyr8eywfz87snc` | v015-test-64 | 64 | all | 1 | Stuck at ckpt 1 (step 2) |
| `i9bmguxoa3fn79f0qreqsesc` | v015-test-128 | 128 | all | 1 | Stuck at ckpt 1 (step 2) |
| `mjjk90a8q425ax4pgg85qxl6` | Thinking-bootstrap | 128 | bootstrap | 1 | Stuck at ckpt 1 |
| `zkzholsamw9o924r7uzgzry5` | Instruct-bootstrap | 128 | bootstrap | 1 | Steps 0-3, stuck at ckpt 3 |
| `kdmvezp0xxj3k3sceq63wm8q` | async2-batch32 | 32 | bootstrap | 2 | Steps 0-4, stuck at ckpt 3 |
| `bvm2x1i7eulqeelqm2oxblmh` | async2-batch64 | 64 | bootstrap | 2 | Steps 0-4, stuck at ckpt 3 |

### Successful Run (Reference)

| Run ID | Config | Batch | Mode | Async | Observation |
|--------|--------|-------|------|-------|-------------|
| `kfnq9gn73pb044hg7hb55idj` | ablation-async0 | 128 | bootstrap | 1 | Completed 50 steps |

---

## What We Observed

### Pattern 1: Checkpoint Barrier Timing

With `max_async_level=1`:
- Step 0-1 complete normally
- Step 2 waits for checkpoint 1
- In some cases, checkpoint never arrives

With `max_async_level=2`:
- Steps 0-2 complete normally
- Step 3 waits for checkpoint 1 (recovered after ~2 min in our tests)
- Steps 3-4 complete
- Step 5 waits for checkpoint 3 (did not recover after 12+ min)

### Pattern 2: First Barrier vs Later Barriers

We noticed the first checkpoint barrier sometimes recovers, but later barriers seem to get stuck more persistently. This is interesting but we don't fully understand why.

### Pattern 3: Task Complexity Impact

- `mode=all` runs got stuck earlier (step 2)
- `mode=bootstrap` runs got further (step 3-4) before getting stuck

This could suggest that simpler/faster tasks allow more progress, though the runs still eventually got stuck.

### Pattern 4: The "Instant Step" Anomaly

Run `zkzholsamw9o924r7uzgzry5` recorded **0.08s** for Step 3 - this seems physically impossible for 128 rollouts. This could indicate:
- The step was skipped or cached
- 0 samples were produced (all filtered?)
- Some other system behavior we don't understand

This anomaly immediately preceded the stuck barrier at checkpoint 3.

---

## Hypotheses We Tested

### 1. Batch Size

We tried batch sizes 32, 64, and 128 with `mode=all`. All got stuck at the same point (checkpoint 1, step 2). **Batch size alone doesn't seem to explain it.**

### 2. Task Complexity (mode)

Bootstrap mode got further than "all" mode. This could mean task execution time plays a role, or it could be coincidental.

### 3. Async Level Buffering

Higher async level allowed more steps before hitting a barrier, but didn't prevent eventual barrier issues.

### 4. v0.1.5 Changes

Our v0.1.5 includes performance optimizations (PooledSubprocessExecutor, unified AST analysis). These changes affect local execution in the environment, not trainer-orchestrator communication. We don't think these are related, but we can't rule it out completely.

---

## Possible Explanations (Speculative)

We're not sure what's causing this, but here are some possibilities we're considering:

1. **Timing sensitivity**: Maybe the checkpoint sync has timing assumptions that our environment doesn't always meet?

2. **Multi-tenant effects**: Lab Hosted is shared infrastructure. Perhaps load from other users affects checkpoint timing?

3. **Something specific to our environment**: Maybe something about our environment's execution patterns interacts poorly with the checkpoint mechanism?

4. **Transient infrastructure state**: The successful run used the same settings that fail now. Perhaps something changed on the infrastructure side?

5. **Empty batch deadlock**: If the "Instant Step" (0.08s) indicates 0 samples were produced, this could break the checkpoint heartbeat mechanism.

We genuinely don't know which (if any) of these is correct.

---

## What We're Seeing in Logs

When stuck at a barrier, the orchestrator logs show it's waiting for a checkpoint that hasn't arrived yet. The trainer side may still be processing, or there may be some sync issue we're not seeing from our side.

---

## Questions for Prime Team

We'd like to understand:

1. Is this expected behavior under certain conditions?
2. Are there configuration options we should try?
3. What does normal checkpoint production timing look like?
4. Is there anything about our environment that might cause issues?
5. Did run `zkzholsamw9o924r7uzgzry5` Step 3 have `num_samples = 0`?
6. Do trainer logs show checkpoint creation attempts or stalls for these run IDs?

**Run IDs for reference:**
- Stuck runs: `heeh6kmc19fbh4z0ch7pq4x7`, `qhubcp7xqykyr8eywfz87snc`, `i9bmguxoa3fn79f0qreqsesc`, `zkzholsamw9o924r7uzgzry5`, `kdmvezp0xxj3k3sceq63wm8q`, `bvm2x1i7eulqeelqm2oxblmh`, `mjjk90a8q425ax4pgg85qxl6`
- Successful run: `kfnq9gn73pb044hg7hb55idj`

---

## Our Working Notes

### Architecture Understanding (based on docs)

Prime RL Lab Hosted has three components:
- **Trainer**: Shared multi-tenant, handles optimization
- **Inference**: Shared multi-tenant LoRA for generation
- **Orchestrator**: Dedicated per run, runs our environment

The orchestrator can run ahead of the trainer up to `max_async_level` steps before needing to wait for a checkpoint.

### Configs Tested

All configs are in `configs/lab/`:
- `qwen3-4b-v015-test-64.toml`
- `qwen3-4b-v015-test-128.toml`
- `qwen3-4b-v015-bootstrap.toml`
- `qwen3-4b-async2-batch32.toml`
- `qwen3-4b-async2-batch64.toml`

---

## External Review Analysis

We had two independent reviewers (Codex and Gemini) analyze the run data.

### Codex Conclusions

**Ranked hypotheses (most to least likely):**
1. Trainer checkpoint production is intermittently delayed or stalled
2. Shared infrastructure contention (multi-tenant load)
3. Trainer-orchestrator handshake bug under certain timing
4. Model variants have slower trainer step time (unlikely)
5. v0.1.5 optimizations indirectly expose trainer lag
6. Batch size is causal (RULED OUT)

**Key insight:** Trainer may stall after initial checkpoint or checkpoint production slows progressively, so later checkpoints never arrive.

### Gemini Conclusions

**Key finding:** The "Instant Step" anomaly (0.08s for Step 3) is a critical data point.

**Hypotheses:**
1. Trainer Resource Exhaustion for `mode=all`
2. "Perfect Score" Deadlock - if Instruct model achieves 100% and filtering removes easy tasks
3. Checkpoint Race Condition due to fast orchestrator

### Synthesis

Both reviewers agree:
- Batch size is definitively ruled out as a factor
- Trainer-side issues are the most likely root cause
- The "Instant Step" anomaly deserves investigation
- v0.1.5 optimizations may contribute by making orchestrator faster, exposing timing issues

---

## Next Steps

1. **Check WandB** for run `zkzholsamw9o924r7uzgzry5` Step 3 metrics (num_samples)
2. **Share this document** with Prime team
3. **Wait for guidance** before attempting more runs
4. **Consider self-hosted** as backup option if Lab Hosted issues persist
