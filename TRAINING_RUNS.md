# Training Runs Log

Track all RL training experiments for BeautifulSoup environment.

## Active Runs

### Production Training (2026-01-25) - Lab Hosted

**Context**: All checkpoint barriers confirmed fixed. Launching parallel production runs.

#### 4B Instruct Production (lsbkg50x9v9func6kk8t0cco) - RUNNING 🏃

- **Run ID**: `lsbkg50x9v9func6kk8t0cco`
- **Config**: `configs/lab/qwen3-4b-production.toml`
- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **Status**: 🏃 RUNNING
- **Dashboard**: https://app.primeintellect.ai/dashboard/training/lsbkg50x9v9func6kk8t0cco

| Setting | Value |
|---------|-------|
| max_steps | 500 |
| batch_size | 128 |
| rollouts_per_example | 8 |
| max_async_level | 2 |
| mode | all |
| max_tokens | 4096 |

#### 30B Thinking Production (re0my0c3qu5o8c6dwwencwob) - RUNNING 🏃

- **Run ID**: `re0my0c3qu5o8c6dwwencwob`
- **Config**: `configs/lab/qwen3-30b-production.toml`
- **Model**: Qwen/Qwen3-30B-A3B-Thinking-2507
- **Status**: 🏃 RUNNING
- **Dashboard**: https://app.primeintellect.ai/dashboard/training/re0my0c3qu5o8c6dwwencwob

| Setting | Value |
|---------|-------|
| max_steps | 500 |
| batch_size | 32 |
| rollouts_per_example | 4 |
| max_async_level | 2 |
| mode | all |
| max_tokens | 4096 |

---

### Post-Fix Ablation Testing (2026-01-25) - Lab Hosted

**Context**: Prime team deployed additional fixes. Testing systematically to verify all checkpoint barriers are resolved.

#### Executive Summary: FIXES CONFIRMED! ✅

| Test | Config | Model | Result | Notes |
|------|--------|-------|--------|-------|
| 1 | qwen3-235b-async2.toml | 235B Instruct | SKIPPED | Model not available |
| 2 | qwen3-4b-validation.toml | 4B Instruct | ✅ **SUCCESS** | 5/5 steps, all checkpoints |
| 3 | qwen3-4b-ablation-sync.toml | 4B Instruct | ✅ **COMPLETED** | 50/50 steps |
| 4 | qwen3-4b-ablation-small.toml | 4B Instruct (async=2) | ✅ **SUCCESS** | Async=2 working, confirmed checkpoints |
| 5 | qwen3-30b-a3b-validation.toml | **30B Thinking** | ✅ **COMPLETED** | **10/10 steps complete!** |
| 6 | qwen3-4b-v015-bootstrap.toml | **4B Thinking** | ✅ **COMPLETED** | 50/50 steps |

**Key Findings:**
1. **Checkpoint 2 barrier FIXED** - Previously all runs stuck here
2. **Thinking models FIXED** - Previously stuck at checkpoint 1
3. **async=2 working** - Full async buffering operational
4. **All model types passing** - Instruct and Thinking both work now

#### Test 2: 4B Instruct batch=32 (bvlb6u0xa216ped2csm3opfk) - COMPLETED ✅

- **Run ID**: `bvlb6u0xa216ped2csm3opfk`
- **Config**: `configs/lab/qwen3-4b-validation.toml`
- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **Status**: ✅ **COMPLETED** (5/5 steps)
- **Dashboard**: https://app.primeintellect.ai/dashboard/training/bvlb6u0xa216ped2csm3opfk

| Step | Time | Reward | Checkpoint | Notes |
|------|------|--------|------------|-------|
| 0 | 23.40s | 0.0625 | ✅ | Init |
| 1 | 19.73s | 0.1594 | ✅ | |
| 2 | 102.13s | 0.0021 | ✅ | Checkpoint 1 passed |
| 3 | 12.89s | 0.0625 | ✅ | **Checkpoint 2 passed** (prev failure) |
| 4 | 91.18s | 0.3516 | ✅ | Checkpoint 3 passed |
| 5 | - | - | ✅ | **Orchestrator finished!** |

**Result**: All checkpoints passed. Checkpoint 2 barrier is FIXED.

#### Test 5: 30B Thinking (ebwreyf6e4qr9htk0l5jd1da) - COMPLETED ✅

- **Run ID**: `ebwreyf6e4qr9htk0l5jd1da`
- **Config**: `configs/lab/qwen3-30b-a3b-validation.toml`
- **Model**: Qwen/Qwen3-30B-A3B-Thinking-2507
- **Status**: ✅ **COMPLETED** (10/10 steps)
- **Dashboard**: https://app.primeintellect.ai/dashboard/training/ebwreyf6e4qr9htk0l5jd1da

| Step | Time | Reward | Checkpoint | Notes |
|------|------|--------|------------|-------|
| 0 | 15.66s | 0.0312 | ✅ | |
| 1 | 59.80s | 0.1250 | ✅ | |
| 2 | 30.71s | 0.1562 | ✅ | |
| 3 | 32.97s | 0.1250 | ✅ | |
| 4 | 44.95s | 0.0312 | ✅ | Checkpoint 2 passed |
| 5 | 45.36s | 0.0312 | ✅ | Checkpoint 3 passed |
| 6 | 116.06s | 0.0312 | ✅ | Checkpoint 4 passed |
| 7 | 36.80s | 0.1250 | ✅ | Checkpoint 5 passed |
| 8 | 60.05s | 0.0625 | ✅ | Checkpoint 6 passed |
| 9 | 75.46s | 0.1875 | ✅ | Checkpoint 7 passed |
| 10 | - | - | ✅ | **Orchestrator finished!** |

**MAJOR BREAKTHROUGH**: Thinking models were previously stuck at checkpoint 1. Now fully working!

#### Test 3: 4B Instruct sync 50 steps (injfscl3651zqi1onzeq2d4h) - RUNNING

- **Run ID**: `injfscl3651zqi1onzeq2d4h`
- **Config**: `configs/lab/qwen3-4b-ablation-sync.toml`
- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **Status**: 🏃 RUNNING (5+ steps completed)
- **Dashboard**: https://app.primeintellect.ai/dashboard/training/injfscl3651zqi1onzeq2d4h

Steps 0-5 completed, all checkpoints passing. 50-step run in progress.

#### Test 4: 4B async=2 bootstrap (vbbw7g3569dayx1j1a0ifhwq) - STOPPED (validated)

- **Run ID**: `vbbw7g3569dayx1j1a0ifhwq`
- **Config**: `configs/lab/qwen3-4b-ablation-small.toml`
- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **Status**: ✅ STOPPED after validation (checkpoints working)
- **Dashboard**: https://app.primeintellect.ai/dashboard/training/vbbw7g3569dayx1j1a0ifhwq

Confirmed async=2 working with Async Level reaching 2. Stopped early to test Thinking models.

#### Test 6: 4B Thinking bootstrap (wtrrji00xf42b7c7aa2gk8ct) - RUNNING

- **Run ID**: `wtrrji00xf42b7c7aa2gk8ct`
- **Config**: `configs/lab/qwen3-4b-v015-bootstrap.toml`
- **Model**: Qwen/Qwen3-4B-Thinking-2507
- **Status**: 🏃 RUNNING
- **Dashboard**: https://app.primeintellect.ai/dashboard/training/wtrrji00xf42b7c7aa2gk8ct

Large batch (128 x 8 rollouts), checkpoint 1+ passed.

---

### Race Condition Fix Retest (2026-01-17) - Lab Hosted

**Context**: Prime team confirmed and fixed a race condition that was causing runs to get stuck after step 2. Testing to verify the fix.

#### Retest 1: qwen3-4b-validation (2026-01-17) - CANCELLED

- **Run ID**: `uk7w1k2scdrdgigvkhs2jffi`
- **Config**: `configs/lab/qwen3-4b-validation.toml`
- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **Previous Run**: `heeh6kmc19fbh4z0ch7pq4x7` (stuck at step 2, checkpoint 1)
- **Status**: ⚠️ CANCELLED - Passed checkpoint 1, stuck at checkpoint 2
- **Dashboard**: https://app.primeintellect.ai/dashboard/training/uk7w1k2scdrdgigvkhs2jffi

| Step | Time | Reward | Checkpoint | Notes |
|------|------|--------|------------|-------|
| 0 | 39.89s | 0.0625 | ✅ | Healthy |
| 1 | 15.66s | 0.0906 | ✅ | Healthy |
| 2 | 122.35s | 0.2156 | ✅ | **PASSED checkpoint 1!** (prev failure point) |
| 3 | - | - | ❌ | Stuck waiting for checkpoint 2 (30+ min), cancelled |

**Result**: ⚠️ PARTIAL SUCCESS - Race condition fix helped pass checkpoint 1, but new issue at checkpoint 2

#### Retest 2: qwen3-4b-v015-test-64 (2026-01-17) - BLOCKED

- **Run ID**: `g3f7tut16uraprkrxe902ny0`
- **Config**: `configs/lab/qwen3-4b-v015-test-64.toml`
- **Model**: Qwen/Qwen3-4B-Thinking-2507 (Instruct no longer available)
- **Previous Run**: `qhubcp7xqykyr8eywfz87snc` (stuck at step 2)
- **Status**: ❌ FAILED (WandB API key error)
- **Dashboard**: https://app.primeintellect.ai/dashboard/training/g3f7tut16uraprkrxe902ny0

**Note**: Config fixed (wandb disabled), but hit concurrent run limit.

#### Retest 3: qwen3-30b-a3b-validation (2026-01-17) - CANCELLED

- **Run ID**: `t5x4tt2htn5zy2642yqnn5cl`
- **Config**: `configs/lab/qwen3-30b-a3b-validation.toml`
- **Model**: Qwen/Qwen3-30B-A3B-Thinking-2507 (larger MoE model)
- **Status**: ❌ CANCELLED - Stuck at checkpoint 1 (same as old failure)
- **Dashboard**: https://app.primeintellect.ai/dashboard/training/t5x4tt2htn5zy2642yqnn5cl

| Step | Time | Reward | Checkpoint | Notes |
|------|------|--------|------------|-------|
| 0 | 39.95s | 0.0312 | ✅ | Healthy |
| 1 | 42.98s | 0.0625 | ✅ | Healthy |
| 2 | - | - | ❌ | Stuck waiting for checkpoint 1 (15+ min), cancelled |

**Key Finding**: Thinking models still have checkpoint 1 issue. Race condition fix may only work for Instruct models.

#### Summary of Retest Results (2026-01-17)

| Model | Mode | Async | Ckpt 1 | Ckpt 2 | Best Reward | Notes |
|-------|------|-------|--------|--------|-------------|-------|
| **4B Instruct** | all | 1 | ✅ | ❌ | 21.6% | Race fix helped partially |
| **30B A3B Thinking** | all | 1 | ❌ | - | 6.3% | Same as before fix |
| **4B Thinking** | bootstrap | 1 | ❌ | - | 31.3% | Bootstrap mode didn't help |
| **235B Instruct** | all | 1 | ❌ | - | **92.2%** | Excellent baseline, stuck early |
| **235B Instruct** | all | **2** | ✅ | ❌ | **100%** | async=2 helped, stuck at ckpt 2 |
| **30B A3B Thinking** | all | **2** | ❌ | - | 9.4% | Thinking still stuck with async=2 |

**Key Findings**:
1. **async_level=2 helps Instruct models** - 235B passed checkpoint 1 with async=2 (9 min wait)
2. **Thinking models consistently fail** - Stuck at checkpoint 1 regardless of async level
3. **235B Instruct has exceptional baseline** - 100% and 84% rewards in early steps
4. **Checkpoint barriers persist** - Even when ckpt 1 passes, ckpt 2 fails
5. **Model availability unstable** - 4B Instruct and 4B Thinking rotated out during testing

**Recommendation**:
- Use **235B Instruct with async_level=2** for best results
- Avoid Thinking models until Prime fixes the checkpoint barrier
- Report to Prime that checkpoint 2 barrier still occurs even after checkpoint 1 passes

#### Retest 6: qwen3-235b-a22b-async2 (2026-01-17) - CANCELLED

- **Run ID**: `ic1hz2we3u7i58efwk7pji47`
- **Config**: `configs/lab/qwen3-235b-async2.toml`
- **Model**: Qwen/Qwen3-235B-A22B-Instruct-2507
- **Mode**: all, **max_async_level=2**
- **Status**: ⚠️ CANCELLED - Passed ckpt 1, stuck at ckpt 2
- **Dashboard**: https://app.primeintellect.ai/dashboard/training/ic1hz2we3u7i58efwk7pji47

| Step | Time | Reward | Checkpoint | Notes |
|------|------|--------|------------|-------|
| 0 | 27.91s | **1.0000** | ✅ | 100% reward! |
| 1 | 35.33s | 0.2812 | ✅ | Healthy |
| 2 | 23.78s | **0.8438** | ✅ | 84% reward |
| 3 | 542.30s | 0.3900 | ✅ | **PASSED checkpoint 1** after 9 min wait |
| 4 | - | - | ❌ | Stuck waiting for checkpoint 2 (15+ min), cancelled |

**Result**: async_level=2 helped pass checkpoint 1, but stuck at checkpoint 2. Excellent baseline rewards (100%, 84%).

#### Retest 7: qwen3-30b-a3b-async2 (2026-01-17) - CANCELLED

- **Run ID**: `bizc538ns5o4fjnm07n132ux`
- **Config**: `configs/lab/qwen3-30b-a3b-validation.toml`
- **Model**: Qwen/Qwen3-30B-A3B-Thinking-2507
- **Mode**: all, **max_async_level=2**
- **Status**: ❌ CANCELLED - Stuck at checkpoint 1
- **Dashboard**: https://app.primeintellect.ai/dashboard/training/bizc538ns5o4fjnm07n132ux

| Step | Time | Reward | Checkpoint | Notes |
|------|------|--------|------------|-------|
| 0 | 15.60s | 0.0625 | ✅ | Healthy |
| 1 | 14.22s | 0.0938 | ✅ | Healthy |
| 2 | 18.71s | 0.0312 | ✅ | Healthy |
| 3 | - | - | ❌ | Stuck waiting for checkpoint 1 (20+ min), cancelled |

**Result**: Thinking models still stuck even with async_level=2.

**Note**: 4B Thinking model no longer available - rotated out during testing.

#### Retest 4: qwen3-4b-v015-bootstrap (2026-01-17) - CANCELLED

- **Run ID**: `v5t3j546osfypw3qgq2mbvk1`
- **Config**: `configs/lab/qwen3-4b-v015-bootstrap.toml`
- **Model**: Qwen/Qwen3-4B-Thinking-2507
- **Mode**: bootstrap (simpler tasks)
- **Status**: ❌ CANCELLED - Stuck at checkpoint 1
- **Dashboard**: https://app.primeintellect.ai/dashboard/training/v5t3j546osfypw3qgq2mbvk1

| Step | Time | Reward | Checkpoint | Notes |
|------|------|--------|------------|-------|
| 0 | 130.71s | 0.3125 | ✅ | Healthy |
| 1 | 115.31s | 0.2709 | ✅ | Healthy |
| 2 | - | - | ❌ | Stuck waiting for checkpoint 1 (10+ min), cancelled |

**Result**: Bootstrap mode didn't help Thinking models avoid checkpoint barrier.

#### Retest 5: qwen3-235b-a22b-validation (2026-01-17) - CANCELLED

- **Run ID**: `th6cz0teyowv5gm7o3e64mmj`
- **Config**: `configs/lab/qwen3-235b-a22b-validation.toml`
- **Model**: Qwen/Qwen3-235B-A22B-Instruct-2507 (large MoE, 22B active)
- **Mode**: all
- **Status**: ❌ CANCELLED - Stuck at checkpoint 1
- **Dashboard**: https://app.primeintellect.ai/dashboard/training/th6cz0teyowv5gm7o3e64mmj

| Step | Time | Reward | Checkpoint | Notes |
|------|------|--------|------------|-------|
| 0 | 15.29s | **0.9219** | ✅ | Excellent baseline! |
| 1 | 18.74s | **0.8554** | ✅ | Excellent baseline! |
| 2 | - | - | ❌ | Stuck waiting for checkpoint 1 (10+ min), cancelled |

**Result**: Even 235B Instruct stuck at checkpoint 1, despite earlier 4B Instruct passing it.

---

### Batch Size Investigation (2026-01-17) - Lab Hosted

**Context**: v0.1.5 validation run (`heeh6kmc19fbh4z0ch7pq4x7`) got stuck at step 2 waiting for checkpoint 1. Root cause: batch_size=32 too small for trainer to produce checkpoint with `max_async_level=1`.

**Hypothesis**: Trainer requires minimum batch size (64-128) to produce checkpoints when async sync is enforced.

#### Experiment A: batch_size=64

- **Run ID**: `qhubcp7xqykyr8eywfz87snc`
- **Config**: `configs/lab/qwen3-4b-v015-test-64.toml`
- **Settings**: batch_size=64, rollouts_per_example=8, max_async_level=1
- **WandB**: beautifulsoup-rl/v0.1.5-batch64-test
- **Status**: ❌ STUCK (async barrier deadlock)
- **Dashboard**: https://app.primeintellect.ai/dashboard/training/qhubcp7xqykyr8eywfz87snc

| Step | Time | Reward | Checkpoint Sync | Notes |
|------|------|--------|-----------------|-------|
| 0 | 36.83s | 0.1898 | ✅ | Healthy |
| 1 | 33.78s | 0.1818 | ✅ | Healthy |
| 2 | - | - | ❌ STUCK | Waiting for checkpoint 1 (10+ min) |

**Result**: ❌ FAILED - Stuck at async barrier like original run

#### Experiment B: batch_size=128

- **Run ID**: `i9bmguxoa3fn79f0qreqsesc`
- **Config**: `configs/lab/qwen3-4b-v015-test-128.toml`
- **Settings**: batch_size=128, rollouts_per_example=8, max_async_level=1
- **WandB**: beautifulsoup-rl/v0.1.5-batch128-test
- **Status**: ❌ STUCK (async barrier deadlock)
- **Dashboard**: https://app.primeintellect.ai/dashboard/training/i9bmguxoa3fn79f0qreqsesc

| Step | Time | Reward | Checkpoint Sync | Notes |
|------|------|--------|-----------------|-------|
| 0 | 112.78s | 0.0505 | ✅ | Healthy |
| 1 | 66.41s | 0.1227 | ✅ | Healthy |
| 2 | - | - | ❌ STUCK | Waiting for checkpoint 1 (10+ min) |

**Result**: ❌ FAILED - Stuck at async barrier like original run

#### Conclusion: Batch Size is NOT the Root Cause

**Key Finding**: Both batch_size=64 and batch_size=128 got stuck at the same checkpoint barrier.

**Root Cause Identified**: The successful run (`kfnq9gn73pb044hg7hb55idj`) used `mode = "bootstrap"`, NOT `mode = "all"`.

| Config | Mode | Batch | Async | Result |
|--------|------|-------|-------|--------|
| v0.1.5-validation (stuck) | all | 32 | 1 | ❌ Stuck at step 2 |
| v0.1.5-batch64-test | all | 64 | 1 | ❌ Stuck at step 2 |
| v0.1.5-batch128-test | all | 128 | 1 | ❌ Stuck at step 2 |
| kfnq9gn73pb044hg7hb55idj (success) | **bootstrap** | 128 | 1 | ✅ 50 steps |

**Why mode=all causes deadlock**:
1. Complex tasks in `mode=all` produce long outputs (up to 4096 tokens)
2. Longer outputs = more trainer processing time
3. Trainer can't produce checkpoint 1 before orchestrator step 2
4. Orchestrator waits indefinitely at async barrier

**Update (2026-01-17 16:30)**: Even `mode = "bootstrap"` with `Qwen3-4B-Instruct` got stuck at checkpoint 3 barrier after completing steps 0-3. This appears to be a **Prime infrastructure issue**, not a configuration issue.

#### Bootstrap Test Run: zkzholsamw9o924r7uzgzry5

- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **Mode**: bootstrap
- **Status**: ❌ STUCK at checkpoint 3 barrier

| Step | Time | Reward | Notes |
|------|------|--------|-------|
| 0 | 130.40s | 0.3173 | ✅ |
| 1 | 72.06s | 0.3432 | ✅ |
| 2 | 64.41s | 0.2896 | ✅ (recovered from ckpt 1 barrier) |
| 3 | 0.08s | 0.2886 | ✅ (instant - reused samples?) |
| 4 | - | - | ❌ Stuck waiting for checkpoint 3 |

**Conclusion**: Lab Hosted checkpoint sync is unreliable. This is a Prime infrastructure bug, not user config. Need to report to Prime team.

---

### Comprehensive Investigation Analysis (2026-01-17)

**Context**: Multiple runs got stuck at checkpoint barriers despite varying batch sizes, modes, and async levels. We conducted parallel deep reviews with Codex and Gemini to analyze the patterns.

#### All Tested Runs Summary

| Run ID | Config | Batch | Mode | Async | Result |
|--------|--------|-------|------|-------|--------|
| `heeh6kmc19fbh4z0ch7pq4x7` | v0.1.5-validation | 32 | all | 1 | ❌ Stuck at ckpt 1 (step 2) |
| `qhubcp7xqykyr8eywfz87snc` | v015-test-64 | 64 | all | 1 | ❌ Stuck at ckpt 1 (step 2) |
| `i9bmguxoa3fn79f0qreqsesc` | v015-test-128 | 128 | all | 1 | ❌ Stuck at ckpt 1 (step 2) |
| `mjjk90a8q425ax4pgg85qxl6` | Thinking-bootstrap | 128 | bootstrap | 1 | ❌ Stuck at ckpt 1 |
| `zkzholsamw9o924r7uzgzry5` | Instruct-bootstrap | 128 | bootstrap | 1 | ❌ Steps 0-3, stuck at ckpt 3 |
| `kdmvezp0xxj3k3sceq63wm8q` | async2-batch32 | 32 | bootstrap | 2 | ❌ Steps 0-4, stuck at ckpt 3 |
| `bvm2x1i7eulqeelqm2oxblmh` | async2-batch64 | 64 | bootstrap | 2 | ❌ Steps 0-4, stuck at ckpt 3 |
| `kfnq9gn73pb044hg7hb55idj` | ablation-async0 | 128 | bootstrap | 1 | ✅ Completed 50 steps |

#### Codex Analysis (Ranked Hypotheses)

1. **Trainer checkpoint production intermittently delayed/stalled** (MOST LIKELY)
   - Support: All failures are "stuck at checkpoint barrier," not execution errors
   - async=2 reaches later steps but still stalls
   - Bootstrap (faster steps) gets further before stall

2. **Shared infrastructure contention (multi-tenant load)**
   - Support: Same config succeeded earlier but later failed
   - Multiple runs on same date all stuck
   - Time-of-day or bursty load could slow checkpoint creation

3. **Trainer–orchestrator handshake bug under certain timing**
   - Support: "Checkpoint 1 sometimes recovers but checkpoint 3 never recovers"
   - Suggests inconsistent recovery/timeout behavior

4. **Model variants have slower trainer step time** - UNLIKELY
   - Stuck happens with all variants, not model-specific

5. **v0.1.5 optimizations indirectly expose trainer lag**
   - Faster env execution can drive orchestrator ahead
   - Reveals trainer slowness that was previously masked

6. **Batch size is causal** - RULED OUT
   - Failures at 32/64/128 with same pattern
   - Successful run was batch128

#### Gemini Analysis (Different Perspective)

**Key Finding: The "Instant Step" Anomaly**

Run `zkzholsamw9o924r7uzgzry5` recorded **0.08s** for Step 3 - physically impossible for 128 rollouts unless:
- Step was skipped
- Samples were cached/reused
- Step had 0 samples (all filtered out)

**Gemini's Hypotheses:**

1. **H1: Trainer Resource Exhaustion (mode=all)**
   - Shared Trainer can't handle compute/memory intensity of mode=all
   - 4096 tokens, complex graphs may cause silent crashes during Step 0

2. **H2: "Perfect Score" Deadlock (mode=bootstrap)**
   - Instruct model achieves 100% reward on trivial bootstrap tasks
   - If filtering removes "easy" tasks, Step 3 became empty (0 samples)
   - Empty step breaks checkpoint heartbeat, causing hang

3. **H3: Checkpoint Race Condition**
   - 0.08s step indicates Orchestrator raced ahead
   - Dependency graph locks up when Orchestrator moves to Step 4 before Trainer acknowledges Step 2/3

#### Key Insights from Both Reviews

1. **Batch size definitively ruled out** - Both reviewers agree
2. **The "Instant Step" anomaly** is a new lead - if Step 3 had 0 samples, explains deadlock
3. **Trainer-side stalling** is most likely root cause
4. **v0.1.5 optimizations** may indirectly contribute by making orchestrator faster
5. **Model behavior matters** - Instruct vs Thinking may produce different training data patterns

#### Questions for Prime Team

1. Do trainer logs show checkpoint creation attempts or stalls for these run IDs?
2. Any infra incidents or load spikes on 2026-01-17?
3. Any recent changes in trainer checkpointing cadence?
4. Are checkpoint files created but failing to upload/commit?
5. Can we compare trainer step time between successful and failed runs?
6. Did `zkzholsamw9o924r7uzgzry5` Step 3 have `num_samples = 0`?

#### Data Needed

- [ ] WandB "Samples" metric for Step 3 of `zkzholsamw9o924r7uzgzry5`
- [ ] Trainer-side logs for stuck runs
- [ ] Infra metrics: trainer queue length, GPU utilization, storage I/O latency
- [ ] Compare wall-clock time to reach step 2/3 across runs

---

#### Success Criteria

- At least one run completes 10+ steps without checkpoint deadlock
- Identify minimum viable batch_size
- Document findings for production config

---

### Run: qwen3-4b-optimized-v3-500steps (2026-01-14) - Lab Hosted - FAILED ❌

- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **Config**: configs/lab/qwen3-4b-optimized.toml (v3)
- **Platform**: Prime Intellect Lab Hosted
- **Run ID**: yyjibqp3dtj3e2r5r26qnf00
- **WandB**: beautifulsoup-rl/qwen3-4b-optimized-v3-500steps
- **Status**: **FAILED** ❌ (crashed at step 10)
- **Target Steps**: 500
- **Crash Point**: Step 10 during `load_lora_adapter` weight sync

#### Crash Analysis (FOR PRIME TEAM)

**This run crashed consistently at step 8-10, same as previous attempts. Full investigation and ablation study below.**

---

## Lab Hosted Stability Investigation (2026-01-14)

### Executive Summary

Lab Hosted runs with `mode=all` (full task diversity) consistently crash at step 8-10 during LoRA weight sync. **Root cause CONFIRMED: sandbox memory exhaustion from large HTML documents.**

**Solution**: Set `memory_gb=16` in env args for `mode=all` training. The default 4GB is insufficient for BeautifulSoup to parse 60-100KB HTML documents.

**Key Evidence**:
| Config | memory_gb | Mode | Result |
|--------|-----------|------|--------|
| Failed runs | 4 | all | ❌ Crashed step 8-10 |
| High-memory test | 16 | all | ✅ **Passed step 11+** |
| Bootstrap runs | 4 | bootstrap | ✅ Passed (small HTML) |

### Crash Logs (Full Traceback)

```
File "/app/src/prime_rl/orchestrator/orchestrator.py", line 248, in orchestrate
    update_policy_task.result()  # Raises if the task failed
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/app/src/prime_rl/orchestrator/scheduler.py", line 163, in update_policy_loop
    await self.update_policy()
File "/app/src/prime_rl/orchestrator/scheduler.py", line 191, in update_policy
    await update_weights(
File "/app/src/prime_rl/utils/client.py", line 124, in update_weights
    await load_lora_adapter(admin_clients, lora_name, weight_dir)
File "/app/src/prime_rl/utils/client.py", line 196, in load_lora_adapter
    await asyncio.gather(*[_load_lora_adapter(admin_client) for admin_client in admin_clients])
File "/app/.venv/lib/python3.12/site-packages/tenacity/asyncio/__init__.py", line 189, in async_wrapped
    return await copy(fn, *args, **kwargs)
File "/app/src/prime_rl/utils/client.py", line 190, in _load_lora_adapter
    response = await admin_client.post(
File "/app/.venv/lib/python3.12/site-packages/httpx/_client.py", line 1730, in _send_single_request
    response = await transport.handle_async_request(request)
File "/app/.venv/lib/python3.12/site-packages/httpx/_transports/default.py", line 118, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.ConnectError: [Errno -2] Name or service not known
```

**Error Location**: `client.py:190` - POST to inference admin endpoint during LoRA weight reload

**Error Type**: DNS lookup failure - inference pod hostname no longer resolves (pod died)

### Observed Behavior Pattern

| Step | Behavior |
|------|----------|
| 0-5 | Normal operation, stable throughput 2000-2500 tok/s |
| 6-8 | Async barrier waits increase (7+ minutes observed) |
| 8-10 | Crash during `load_lora_adapter` weight sync |

**Key Metric**: When `time/step` exceeds 5+ minutes, crash is imminent. Normal step time is ~90-150s.

### Hypothesis: Inference Pod Resource Exhaustion

We believe the crash is caused by a "thundering herd" effect:

1. **Hard tasks generate long outputs**: With `mode=all`, tasks like regex extraction can produce 4096-token outputs with complex reasoning
2. **KV cache memory accumulates**: `batch_size=128` + `max_tokens=4096` = massive KV cache
3. **Async level allows orchestrator to race ahead**: With `max_async_level=2`, inference pod builds up a queue
4. **Weight sync attempts during high load**: When trainer finishes a step, it signals `load_lora_adapter`
5. **Pod runs out of resources**: The combination of KV cache memory + weight reload triggers OOM or timeout
6. **K8s health check fails**: Pod stops responding, K8s kills it, DNS no longer resolves

**Evidence for this hypothesis**:
- Crash always occurs during `load_lora_adapter` (weight sync)
- Long async barrier waits (7+ min) precede crash
- Bootstrap mode (simpler tasks, shorter outputs) never crashes

### Ablation Study Design

To isolate the cause, we ran 4 controlled 50-step tests:

| Test | Config | Key Changes | Run ID |
|------|--------|-------------|--------|
| **Baseline** | `qwen3-4b-bootstrap.toml` | `mode=bootstrap` only | skuz080dq7fuyst5jn26bfjn |
| **Small Batch** | `qwen3-4b-ablation-small.toml` | batch=64, rollouts=4 | vg7e1r0mjj146b3dy0ylxvf6 |
| **Async Level 1** | `qwen3-4b-ablation-async0.toml` | `max_async_level=1` | kfnq9gn73pb044hg7hb55idj |
| **Conservative** | `qwen3-4b-ablation-conservative.toml` | batch=32, async=1, tokens=1024 | z5ou53akdxrt5f0jzmnbv6lg |

### Ablation Results

| Test | Status | Steps | Best Reward | Avg Step Time | Notes |
|------|--------|-------|-------------|---------------|-------|
| Baseline (bootstrap) | ✅ COMPLETED | 50/50 | **99.7%** | ~90s | No crashes, stable throughput |
| Small Batch | ✅ COMPLETED | 50/50 | **100%** | ~60s | No crashes, faster steps |
| Async Level 1 | 🏃 RUNNING | TBD | TBD | TBD | In progress |
| Conservative | 🏃 RUNNING | TBD | TBD | TBD | In progress |

**Key Finding**: Both completed runs used `mode=bootstrap`. This is the critical variable.

### Comparison: Failed (mode=all) vs Success (mode=bootstrap)

| Metric | Failed Run (yyjibqp3dtj3e2r5r26qnf00) | Success Run (skuz080dq7fuyst5jn26bfjn) |
|--------|---------------------------------------|----------------------------------------|
| Mode | `all` (full task diversity) | `bootstrap` (primer + easy) |
| Steps Completed | 10/500 (2%) | 50/50 (100%) |
| Crash Point | Step 10, `load_lora_adapter` | N/A - no crash |
| Avg Step Time | 90-150s (with 7+ min spikes) | ~90s (consistent) |
| Max Seq Length | 4096 tokens | ~1600 tokens (shorter) |
| Async Barrier Waits | 7+ minutes observed | <30 seconds |

### Why Bootstrap Mode Works

`mode=bootstrap` in our environment selects only "primer" archetypes:
- Ultra-simple tasks like `<span id="target">Hello</span>` → extract "Hello"
- Model outputs are short (~500-1500 tokens)
- Low cognitive load = faster inference
- Less KV cache memory used
- No long reasoning chains

`mode=all` includes hard tasks:
- Complex HTML with nested structures
- Regex patterns requiring multi-step reasoning
- Limitation tasks requiring evidence extraction
- Model outputs can reach 4096 tokens
- Long reasoning chains = more tool calls = more latency

### Recommendations for Prime Team

#### 1. Inference Pod Health Monitoring

**Request**: Add observability into inference pod health:
- GPU memory utilization (especially KV cache)
- Request queue depth
- Time since last successful inference
- Proactive health warnings before crash

**Why**: Currently crashes are silent - we only see DNS failure after the fact. Early warnings would help users adjust configs before losing a run.

#### 2. Graceful Weight Sync Handling

**Request**: Implement graceful degradation during `load_lora_adapter`:
- If inference pod is under heavy load, delay weight sync
- Add backpressure mechanism to orchestrator
- Implement retry with exponential backoff before giving up

**Why**: The current behavior crashes the entire run on first weight sync failure. A more resilient approach would try to recover.

#### 3. User-Facing Guidance

**Request**: Document the relationship between:
- `batch_size` × `max_tokens` × `max_async_level` = memory pressure
- Task complexity → output length → KV cache usage
- Provide sizing guidance based on model size and task type

**Why**: Users currently have no guidance on safe parameter combinations. We discovered `mode=bootstrap` works through trial and error.

#### 4. Async Level 0 Support

**Request**: Allow `max_async_level=0` for fully synchronous operation.

**Why**: We tried to set `max_async_level=0` for our ablation test but got HTTP 422:
```
Input should be greater than or equal to 1
```

Fully synchronous mode would be valuable for debugging stability issues.

### Config Files Created

All configs in `configs/lab/`:

**qwen3-4b-bootstrap.toml** (baseline - stable):
```toml
model = "Qwen/Qwen3-4B-Instruct-2507"
max_steps = 50
batch_size = 128
rollouts_per_example = 8
trajectory_strategy = "interleaved"
max_async_level = 2
oversampling_factor = 2.0

[sampling]
max_tokens = 4096

[[env]]
id = "seconds-0/beautiful-soup-env"
args = { split = "train", mode = "bootstrap", cpu_cores = 2, memory_gb = 4 }
```

**qwen3-4b-ablation-conservative.toml** (maximum stability):
```toml
batch_size = 32               # Minimum
rollouts_per_example = 4      # Reduced
max_async_level = 1           # Minimum (0 not allowed)
oversampling_factor = 1.0     # No oversampling

[sampling]
max_tokens = 1024             # Shorter outputs
```

### Next Steps

1. Wait for remaining ablation tests to complete
2. If all pass, root cause confirmed as task complexity
3. Deploy production training with `mode=bootstrap` for initial model improvement
4. Gradually introduce harder tasks once model has baseline capability

---

### Early Progress (Before Crash)

| Step | Time | Reward | Throughput | Seq Length |
|------|------|--------|------------|------------|
| 0 | 94s | 13.0% | 2357 tok/s | 1725 |
| 1 | 143s | 11.3% | 1655 tok/s | 1850 |

---

## Speed & Throughput Comparison: Lab Hosted Optimizations

### Hyperparameter Changes (Default → Optimized)

| Parameter | Default | Optimized | Impact |
|-----------|---------|-----------|--------|
| `batch_size` | 32 | **128** | 4x more work per step, amortizes network overhead |
| `rollouts_per_example` | 4 | **8** | Better gradient quality, more diverse samples |
| `max_async_level` | 0 | **2** | Inference can run ahead of training |
| `oversampling_factor` | 1.0 | **2.0** | More rollouts in flight |
| `max_tokens` | 512 | **4096** | Matches production, avoids truncation |
| `cpu_cores` | 1 | **2** | Faster sandbox execution |
| `memory_gb` | 2 | **4** | Larger sandbox capacity |

### Throughput Comparison

| Run | Model | Avg Throughput | Peak Throughput | Avg Step Time |
|-----|-------|---------------|-----------------|---------------|
| Lab Hosted v2 | Qwen3-4B | ~2,800 tok/s | 6,394 tok/s | ~126s |
| Lab Hosted v3 | Qwen3-4B | ~2,000 tok/s | 2,357 tok/s | ~118s |
| Self-Hosted v3 (local) | Qwen3-8B | ~1,500 tok/s | ~2,500 tok/s | ~200s |
| Self-Hosted v4 (local) | Qwen3-8B | ~1,200 tok/s | ~2,000 tok/s | ~560s |

### Key Insight

**Lab Hosted is competitive with self-hosted for 4B-8B models**, with the main advantage being zero infrastructure management. The hyperparameter optimizations (larger batches, more rollouts, async inference) help maximize utilization of Prime's managed infrastructure.

**What we changed:** We bumped batch size 4x (32→128), doubled rollouts (4→8), and enabled async inference (max_async_level=2) to better saturate the distributed infrastructure - this lets the system pipeline work more efficiently instead of waiting between steps.

---

### Run: qwen3-4b-optimized-v2 (2026-01-14) - Lab Hosted - COMPLETED ✅

- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **Config**: configs/lab/qwen3-4b-optimized.toml (v2)
- **Platform**: Prime Intellect Lab Hosted
- **Run ID**: tea6fd7i0tlf5n4qztcit41p
- **WandB**: https://wandb.ai/seconds-0-domus-magna-inc/beautifulsoup-rl/runs/1dxl2hrq
- **Status**: **COMPLETED** ✅

#### Results

| Step | Time | Reward | Throughput | Solve None |
|------|------|--------|------------|------------|
| 0 | 108s | 17.9% | 2079 tok/s | 44% |
| 1 | 177s | 3.2% | 1439 tok/s | 75% |
| 2 | 35s | 11.1% | 6394 tok/s | 69% |
| 3 | 65s | 17.8% | 3741 tok/s | 44% |
| 4 | 208s | 6.1% | 1229 tok/s | 62% |
| 5 | 115s | 32.7% | 2046 tok/s | 19% |
| 6 | 52s | 38.5% | 4976 tok/s | 38% |
| 7 | 86s | 30.0% | 3008 tok/s | 38% |
| 8 | 207s | **52.8%** | 1211 tok/s | 25% |
| 9 | 211s | **50.3%** | 1289 tok/s | 25% |

**Summary:**
- Total time: 21.1 minutes (10 steps)
- Best reward: **52.78%** (step 8)
- Final reward: **50.29%**
- Average step time: ~2 min/step

#### Key Improvements from v1

1. **Added per-worker logging** via `[env.log]` config
2. **Created WandB monitoring script** (`scripts/wandb_alert_monitor.py`)
3. **Added verbose WandB logging** via `[wandb.log_extras]`
4. **Increased max_steps** to 10 (was 5)

#### Diagnostic Patterns Validated

When inference server dies (observed in v1), you see:
- Throughput drop >50% from peak
- `time/update_weights` DECREASES (paradoxically - timeouts complete faster)
- `batch/solve_none` spikes to >70%

This run showed stable `time/update_weights` at 8.3s throughout = healthy.

---

### Run: qwen3-4b-optimized-v1 (2026-01-13) - Lab Hosted - CRASHED ⚠️

- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **Config**: configs/lab/qwen3-4b-optimized.toml
- **Platform**: Prime Intellect Lab Hosted
- **Run ID**: v0bl0g0i7nhel34ypaoftrt2
- **Status**: CRASHED (network error at step 3)
- **W&B Project**: beautifulsoup-rl

#### Results

| Step | Time | Reward | Throughput | Notes |
|------|------|--------|------------|-------|
| 0 | 96s | 8.06% | 2374 tok/s | Base model, no sync |
| 1 | 19.6 min | 10.32% | 201 tok/s | First checkpoint sync |
| 2 | 44s | **22.58%** | 4916 tok/s | System warmed up |
| 3 | 2.5 min | 6.78% | 1533 tok/s | Crashed after logging |

**Key findings:**
- **Step 1 bottleneck confirmed** - First weight sync takes ~20 min (expected architectural constraint)
- **Step 2 was very fast** - Once system warms up, subsequent steps are much faster
- **Excellent reward progression** - 8% → 22.6% in 3 steps shows strong RL signal

#### Step Timing Analysis (Research Synthesis)

All three research sources (Codex, Gemini, Explore agent) confirmed:

1. **Step 0 is fast** because:
   - Uses pre-loaded base model (no weight sync)
   - No checkpoint dependency
   - Inference servers ready immediately

2. **Step 1 is slow** because:
   - Trainer must complete training on step 0 data
   - Weight broadcast to filesystem (`broadcasts/step_1/STABLE`)
   - Orchestrator waits for `STABLE` marker (1-second polling)
   - All inference servers reload weights (blocking operation)
   - NCCL handshake if using NCCL broadcast

3. **Step 2+ can be faster** because:
   - Filesystem caches populated
   - Weight loading path warm
   - No "cold start" penalty

**Metrics to check in WandB:**
- `time/wait_for_ckpt` - Time waiting for checkpoint marker
- `time/update_weights` - Time updating inference server weights
- `time/broadcast_weights` - Trainer-side weight broadcast time

#### Error

```
httpx.ConnectError: [Errno -2] Name or service not known
```

Transient network error in Lab Hosted infrastructure. Not a config issue.

#### Config

```toml
model = "Qwen/Qwen3-4B-Instruct-2507"
max_steps = 5
batch_size = 128
rollouts_per_example = 8
trajectory_strategy = "interleaved"
max_async_level = 2
oversampling_factor = 2.0

[sampling]
max_tokens = 4096

[[env]]
id = "seconds-0/beautiful-soup-env"
args = { split = "train", mode = "all", cpu_cores = 2, memory_gb = 4 }

[wandb]
project = "beautifulsoup-rl"
name = "qwen3-4b-optimized-v1"
```

#### Lessons Learned

1. **First weight sync is slow by design** - Not a config issue, architectural constraint
2. **max_tokens=4096 is correct** - 512 caused truncation (reward dropped from 9% to 6%)
3. **batch_size=128 works** - Better than 32 for amortizing network overhead
4. **Lab Hosted can show good RL signal** - 22.6% reward at step 2 is excellent

---

### Run: bs4-rl-qwen3-8b-2xh100-v4-resilient (2026-01-06 → 2026-01-12) - COMPLETED ✅

- **Model**: Qwen/Qwen3-8B (8.2B params)
- **Config**: /root/config_v4_fixed.toml (2x H100 DataCrunch spot)
- **Pod**: DataCrunch 2x H100 80GB (spot instance)
- **Status**: COMPLETED ✅ (terminated by spot preemption at step 920)
- **W&B Project**: beautiful-soup-env
- **Final Step**: 920 / 1000 (92% complete)
- **Mean Reward**: ~0.60 (plateau from step 775 onwards)
- **Best Reward**: 0.86
- **Checkpoint**: `b2://beautifulsoup-rl/bs4-qwen3-8b-v4-resilient/step_920/`

#### Final Summary

| Metric | Value |
|--------|-------|
| Total Steps | 920 |
| Mean Reward | ~0.60 |
| Best Reward | 0.86 |
| Total Runtime | ~6 days |
| Hardware | 2x H100 (DataCrunch spot) |
| LoRA Rank | 8 |
| Target Modules | q,k,v,o,gate,up,down_proj |

#### Training Timeline

1. **2026-01-06**: Started fresh from step 0
2. **2026-01-09**: Pod crashed at step ~780
3. **2026-01-10**: Resumed from step 775 checkpoint
4. **2026-01-11 09:19**: Training stuck at step 857, restarted from step 855
5. **2026-01-12 00:45**: Training stuck at step 935, restarted
6. **2026-01-12 10:05**: Pod terminated by spot preemption (step 935 local, step 920 synced to B2)

#### Key Observations

- **No grokking observed** - Rewards plateaued at ~0.60 from step 775 onwards
- **Resilient checkpoint system worked** - B2 sync every 5 min saved us multiple times
- **Spot instances are risky** - DataCrunch terminated without warning

#### Artifacts

- **B2 Checkpoint**: `b2://beautifulsoup-rl/bs4-qwen3-8b-v4-resilient/step_920/`
- **LoRA Export**: Pending (use `scripts/convert_checkpoint_to_peft.py`)
- **Can resume**: Yes, from step 920 with `--ckpt.resume-step 920`

#### Recent Investigation (2026-01-07 16:55 UTC)

WandB monitoring reported training stalled at step 390 for ~100 minutes. SSH investigation showed:
- **Training is HEALTHY** - step 395 completed at 16:54 UTC
- Both GPUs active (GPU0: 66% util, GPU1: trainer running)
- Disk at 71% (53GB free) - healthy
- Cleanup/sync daemons running

**Root cause**: WandB metrics not syncing to API (one 502 error logged on Jan 6). Training continued locally but WandB dashboard lagged behind.

**Also fixed** workflow bugs that prevented auto-recovery from detecting the run:
- RUN_ID mismatch in `.github/workflows/training-monitor.yml` (was `bs4-qwen3-8b`, should be `bs4-rl-qwen3-8b-2xh100-v4-resilient`)
- NoneType bug in `training_controller.py` and `provision_vast.py` when instance has null label

#### Previous Investigation (2026-01-06 11:30 UTC)

Training appeared stalled at step 98 but actually was progressing. Investigation showed:
- WandB sampling lag caused apparent 141-min gap
- Actual gaps were 30-33 min (checkpointing/disk cleanup)
- Training continued to step 101

A backup Vast.ai pod (instance 29547800) was provisioned but terminated after confirming Prime pod was healthy

#### Incidents and Recovery

**1. vLLM Crash After Step 6 (03:00 UTC)**
- Inference server on GPU 0 died silently
- GPU 0 showed 0 MiB memory
- Trainer stuck waiting for batches

**Recovery:**
```bash
# Kill all processes
pkill -9 python

# Restart from checkpoint step 5
uv run rl @ /root/config.toml --ckpt --ckpt.resume-step 5 --ckpt.interval 5 --ckpt.keep-last 2
```

Training resumed successfully from step 5 at 03:15 UTC.

**2. Disk Space Management (RESOLVED)**
- Checkpoints are ~47GB each (31GB checkpoint + 16GB weights)
- `--ckpt.keep-last 2` does NOT auto-delete old checkpoints

**Solution deployed (2026-01-06 09:22 UTC):**
1. **Disk cleanup daemon** (`/root/disk_cleanup.sh`): Automatically deletes oldest checkpoint when disk > 85%
2. **B2 sync daemon** (`/root/periodic_sync.sh`): Syncs latest checkpoint to Backblaze B2 every 10 min
3. **Updated `scripts/checkpoint_sync.sh`**: Now cleans up old local checkpoints after successful B2 sync

**Daemons running:**
```bash
# Verify with:
ps aux | grep -E "(cleanup|sync)" | grep -v grep
```

#### Config (v4 - Resilient)

```toml
inference_gpu_ids = [0]
trainer_gpu_ids = [1]
max_steps = 1000

[model]
name = "Qwen/Qwen3-8B"

[wandb]
project = "beautiful-soup-env"
name = "bs4-rl-qwen3-8b-2xh100-v4-resilient"

[trainer.optim]
lr = 1e-5
weight_decay = 0.0

[trainer.model]
seq_len = 4096

[trainer.model.lora]
rank = 8
alpha = 32
dropout = 0.0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

[orchestrator]
batch_size = 128
rollouts_per_example = 8
seq_len = 4096
oversampling_factor = 2.0
lora_name = "qwen3-8b-bs4-lora"

[orchestrator.sampling]
max_tokens = 4096
temperature = 0.7

[orchestrator.buffer]
online_difficulty_filtering = true

[[orchestrator.env]]
id = "seconds-0/beautiful-soup-env"

[orchestrator.env.args]
split = "train"
mode = "tiered"
difficulty = "mixed"
seed = 42
executor_backend = "local"
network_access = true
timeout_s = 30.0
max_output_chars = 10000

[inference]
gpu_memory_utilization = 0.90

[inference.model]
enable_auto_tool_choice = true
tool_call_parser = "hermes"
enforce_eager = true
```

#### Lessons Learned

1. **vLLM can crash silently** - Monitor GPU memory, not just process count
2. **Checkpoint cleanup doesn't work** - `--ckpt.keep-last N` doesn't delete old checkpoints
3. **47GB per checkpoint** - Plan for manual cleanup every 5 steps with 65GB free disk
4. **Resume works well** - Training successfully resumes from checkpoint with full continuity

---

### Run: bs4-rl-qwen3-8b-2xh100-v3 (2026-01-05) - RUNNING ✅

- **Model**: Qwen/Qwen3-8B (8.2B params)
- **Config**: configs/prime-rl/qwen3-8b-2xh100.toml:v3
- **Pod**: 2x H100 80GB (bs4-llama-2xh100)
- **Status**: RUNNING ✅
- **W&B Run**: https://wandb.ai/seconds-0-domus-magna-inc/beautiful-soup-env/runs/r1hgf2ie
- **Step Time**: ~3-4 minutes (down from 27 min with v2!)
- **Speedup**: **7-8x** achieved

#### v3 Fix: Local Executor (CRITICAL!)

**Root cause found:** Training was LATENCY-BOUND, not throughput-bound!

| Setting | v2 | v3 | Impact |
|---------|----|----|--------|
| `executor_backend` | `"prime"` | `"local"` | **10-15x speedup** |
| `rollouts_per_example` | 4 | 8 | Restored (local is fast) |
| `oversampling_factor` | 1.0 | 2.0 | Restored (local is fast) |

#### Step Time Comparison

| Config | Step Time | Per-Rollout | Bottleneck |
|--------|-----------|-------------|------------|
| v2 (prime executor) | ~27 min | 3.2s | Network latency |
| **v3 (local executor)** | **~3-4 min** | ~0.2s | GPU-bound |

#### Incident: Inference Server Crash (Step 5)

**What happened:**
- Training crashed after step 5 completed
- Inference server (GPU 0) died silently - 0% utilization, 0 memory
- Orchestrator kept retrying but got `APIConnectionError('Connection error.')` for all 128 rollouts
- Training appeared "stuck" for ~3 hours with no progress

**Diagnosis:**
- Original logs were overwritten when training was restarted
- Found warning in logs: `Found ulimit of 32000 and failed to automatically increase`
- vLLM recommends 65536 file descriptors; 32000 may cause "Too many open files" errors
- Most likely causes: FD exhaustion, GPU memory fragmentation, or CUDA context corruption

**Recovery:**
```bash
# Kill stuck processes
ps aux | grep -E "(rl|vllm|ray)" | grep -v grep | awk '{print $2}' | xargs -r kill -9

# Resume from checkpoint
uv run rl @ /tmp/config.toml --ckpt --ckpt.resume-step 5 --ckpt.interval 5 --ckpt.keep-last 3
```

**Fix Applied:**
- Added `ulimit -n 65536` to `scripts/pod_setup.sh`
- Added to `.bashrc` for persistence

---

### Run: bs4-rl-qwen3-8b-2xh100-v2 (2026-01-05) - POSTMORTEM ⚠️

- **Model**: Qwen/Qwen3-8B (8.2B params)
- **Config**: configs/prime-rl/qwen3-8b-2xh100.toml:v2
- **Pod**: 2x H100 80GB
- **Status**: POSTMORTEM ⚠️ (optimizations failed, root cause found)
- **Step Time**: ~27 min (only 10% improvement from v1!)
- **Expected**: ~4-6 min (5-8x speedup)
- **Actual**: ~27 min (10% speedup)

#### v2 Postmortem: Wrong Bottleneck Identified

**What we tried:**
- Reduced max_tokens (10k → 4k) - No impact
- Reduced rollouts (8 → 4) - Made per-rollout time WORSE
- Added prefix caching - Not supported by prime-rl
- Added max_num_seqs, max_num_batched_tokens - Not supported

**Why it failed:**
Multi-turn tool-calling RL is **LATENCY-BOUND**, not throughput-bound!

- Remote sandbox (`executor_backend = "prime"`) adds ~1.5s per tool call
- 1024 rollouts × 1.5s = ~25 minutes of waiting (GPUs idle!)
- Reducing rollouts to 512 didn't reduce latency, just did less work

**The fix:** Switch to `executor_backend = "local"` (see v3)

#### Lessons Learned

1. **Profile before optimizing** - We optimized inference params, but tool execution was the bottleneck
2. **Latency vs throughput** - Multi-turn interactions are latency-bound
3. **Reducing work doesn't reduce fixed latency** - 512 rollouts took same time as 1024
4. **Question defaults** - `executor_backend="prime"` is safe but slow

---

### Run: bs4-rl-qwen2.5-7b-h100-v6 (2026-01-05) - RUNNING ✅

- **Model**: Qwen/Qwen2.5-7B-Instruct (7B params)
- **Config**: /tmp/config.toml (remote) - see below
- **Pod**: 1x H100 80GB SXM5 (DataCrunch spot, $0.99/hr)
- **Pod ID**: fd6a88cad0f04019841355376d088f5b
- **Start**: 2026-01-05 00:16 UTC
- **Status**: RUNNING ✅
- **W&B Run**: https://wandb.ai/seconds-0-domus-magna-inc/beautiful-soup-env/runs/189rkyj5
- **GPU Usage**: 72GB/81GB (89%) with activation checkpointing
- **Baseline**: N/A (OpenRouter doesn't support tool calling for Qwen2.5-7B)
- **Env Version**: 0.1.1 (fixed reward function crash on malformed JSON)

---

### Run: bs4-rl-qwen2.5-7b-h100-v5 (2026-01-05) - ABANDONED ⚠️

- **Model**: Qwen/Qwen2.5-7B-Instruct (7B params)
- **Config**: /tmp/config.toml (remote)
- **Pod**: 1x H100 80GB SXM5 (DataCrunch spot, $0.99/hr)
- **Pod ID**: fd6a88cad0f04019841355376d088f5b
- **Start**: 2026-01-04 23:50 UTC
- **End**: 2026-01-05 00:14 UTC (killed for v6 restart)
- **Status**: ABANDONED ⚠️ (reward function bug caused errors, restarted with v0.1.1)
- **W&B Run**: https://wandb.ai/seconds-0-domus-magna-inc/beautiful-soup-env/runs/t6agrtyb
- **GPU Usage**: 74GB/81GB (91%) with activation checkpointing
- **Baseline**: N/A (OpenRouter doesn't support tool calling for Qwen2.5-7B)
- **Progress**: 1/1000 steps completed (reward 0.0238, 287s, 50 tokens/s)
- **Issue**: Reward function crashed on malformed JSON: `'list' object has no attribute 'get'`
- **Fix**: Published env v0.1.1 with guard against malformed JSON arguments

#### Working Config (H100 Single GPU - Final v5)

```toml
# Key settings for single H100 with shared trainer/inference
# v5: Fixed max_tokens to fit within max_model_len
inference_gpu_ids = [0]
trainer_gpu_ids = [0]
max_steps = 1000

[model]
name = "Qwen/Qwen2.5-7B-Instruct"

[trainer.model]
seq_len = 2048

[trainer.model.ac]
freq = 1                  # CRITICAL: Full activation checkpointing saves ~20GB

[trainer.model.lora]
rank = 8
alpha = 32

[orchestrator]
batch_size = 8            # Minimal for single GPU
rollouts_per_example = 2  # Minimal for single GPU
seq_len = 2048

[orchestrator.sampling]
max_tokens = 2000         # CRITICAL: Must fit within max_model_len - input_tokens

[inference]
gpu_memory_utilization = 0.50

[inference.model]
enable_auto_tool_choice = true
tool_call_parser = "hermes"
enforce_eager = true
max_model_len = 4096
```

#### Memory Optimization Journey

1. **First attempt (OOM)**: gpu_memory_utilization = 0.45, no checkpointing → 80GB total, OOM
2. **Second attempt (device mismatch)**: fsdp_cpu_offload = true → RuntimeError: device mismatch CPU/CUDA
3. **Third attempt (config error)**: ac = "selective" → ValidationError: expects config object
4. **Fourth attempt (v4 - stuck)**: ac = {freq = 1} → 72GB used, but max_tokens=4000 > available context
5. **Fifth attempt (v5 - SUCCESS)**: max_tokens = 2000 → Training running!

#### Model Selection (2026-01-05)

**Winner: Qwen/Qwen2.5-7B-Instruct** (Ministral failed to load)

| Model | Tool Calls | Baseline | vLLM | Status |
|-------|------------|----------|------|--------|
| Ministral-3-8B | ✅ 1148 | 63.2% | ❌ KeyError | transformers doesn't support |
| **Qwen2.5-7B** | N/A | N/A | ✅ Works | **SELECTED** |
| Qwen3-8B | ❌ 0 calls | 39.6% | ? | Doesn't use tools |
| gpt-oss-20b | N/A | 49.2% | ❌ vLLM bug | Blocked |

**Key fixes for single-GPU training:**
1. `[inference] gpu_memory_utilization = 0.50` - leave room for trainer
2. `[trainer.model.ac] freq = 1` - activation checkpointing (NOT fsdp_cpu_offload)
3. Reduced batch_size (8) and rollouts_per_example (2)

---

### Run: bs4-rl-qwen2.5-7b-1000steps (2026-01-04) - ABANDONED

- **Model**: Qwen/Qwen2.5-7B-Instruct (7B params)
- **Config**: configs/prime-rl/beautiful-soup-env.toml
- **Pod**: bs4-rl-training (2x A6000 48GB)
- **Start**: 2026-01-04 21:03 UTC
- **Status**: ABANDONED ❌ (switched to Ministral-8B)
- **W&B Project**: beautiful-soup-env
- **W&B Run**: https://wandb.ai/seconds-0-domus-magna-inc/beautiful-soup-env/runs/aochk8k1
- **Baseline**: Could not benchmark (OpenRouter doesn't support tool calling for this model)

#### Config (v8 - Qwen2.5-7B)
```toml
max_steps = 1000
inference_gpu_ids = [0]
trainer_gpu_ids = [1]

[model]
name = "Qwen/Qwen2.5-7B-Instruct"  # Widely supported, good tool calling

[trainer.model.experimental.lora]
rank = 8
alpha = 32

[orchestrator]
batch_size = 128        # 7B model allows larger batches
rollouts_per_example = 8
seq_len = 8192
```

#### Model Selection Journey
1. **gpt-oss-20b** - vLLM weight reload bug (`default_weight_loader() got unexpected keyword argument 'weight_name'`)
2. **qwen/qwen3-vl-8b-instruct** - VL (Vision-Language) model, not supported by AutoModelForCausalLM
3. **mistralai/mistral-small-3.2-24b-instruct** - Not on HuggingFace Hub
4. **Qwen/Qwen2.5-7B-Instruct** - ✅ Current choice (7B, text-only, widely supported)

#### Issues Encountered

1. **Config format errors** (attempt 1):
   - `seq_len` must be under `[orchestrator]`, not top-level
   - `top_p` not allowed in `[orchestrator.sampling]`
   - `[orchestrator.buffer]` uses `online_difficulty_filtering = true`, not `type = "online-difficulty"`

2. **WandB API key missing** (attempt 2):
   - Error: `No API key configured`
   - Fix: Set `offline = true` in config, or configure `/root/.netrc`

3. **CUDA OOM with gpt-oss-20b** (attempt 3):
   - Error: `CUDA out of memory. Tried to allocate 2.16 GiB`
   - Root cause: 20B model weights (~40GB BF16) + inference server using 42.83GB on GPU 0
   - Fix: Enable `fsdp_cpu_offload = true`, reduce batch_size to 32, seq_len to 8192

4. **vLLM weight reload bug with gpt-oss-20b** (attempt 4):
   - Error: `TypeError: default_weight_loader() got an unexpected keyword argument 'weight_name'`
   - Root cause: `gpt-oss-20b` model loader incompatible with vLLM `/reload_weights` endpoint
   - Fix: Switch to different model

5. **qwen3-vl-8b is Vision-Language model** (attempt 5):
   - Error: `Unrecognized configuration class Qwen3VLConfig for AutoModelForCausalLM`
   - Root cause: VL models require different model class (AutoModelForVision2Seq)
   - Fix: Switch to text-only model

6. **Mistral model not on HuggingFace** (attempt 6):
   - Error: `mistralai/mistral-small-3.2-24b-instruct is not a valid model identifier`
   - Root cause: Model only available via Mistral API, not on HF Hub
   - Fix: Switch to Qwen/Qwen2.5-7B-Instruct

7. **Final model selection** (attempt 7-8):
   - Qwen/Qwen2.5-7B-Instruct: 7B params, text-only, widely supported, good tool calling
   - No CPU offloading needed on 2x A6000 (7B fits comfortably)

8. **vLLM V1 engine CUDA segfault** (attempt 9-10):
   - Error: `Segfault encountered` in `THCPModule_initExtension` / `cuCtxGetDevice`
   - Root cause: vLLM V1 engine spawns child processes that inherit invalid CUDA context
   - Fix: Set environment variables before launching:
     ```bash
     export VLLM_USE_V1=0
     export VLLM_WORKER_MULTIPROC_METHOD=spawn
     ```
   - Also added `enforce_eager = true` in config to disable CUDA graphs

#### WandB Setup on Pod
```bash
# Configure API key
echo -e 'machine api.wandb.ai\n  login user\n  password YOUR_KEY' > /root/.netrc
chmod 600 /root/.netrc
```

---

## Completed Runs

| Run ID | Model | Start | Duration | Final Reward | Notes |
|--------|-------|-------|----------|--------------|-------|
| bs4-qwen3-8b-v4-resilient | Qwen3-8B | 2026-01-06 | 6 days | 0.60 avg | Step 920/1000, spot preempted |

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

| Model | Baseline (n=680) | Cost | Training Priority |
|-------|------------------|------|-------------------|
| openai/gpt-oss-20b | **49.2%** | $0.07/$0.30 | **PRIMARY TARGET** |
| prime-intellect/intellect-3 | ~78% | $0.20/$1.10 | Good but high baseline |
| qwen/qwen3-235b-a22b-instruct-2507 | 71.2% | $0.22/$0.88 | Good candidate |
| z-ai/glm-4.5-air | 66.8% | $0.20/$1.10 | Room for improvement |

## Lessons Learned

### Executor Backend: Local vs Prime (CRITICAL!)

**Multi-turn tool-calling RL is LATENCY-BOUND, not throughput-bound!**

| Backend | Tool Call Latency | 1024 Rollouts | Use Case |
|---------|-------------------|---------------|----------|
| `"prime"` | ~1.5 seconds | ~25 minutes | Production (secure) |
| `"local"` | ~0.05 seconds | **~1 minute** | Training on dedicated pods |

```toml
[orchestrator.env.args]
executor_backend = "local"    # CRITICAL: 10-15x speedup for training!
```

**Why this matters:**
- Remote sandbox adds network latency per tool call
- GPUs sit idle waiting for tool execution
- Reducing rollouts doesn't reduce fixed latency
- Always use `"local"` for training on dedicated/ephemeral pods

**Safe to use local because:**
- BeautifulSoup tasks only parse HTML (no file/network access)
- Training pods are ephemeral (no persistent data at risk)

### 2-GPU Setup (2x A6000 48GB)

1. **Memory**: 20B model (BF16) needs ~40GB. Split inference/trainer across GPUs.
2. **CPU Offload**: Required for FSDP with limited GPU memory. Add `fsdp_cpu_offload = true` under `[model]`.
3. **Batch Size**: Start small (32) and increase if memory allows.
4. **seq_len**: 8192 is safe; 16384 may cause OOM with large models.

### Config Format (prime-rl)

- `seq_len` → under `[orchestrator]`, NOT top-level
- `top_p` → NOT allowed in sampling config
- `[orchestrator.buffer]` → use `online_difficulty_filtering = true`
- `[[orchestrator.env]]` → double brackets, no `name` field needed

### WandB on Pod

```bash
# Option 1: .netrc file
echo -e 'machine api.wandb.ai\n  login user\n  password YOUR_KEY' > /root/.netrc
chmod 600 /root/.netrc

# Option 2: Environment variable
export WANDB_API_KEY=YOUR_KEY
```

### vLLM CUDA Segfault Fix

If you see `THCPModule_initExtension` or `cuCtxGetDevice` segfault in vLLM:

```bash
# Set before launching training
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Also add to config
[inference.model]
enforce_eager = true
```

This disables V1 engine and uses spawn instead of fork for child processes.

### Single-GPU Memory Sharing (Trainer + Inference)

When trainer and inference share the same GPU (e.g., single H100):

```toml
# CRITICAL: Limit vLLM memory so trainer has room
[inference]
gpu_memory_utilization = 0.45  # Default 0.9 will fail

[orchestrator]
batch_size = 32           # Reduce from 64+
rollouts_per_example = 4  # Reduce from 8
```

**Error when this is wrong:**
```
ValueError: Free memory on device (48.97/79.18 GiB) on startup is less than
desired GPU memory utilization (0.9, 71.26 GiB).
```

The trainer loads first (~30GB for 7B model), then vLLM tries to claim 90% = OOM.

### Ministral Model Compatibility

Ministral-3-8B-Instruct-2512 causes `KeyError: 'ministral3'` in transformers:
- Root cause: Model type not recognized by AutoModelForCausalLM
- Workaround: Use Qwen2.5-7B-Instruct instead (similar size, vLLM compatible)

### Activation Checkpointing (Recommended for Single-GPU)

Use activation checkpointing instead of CPU offload for single-GPU training:

```toml
[trainer.model.ac]
freq = 1  # Full checkpointing, saves ~20GB for 7B model
```

**Do NOT use `fsdp_cpu_offload = true`** with LoRA - it causes device mismatch errors:
```
RuntimeError: Attempted to set the storage of a tensor on device "cpu"
to a storage on different device "cuda:0"
```

### Single-GPU RL Training Checklist

For running trainer + vLLM inference on one GPU (e.g., H100 80GB):

1. `[inference] gpu_memory_utilization = 0.50` (or less)
2. `[trainer.model.ac] freq = 1` (activation checkpointing)
3. `[trainer.model] seq_len = 2048` (reduce from 4096)
4. `[orchestrator] batch_size = 8, rollouts_per_example = 2`
5. `[inference.model] enforce_eager = true, max_model_len = 4096`
6. `[orchestrator.sampling] max_tokens = 2000` (must fit within context!)

### max_tokens vs max_model_len Constraint

`max_tokens` (in `[orchestrator.sampling]`) must leave room for input tokens:

```
max_tokens + input_tokens <= max_model_len
```

With `max_model_len = 4096` and typical inputs of ~1300-1500 tokens, use `max_tokens = 2000`.

**Error when this is wrong:**
```
ValueError: 'max_tokens' or 'max_completion_tokens' is too large: 4000.
This model's maximum context length is 4096 tokens and your request has 1297 input tokens
(4000 > 4096 - 1297).
```

**Symptom:** Rollout generation hangs indefinitely at "0/8 rollouts" with no errors in main log.
The actual error appears in the vLLM process logs. Check `/tmp/vllm.log` if rollouts stall.

### Spot Instance Risks

Spot instances (e.g., DataCrunch $0.99/hr H100) can be reclaimed at any time:
- Training may stop without warning after 1-2 steps
- WandB shows "running" even after pod is terminated (no graceful shutdown)
- SSH connection fails when pod is reclaimed

**Detection:** If WandB runtime stops increasing and last update is 10+ minutes old, the pod is dead.

**Mitigation options:**
1. **Use on-demand instances** - More expensive but guaranteed availability
2. **Implement checkpointing** - Save LoRA adapters frequently, resume from last checkpoint (see below)
3. **Use Prime's pods** - More stable than cheap spot providers
4. **Monitor aggressively** - Check WandB every 5 minutes, restart quickly if dead

### Checkpointing (CRITICAL for Spot Instances)

**Checkpointing is OFF by default in prime-rl!** You must explicitly enable it via CLI flags.

| Flag | Purpose |
|------|---------|
| `--ckpt` | Enable checkpointing (OFF by default!) |
| `--ckpt.interval N` | Save every N steps |
| `--ckpt.keep-last N` | Keep only last N checkpoints |
| `--ckpt.save-async` | Non-blocking saves |
| `--ckpt.resume-step N` | Resume from checkpoint |

**Checkpoints saved to:** `checkpoints/step_N/`

**Example: 25-step validation run with checkpointing**
```bash
uv run rl @ configs/prime-rl/qwen2.5-7b-h100.toml \
  --ckpt --ckpt.interval 5 --ckpt.keep-last 3 \
  --max-steps 25
```

**Resume from checkpoint:**
```bash
uv run rl @ configs/prime-rl/qwen2.5-7b-h100.toml \
  --ckpt.resume-step 20 --max-steps 50
```

**What gets checkpointed:**
- **Trainer**: FSDP model shards, optimizer/scheduler state, progress metrics
- **Orchestrator**: Progress (step, tokens, samples)
- **Inference**: Nothing (stateless) - orchestrator reloads correct weights automatically

**WARNING: `--ckpt.keep-last N` does NOT auto-delete old checkpoints!**

As of 2026-01-06, the `--ckpt.keep-last` flag does not actually clean up old checkpoints. Both checkpoints and weights directories accumulate:
- Each checkpoint: ~31GB (optimizer state, model shards)
- Each weight save: ~16GB (model weights)
- Total per step: ~47GB

**Manual cleanup required:**
```bash
# After step 15 checkpoint is saved, delete step 10
rm -rf /app/outputs/checkpoints/step_10 /app/outputs/weights/step_10
```

Plan for disk to fill to ~90% at each checkpoint save, then manually clean up to recover space.

---

## Resilient Training System

Automated checkpoint sync and recovery for spot instance training.

### Architecture

```
+-------------------------------------------------------------------+
|                         TRAINING POD                              |
|  +-------------+    +-------------+    +-----------------------+  |
|  |  prime-rl   |--->| checkpoints |--->| checkpoint_sync.sh    |  |
|  |  training   |    |   /ckpt/    |    | (every 5 min)         |  |
|  +-------------+    +-------------+    +-----------+-----------+  |
|                                                    |               |
|                                                    v               |
|                                          +-----------------------+ |
|                                          |   Backblaze B2        | |
|                                          |  beautifulsoup-rl     | |
|                                          +-----------------------+ |
+-------------------------------------------------------------------+

+-------------------------------------------------------------------+
|                   GITHUB ACTIONS (Always-On)                      |
|  +-----------------------+    +-------------------------------+   |
|  |  training-monitor.yml |--->|  training_controller.py       |   |
|  |  (every 10 min)       |    |  Check WandB, auto-provision  |   |
|  +-----------------------+    +-------------------------------+   |
+-------------------------------------------------------------------+
```

### Components

| Script | Purpose |
|--------|---------|
| `scripts/checkpoint_sync.sh` | Atomic sync of checkpoints to B2 with latest.json pointer |
| `scripts/onstart.sh` | Vast.ai auto-start: pull checkpoint + resume |
| `scripts/training.service` | Systemd unit for process supervision |
| `scripts/training_controller.py` | GitHub Actions: monitor WandB + auto-provision |
| `scripts/provision_vast.py` | Manual Vast.ai provisioning |
| `scripts/wandb_monitor.py` | WandB health check with stall detection |

### Quick Start

```bash
# 1. Set up environment (on pod)
curl -sSL https://raw.githubusercontent.com/seconds-0/beautifulsoup-rl/main/scripts/pod_setup.sh | bash

# 2. Start training (checkpoints auto-sync to B2)
tmux new -s training
uv run rl @ /tmp/config.toml --ckpt --ckpt.interval 5 --ckpt.keep-last 3
```

### After Spot Interruption (Automatic)

1. Vast.ai pauses instance when outbid
2. When price drops, instance resumes
3. `onstart.sh` runs automatically:
   - Pulls latest checkpoint from B2
   - Resumes training from that step
4. Training continues seamlessly

### GitHub Actions Controller

The `.github/workflows/training-monitor.yml` workflow runs every 10 minutes:
- Checks WandB for training status (healthy/failed/stalled)
- Auto-provisions new Vast.ai instance if training died
- Idempotent (safe to run repeatedly)

**Required GitHub Secrets:**
- `WANDB_API_KEY`
- `VAST_API_KEY`
- `B2_APPLICATION_KEY_ID`
- `B2_APPLICATION_KEY`

### Manual Operations

```bash
# Check training status
python scripts/wandb_monitor.py --run-id bs4-qwen3-8b

# Search for Vast.ai instances
python scripts/provision_vast.py search --gpu H100 --count 2 --max-price 2.50
python scripts/provision_vast.py search --gpu 4090 --count 2 --max-price 0.50  # Much cheaper!

# Provision new instance
python scripts/provision_vast.py create --run-id bs4-qwen3-8b --gpu H100 --count 2
python scripts/provision_vast.py create --run-id bs4-qwen3-8b --gpu 4090 --count 2 --max-price 0.50

# Terminate instances
python scripts/provision_vast.py terminate --run-id bs4-qwen3-8b --force
```

### Deployment Checklist

Before deploying a training pod:

1. **Vast.ai Account**
   - [ ] Account has credit (check: `vastai show user --raw | jq .balance`)
   - [ ] API key is set: `export VAST_API_KEY=...`

2. **GitHub Secrets** (for auto-recovery)
   - [ ] `WANDB_API_KEY` - WandB logging
   - [ ] `VAST_API_KEY` - Auto-provisioning
   - [ ] `B2_APPLICATION_KEY_ID` - Checkpoint storage
   - [ ] `B2_APPLICATION_KEY` - Checkpoint storage

3. **B2 Bucket**
   - [ ] Bucket exists: `beautifulsoup-rl`
   - [ ] B2 CLI authorized: `b2 authorize-account`

4. **Config**
   - [ ] Upload config.toml to B2: `b2 file upload beautifulsoup-rl config.toml $RUN_ID/config.toml`

### GPU Pricing (2026-01-05)

| GPU | Count | Price/hr | VRAM | Availability |
|-----|-------|----------|------|--------------|
| RTX 4090 | 2 | $0.45-0.72 | 48GB | Abundant |
| H100 PCIE | 2 | $5.73 | 160GB | Limited |
| H100 SXM5 | 2 | ~$4-6 | 160GB | Rare |

**Recommendation:** Use 2x RTX 4090 for 7-8B models - 10x cheaper than H100 and sufficient VRAM.

### Checkpoint Flow

1. **Training writes**: `checkpoints/step_N/` (includes optimizer.pt when complete)
2. **Sync script detects**: Only syncs directories with `optimizer.pt` (atomic)
3. **Upload to B2**: `b2://beautifulsoup-rl/{RUN_ID}/step_N/`
4. **Update pointer**: `latest.json` with step + WANDB_RUN_ID
5. **On resume**: Download `latest.json`, then only the latest checkpoint

### WandB Run Continuity

The same WandB run ID is preserved across restarts:
- `latest.json` includes `wandb_run_id` field
- `onstart.sh` exports `WANDB_RUN_ID` before resuming
- Metrics continue in the same WandB run (no fragmentation)

### TODO: Multi-Provider Fallback

If Vast.ai has no availability, fall back to RunPod:
- [ ] Add RunPod API integration to `provision_vast.py`
- [ ] Modify `training_controller.py` to try providers in order
- [ ] Handle RunPod's 5s termination warning (SIGTERM → checkpoint → die)
