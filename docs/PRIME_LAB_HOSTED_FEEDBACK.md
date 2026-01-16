# Prime Intellect Lab Hosted Training - Feature Requests & Feedback

**From**: BeautifulSoup RL Environment Team (seconds-0)
**Date**: 2026-01-14
**Context**: Extensive debugging of training crashes over multiple days

---

## Executive Summary

We experienced recurring training crashes that took significant effort to diagnose. The root cause was **inference pod overload** - our environment processes HTML so fast that it overwhelms the inference service. The fix was simple (reduce concurrency), but **discovering** the root cause required:

**Note**: We discovered that `PrimeMonitor` infrastructure already exists in the codebase and Lab Hosted runs ARE configured to send metrics. However, the backend endpoints return 404:
```
WARNING Failed to upload to Prime Intellect API (metrics) after 3 attempts:
HTTPStatusError: Client error '404 Not Found' for url
'https://backend-worker-abruy2xuia-ew.a.run.app/api/internal/rft/metrics'
```
The `/samples` endpoint works, but `/metrics` and `/distributions` return 404. **The plumbing exists - just needs backend implementation.**

Specific debugging steps that took time:
- 10+ failed training runs
- Cloning the prime-rl repo to understand architecture
- External AI code reviews (Codex, Gemini)
- Building elimination test hypotheses from scratch

Most of this work could have been avoided with better observability and configuration options.

---

## Priority 0: Quick Win (Backend Already Exists)

### 0.1 Enable PrimeMonitor Backend Endpoints

**The client-side code already sends metrics to Prime's backend**, but the endpoints return 404:

```python
# From prime_rl/utils/monitor/prime.py - this code RUNS in Lab Hosted:
self._make_request("metrics", {"run_id": self.run_id, "metrics": metrics})
self._make_request("samples", {"run_id": self.run_id, "samples": samples})  # This one works!
self._make_request("distributions", {"run_id": self.run_id, "distributions": distributions})
```

**What's working**: `/samples` endpoint accepts data (we see successful uploads at step 10, 20, etc.)
**What's broken**: `/metrics` and `/distributions` return 404

**Quick fix**: Implement these endpoints in the Lab Hosted backend (`backend-worker-abruy2xuia-ew.a.run.app`). The data is already being sent - just need to receive and display it.

This would immediately give us:
- Per-step metrics visible in Prime dashboard
- Distribution histograms for reward/advantage analysis
- Real-time training progress without needing WandB

---

## Priority 1: Critical (Would Have Saved Days)

### 1.1 Inference Pod Observability

**Problem**: When training crashes, we see only DNS errors like "Name or service not known." We had no visibility into WHY the inference pod died.

**What we needed**:
- Inference pod CPU/GPU utilization metrics
- Inference pod memory usage
- Request queue depth / backpressure indicators
- Pod restart events with termination reason (OOMKilled, Evicted, etc.)
- vLLM internal metrics (KV cache utilization, batch sizes, latency percentiles)

**Impact**: We spent hours testing hypotheses about sandbox memory when the actual problem was inference overload. A simple "inference pod restarted due to OOM" message would have pointed us in the right direction immediately.

**Suggested Implementation**:
```
# In logs or dashboard:
18:41:26 WARNING Inference pod CPU: 95% | GPU Memory: 98% | Queue: 847 requests
18:41:28 ERROR Inference pod restarted (reason: OOMKilled, GPU memory exhausted)
```

### 1.2 Checkpoint Resume for Lab Hosted

**Problem**: Lab Hosted has NO checkpoint resume capability. When training crashes at step 12 of a 1000-step run, we lose all progress and must restart from scratch.

**What we needed**:
```bash
# This doesn't work for Lab Hosted:
prime rl run config.toml --ckpt.resume-step 12
```

**Impact**:
- We had a successful 50-step bootstrap run with good rewards
- Wanted to continue for 950 more steps with full task diversity
- Could not resume - had to restart from step 0
- Every crash means complete loss of training progress

**Suggested Implementation**:
- Automatic checkpointing to Prime's storage
- `prime rl resume <run-id>` command
- Optional: `prime rl run config.toml --resume-from <run-id>`

### 1.3 Inference Configuration Options

**Problem**: We cannot configure inference pod resources or parallelism. Self-hosted prime-rl supports:
```toml
[inference]
api_server_count = 4
gpu_memory_utilization = 0.95

[inference.parallel]
dp = 6  # Data parallelism - multiple vLLM replicas
```

But Lab Hosted ignores these settings entirely.

**What we needed**:
- Ability to request more inference capacity for demanding workloads
- Option to trade cost for stability (e.g., "use 2x inference replicas")
- Or at minimum: documentation of what inference resources are allocated per model size

**Impact**: Our only option was to throttle client-side concurrency, which reduces throughput. We went from ~3000 tok/s to ~1500 tok/s to maintain stability.

---

## Priority 2: High (Would Have Saved Hours)

### 2.1 Resource Limit Documentation

**Problem**: We have no idea what limits exist on sandbox or inference resources. We discovered through trial and error:
- `memory_gb=4` crashes with large HTML
- `memory_gb=16` crashes later
- `memory_gb=32` crashes EARLIER (counterintuitive - sandbox too fast!)

**What we needed**:
- Documentation of default/max values for `memory_gb`, `cpu_cores`, `disk_size_gb`
- Documentation of inference pod resources per model size
- Guidance on tuning concurrency settings for different workload profiles

**Suggested Documentation**:
```markdown
## Resource Allocation

| Model Size | Inference GPUs | Max Concurrent Requests | Recommended batch_size |
|------------|---------------|------------------------|----------------------|
| 1-4B       | 1x H100       | ~500                   | 32-64                |
| 7-8B       | 2x H100       | ~300                   | 16-32                |
| 30B+       | 4x H100       | ~150                   | 8-16                 |

## Sandbox Limits
- memory_gb: 1-64 (default: 4)
- cpu_cores: 1-8 (default: 2)
- Note: Higher memory_gb can increase throughput, which may overwhelm inference
```

### 2.2 Crash Diagnostics in Logs

**Problem**: Crash logs only show the symptom (DNS error), not the cause.

**Current output**:
```
ERROR Failed to reach inference server: Name or service not known
```

**What we needed**:
```
ERROR Inference pod unavailable (pod restarted 3 times in last 5 minutes)
DIAGNOSTIC: High request rate detected (847 pending). Consider reducing:
  - batch_size (current: 128, suggested: 32-64)
  - rollouts_per_example (current: 8, suggested: 4)
  - max_async_level (current: 2, suggested: 1)
```

### 2.3 Backpressure / Rate Limiting

**Problem**: When inference is overwhelmed, requests pile up until the pod dies. There's no graceful degradation.

**What we needed**:
- Automatic backpressure when inference queue is full
- Warning messages before crash ("inference queue at 90% capacity")
- Optional: automatic concurrency reduction when backpressure detected

**Suggested Behavior**:
```
18:41:20 WARNING Inference backpressure detected (queue: 500/600). Throttling new requests.
18:41:25 INFO Backpressure relieved (queue: 200/600). Resuming normal operation.
```

---

## Priority 3: Medium (Quality of Life)

### 3.1 WandB Integration for Inference Metrics

Log inference-side metrics to WandB alongside training metrics:
- `inference/queue_depth`
- `inference/gpu_memory_used`
- `inference/request_latency_p99`
- `inference/batch_size_actual`

This would let users correlate training issues with inference bottlenecks.

### 3.2 Cost Estimation Before Run

Show estimated cost before starting a run:
```bash
$ prime rl run config.toml --dry-run
Estimated cost: $12.50/hour
  - Model: Qwen/Qwen3-4B-Instruct-2507
  - Inference: ~$8/hour
  - Sandbox: ~$4.50/hour (memory_gb=32, cpu_cores=2)

Estimated total for 1000 steps: $45-60
Proceed? [y/N]
```

### 3.3 Preset Configurations

Offer validated presets for common scenarios:
```bash
prime rl run config.toml --preset stable    # Conservative, won't crash
prime rl run config.toml --preset balanced  # Good throughput, stable
prime rl run config.toml --preset aggressive # Maximum throughput, may crash
```

### 3.4 Better Error Messages for Config Validation

Current: Config silently ignores unsupported options
Better: Warn when config contains options that Lab Hosted doesn't support

```
WARNING: [inference.parallel] section ignored - Lab Hosted manages inference automatically
INFO: To configure inference, consider self-hosted deployment: https://docs.primeintellect.ai/self-hosted
```

---

## Priority 4: Nice to Have

### 4.1 Live Metrics Dashboard

Web dashboard showing real-time:
- Training progress (step, reward, throughput)
- Inference health (CPU, GPU, queue depth)
- Sandbox health (memory, active workers)

### 4.2 Alerts / Webhooks

Notify on:
- Training completion
- Training crash (with diagnostics)
- Approaching resource limits

### 4.3 A/B Testing Support

Run multiple configs in parallel and compare results:
```bash
prime rl compare config-a.toml config-b.toml --steps 50
```

---

## Summary: What Would Have Helped Most

| Issue | Time Wasted | Priority |
|-------|-------------|----------|
| No visibility into inference pod death reason | ~8 hours | P1 |
| No checkpoint resume | ~4 hours (lost progress) | P1 |
| No inference config options | ~2 hours research | P1 |
| Unknown resource limits | ~3 hours trial & error | P2 |
| Poor crash diagnostics | ~2 hours debugging | P2 |
| No backpressure handling | ~1 hour retrying | P2 |

**Total time that better tooling could have saved: ~20 hours**

---

## Our Current Workaround

We discovered through extensive testing that **reducing concurrency** prevents inference overload:

```toml
# Working config (stable)
batch_size = 32               # Reduced from 128
rollouts_per_example = 4      # Reduced from 8
max_async_level = 1           # Reduced from 2
oversampling_factor = 1.0     # Reduced from 2.0
memory_gb = 32                # Keep high for HTML parsing

# Result: Stable training at ~1500-2000 tok/s
# Tradeoff: ~50% reduced throughput vs aggressive settings
```

This works, but we had to discover it ourselves. Better docs or diagnostics would have pointed us here immediately.

---

## Contact

Happy to discuss any of these in detail or provide logs/configs from our debugging session.

- GitHub: seconds-0
- Project: beautifulsoup-rl (Prime Environments Hub bounty)
