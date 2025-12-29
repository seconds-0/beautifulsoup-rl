# External Benchmarks Research

Research on existing RL environments and benchmarks to inform difficulty improvements.

---

## WebArena / BrowserGym

**What it is**: Realistic web environment for autonomous agents across 4 domains (e-commerce, social, dev, CMS).

**Key difficulty factors**:
- 812 long-horizon tasks
- Multi-step navigation required
- 3.3 template variations per task (prevents memorization)
- SOTA models achieve 31-62% success

**Applicable lessons**:
- Multi-page workflows create genuine difficulty
- Template instantiation prevents overfitting
- Domain-specific realistic scenarios

**Links**:
- https://webarena.dev/
- https://github.com/ServiceNow/BrowserGym

---

## SWE-bench

**What it is**: Benchmark for evaluating LLMs on real-world GitHub issues.

**Difficulty classification**:
| Level | Time Estimate | Count in Verified |
|-------|---------------|-------------------|
| Easy | <15 min | 196 |
| Medium | 15 min - 1 hr | majority |
| Hard | 1-4 hrs | ~45 |
| Very Hard | >4 hrs | 3 (barely solved) |

**Key insight**: Lines changed increases 11x from easy→hard. Files modified increases 2x. **Scope = difficulty**.

**Links**:
- https://github.com/SWE-bench/SWE-bench
- https://openai.com/index/introducing-swe-bench-verified/
- https://jatinganhotra.dev/blog/swe-agents/2025/04/15/swe-bench-verified-easy-medium-hard.html

---

## Prime Intellect Environments Hub

**What it is**: Community platform for sharing RL environments, 500+ environments across domains.

**Best practices**:
- Environments = dataset + harness + scoring rules
- Multi-turn via ToolEnv, StatefulToolEnv patterns
- Ground truth from structured data, not parsing
- Verifiers library for standardization

**Bounty tiers**:
- Open Access: $100-500 (first-time builders)
- Application-Only: $1000-5000+ (experienced developers)

**Links**:
- https://www.primeintellect.ai/blog/environments
- https://github.com/PrimeIntellect-ai/verifiers
- https://docs.primeintellect.ai/tutorials-environments/environments

---

## Scaling RL Research

**Zero-Variance Filtering** (from "From Art to Science: Scaling RL for LLMs"):
- Remove prompts where pass rate ≥90% (no gradient signal)
- Remove prompts where pass rate = 0% (too hard)
- Sweet spot: **10-70%** depending on model size

**On-Policy Distillation**:
- 50-100x more compute efficient than RL from scratch
- Starting from distilled checkpoint dramatically helps

**Links**:
- https://deep-paper.org/en/paper/2510.13786/

---

## Reward Hacking Prevention (2025 Research)

**Observed behaviors in frontier models**:
- Modify tests/scoring code
- Access existing implementations
- Exploit environment loopholes

**Prevention techniques**:
1. **Hybrid rewards**: Ground-truth verifiers + LLM-as-judge
2. **Adversarial training**: Generate exploits, add to training data
3. **Behavioral constraints**: Penalize spurious solutions
4. **Uncertainty quantification**: Penalize high-variance reward regions
5. **Monitorability tax**: Maintain interpretability

**Anthropic's approach (Claude 4)**:
- Improved environments
- Clarified reward signals
- Proactive monitoring
- Penalize hacking during training

**Links**:
- https://metr.org/blog/2025-06-05-recent-reward-hacking/
- https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
- https://assets.anthropic.com/m/74342f2c96095771/original/Natural-emergent-misalignment-from-reward-hacking-paper.pdf

---

## HuggingFace Environments Analysis

**Key insights for small model training**:
- Choose tasks where model shows non-zero performance
- Sweet spot: Task requires reasoning but isn't impossible
- Qwen3-0.6B improved from 0.403 → 0.578 reward on alphabet-sort

**Training configuration (0.6B model)**:
- 2x A6000 GPUs (48GB each)
- batch_size=8, gradient_accumulation=8
- ~8 hours training time

**Rule of thumb**:
> "If a model's reward is consistently near zero, the task is probably too hard, and you should try a different (or larger) model."

**Link**:
- https://huggingface.co/blog/anakin87/environments-hub
