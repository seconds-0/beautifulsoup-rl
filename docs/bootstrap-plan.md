# Plan: Bootstrapping Mechanism for 0% Models

## Problem Statement

Models like Qwen3-4B and Llama 3.2-3B score **0% pass rate** on our benchmark despite having function calling support. Pure RL training fails because there's no gradient signal—every attempt fails, so the model can't learn what works.

## Research Summary (Prime/Verifiers Docs)

### Critical Finding: Prime-RL Has Online Difficulty Buffer

From verifiers training.md:
> "If your model gets 0% reward after 10+ attempts, the task is too hard."

From prime-rl orchestrator config:
```toml
[orchestrator.buffer]
type = "online-difficulty"
oversampling_factor = 2.0
```

This is a **TRAINER-LEVEL** feature that dynamically filters tasks.

### Environment Pattern (from verifiers docs)

Environments should:
1. **Accept `difficulty` parameter** for pre-filtering
2. **Expose `difficulty` in info dict** for trainer's online buffer

```python
# Recommended pattern from verifiers docs
def load_environment(
    difficulty="all",  # Environment accepts difficulty arg
    **kwargs
):
    if difficulty != "all":
        dataset = dataset.filter(lambda x: x["difficulty"] == difficulty)
    ...

# Info dict should include:
"info": {
    "difficulty": "medium",  # For trainer's online-difficulty buffer
    ...
}
```

### Our Environment ALREADY Supports This!

**Current status** (verified in codebase):
- `TaskInstance.difficulty: str = "medium"` (base.py:68)
- `to_info_dict()` includes `"difficulty": self.difficulty` (base.py:113)
- `EnvConfig.difficulty` parameter for filtering (config.py:41)
- `mode="tiered"` with difficulty-weighted sampling (dataset.py:152)

**We already follow Prime's recommended pattern!**

### What's Actually Needed

The question isn't "how do we add bootstrapping" but:
1. **Does our existing `difficulty="easy"` work for 0% models?**
2. **Do we need to document this workflow better?**
3. **Should we add `mode="bootstrap"` as convenience alias?**

## Root Cause: Easy Archetypes Still Use Realistic Complexity

From `mvp_core_extraction.py:97-98`:
```python
# Always use realistic complexity for real-world difficulty
complexity = "realistic"  # FORCES noise even for "easy" tasks
```

Even `difficulty="easy"` archetypes have full realistic noise, framework patterns, and chrome. This is why 0% models still fail.

## Codex Review Feedback

**Key insight**: Reducing HTML complexity alone won't help if models don't know HOW to use the tool (write Python + BS4). Need explicit "tool-use primer" tasks.

**Gaps identified:**
- Missing Step 0: Ultra-banal tasks that teach the action template
- Missing "moderate noise" stage between low and realistic
- No partial credit / reward shaping

## Revised Training Flow for 0% Models

```
Step 0: Tool-use primer (NEW)
   - Ultra-banal HTML: <span id="target">Hello</span>
   - Fixed patterns, trivial selectors
   - Teaches: load HTML → parse → select → return
   - Expected: 0% → 10%+

Step 1: Bootstrap mode (low complexity)
   - Use: bootstrap=True + difficulty="easy"
   - Clean HTML, no chrome/noise
   - Expected: 10% → 30%+

Step 2: Moderate noise (NEW)
   - Use: complexity="moderate" (between low and realistic)
   - Sparse boilerplate, minimal nesting
   - Expected: 30% → 50%+

Step 3: Full curriculum
   - Use: mode="tiered" with realistic complexity
   - All difficulties with weighted sampling
   - Expected: 50% → 65%+
```

**Alternative**: Mix trivial tasks with decaying weight rather than hard stage switches.

## Implementation Plan

### 1. Add primer archetypes (Step 0)

Create `bs4_env/generators/primer.py` with ultra-banal tasks:

```python
@register(
    archetype_id="primer.extract_by_id",
    category="primer",
    difficulty="primer",  # New difficulty level
    description="Extract text from element with fixed id='target'",
)
class PrimerExtractByIdGenerator(Generator):
    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        text = rng.choice(["Hello", "World", "Test", "Example"])
        html = f'<span id="target">{text}</span>'
        return TaskInstance(
            html=html,
            query="Extract the text from the element with id='target'",
            ground_truth=text,
            ...
        )
```

### 2. Add complexity levels to config
```python
# config.py
complexity: Literal["primer", "low", "moderate", "realistic"] = "realistic"
```

### 3. Wire complexity through generators
```python
# In generators, respect complexity level:
if config.complexity == "primer":
    # Ultra-simple, single element
elif config.complexity == "low":
    # Simple structure, no chrome
elif config.complexity == "moderate":
    # Sparse boilerplate, minimal nesting
else:  # realistic
    # Full noise and chrome
```

### 4. Partial credit shaping

**Research findings:**

Verifiers natively supports continuous rewards (0.0-1.0). The [verifiers documentation](https://github.com/PrimeIntellect-ai/verifiers) shows:

```python
# Built-in partial credit example from verifiers docs
def partial_credit(prompt, completion, answer, state):
    """Give partial credit for containing key terms."""
    key_terms = answer.lower().split()
    response = completion[-1]['content']
    found = sum(1 for term in key_terms if term in response.lower())
    return found / len(key_terms) if key_terms else 0.0

# Rubrics combine multiple reward functions with weights
rubric = vf.Rubric(
    funcs=[exact_match, partial_credit, length_penalty],
    weights=[1.0, 0.5, 0.1]  # Relative importance
)
```

**Our current rubric already has partial credit** (`bs4_env/grading/rubric.py`):
- `REWARD_PARTIAL_MAX = 0.1` (line 30) - Very conservative
- `_partial_credit()` function (line 285) - For objects/arrays only
- `_f1_multiset()` (line 253) - F1 score for array partial credit
- Efficiency multiplier (line 52) - Rewards fewer tool calls
- BS4 usage penalty (line 49) - Gradient toward BS4 usage

**Proposed expansion for 0% models:**

```python
# New partial credit tiers for bootstrapping
REWARD_PARTIAL_TIERS = {
    "bs4_imported": 0.05,      # Credit for importing BS4
    "soup_created": 0.10,      # Credit for creating BeautifulSoup object
    "element_found": 0.20,     # Credit for finding any element
    "correct_type": 0.05,      # Credit for returning correct JSON type
    "partial_content": 0.10,   # Credit for partial content match (F1)
}

# Example implementation
def compute_structural_partial_credit(code_samples: list[str], output: dict) -> float:
    """Award partial credit for structural correctness in code.

    This helps 0% models learn the basic action template even when
    they get the wrong answer.
    """
    credit = 0.0

    # Check BS4 import (learned to use the library)
    if any("from bs4 import" in code or "import bs4" in code for code in code_samples):
        credit += REWARD_PARTIAL_TIERS["bs4_imported"]

    # Check soup creation (learned the API pattern)
    if any("BeautifulSoup(" in code for code in code_samples):
        credit += REWARD_PARTIAL_TIERS["soup_created"]

    # Check element selection (learned to query)
    selection_methods = [".find(", ".find_all(", ".select(", ".select_one("]
    if any(m in code for code in code_samples for m in selection_methods):
        credit += REWARD_PARTIAL_TIERS["element_found"]

    return min(credit, 0.4)  # Cap total structural credit
```

**Academic support:**
- [Dense Reward for Free in RLHF](https://arxiv.org/html/2402.00782v1) - Distributes sparse rewards using attention
- [Process Reward Models for LLM Agents](https://arxiv.org/html/2502.10325v1) - Turn-wise scoring
- [Reward Shaping to Mitigate Reward Hacking](https://arxiv.org/html/2502.18770v1) - Key principles: bounded, rapid initial growth

## Detailed Partial Credit Plan

### Philosophy

The goal is to provide gradient signal for 0% models without enabling reward hacking. We want to reward **correct process** (using BS4 properly) not just **correct output**. This follows the Process Reward Model paradigm from [arxiv:2502.10325](https://arxiv.org/html/2502.10325v1).

### Tier Structure

```
Tier 0: Format compliance (existing)
  └── Valid JSON output structure         → included in format_ok check

Tier 1: Tool invocation (new)
  └── Called run_python at least once     → 0.05 (already enforced as gate)

Tier 2: Library usage (new)
  ├── Imported BS4                        → +0.05
  └── Created BeautifulSoup object        → +0.10
  Subtotal: 0.15

Tier 3: DOM interaction (new)
  ├── Called selection method             → +0.15
  │   (.find, .find_all, .select, etc.)
  └── Accessed element content            → +0.10
      (.text, .get_text(), .string, etc.)
  Subtotal: 0.25

Tier 4: Answer quality (existing, enhanced)
  ├── Correct type (string/int/array)     → +0.05
  ├── Partial content match (F1 > 0.5)    → +0.10
  └── Exact match                         → remaining to 1.0
  Subtotal: 0.15 partial or 0.60 exact

Total possible: 1.0 for correct answer
Maximum partial: 0.55 for correct process but wrong answer
```

### Implementation

```python
# bs4_env/grading/rubric.py

# Partial credit configuration (can be disabled for benchmarking)
PARTIAL_CREDIT_ENABLED = True  # Set False for strict benchmarks

# Tier rewards
TIER_REWARDS = {
    "bs4_imported": 0.05,
    "soup_created": 0.10,
    "selection_method": 0.15,
    "content_access": 0.10,
    "correct_type": 0.05,
    "partial_match": 0.10,  # When F1 > 0.5
}
PARTIAL_CREDIT_CAP = 0.55  # Max without correct answer


def compute_process_reward(code_samples: list[str], output: dict) -> tuple[float, dict]:
    """Compute tiered partial credit based on code structure.

    Uses AST analysis where possible to avoid string-matching bypasses.
    Returns (reward, breakdown_dict) for debugging.
    """
    if not PARTIAL_CREDIT_ENABLED or not code_samples:
        return 0.0, {}

    breakdown = {}
    reward = 0.0

    # Tier 2: Library usage (use existing AST-based detection)
    bs4_imported = any(_check_bs4_import_ast(c) for c in code_samples)
    soup_created = any(_check_soup_creation_ast(c) for c in code_samples)

    if bs4_imported:
        reward += TIER_REWARDS["bs4_imported"]
        breakdown["bs4_imported"] = True
    if soup_created:
        reward += TIER_REWARDS["soup_created"]
        breakdown["soup_created"] = True

    # Tier 3: DOM interaction
    selection_used = any(_check_selection_method_ast(c) for c in code_samples)
    content_accessed = any(_check_content_access_ast(c) for c in code_samples)

    if selection_used:
        reward += TIER_REWARDS["selection_method"]
        breakdown["selection_method"] = True
    if content_accessed:
        reward += TIER_REWARDS["content_access"]
        breakdown["content_access"] = True

    return min(reward, PARTIAL_CREDIT_CAP), breakdown


def _check_bs4_import_ast(code: str) -> bool:
    """AST check for BS4 import."""
    # Reuse existing _check_bs4_usage_ast logic for imports only
    ...

def _check_soup_creation_ast(code: str) -> bool:
    """AST check for BeautifulSoup() constructor call."""
    ...

def _check_selection_method_ast(code: str) -> bool:
    """AST check for .find(), .find_all(), .select(), .select_one()."""
    ...

def _check_content_access_ast(code: str) -> bool:
    """AST check for .text, .get_text(), .string, ['href'], etc."""
    ...
```

### Integration with compute_reward()

```python
def compute_reward(...) -> tuple[float, dict[str, Any]]:
    # ... existing validation ...

    # Step 4: Compute correctness
    if status == "ok":
        base_reward, metrics = _grade_ok_response(output, task_info, metrics)
    # ... etc ...

    # Step 5: If wrong answer but valid process, add partial credit
    if base_reward == REWARD_WRONG and code_samples:
        process_reward, process_breakdown = compute_process_reward(code_samples, output)
        if process_reward > 0:
            base_reward = process_reward
            metrics["process_partial_credit"] = process_breakdown
            metrics["partial_credit_source"] = "process"

    # ... rest of function ...
```

### Anti-Hacking Measures

1. **AST-based detection**: String matching is easily bypassed. AST ensures actual code execution.

2. **Capped at 0.55**: Can never exceed correct answer reward. Models must eventually learn correctness.

3. **Configurable flag**: `PARTIAL_CREDIT_ENABLED = False` for benchmarks and competitions.

4. **Requires run_python**: Existing gate still applies - must call run_python to get any reward.

5. **Diminishing returns**: Later tiers require earlier tiers to be meaningful (can't claim credit for soup_created without bs4_imported).

### Testing Strategy

```python
def test_partial_credit_tiers():
    """Verify each tier awards correct credit."""

    # Tier 2: Import only
    code = "from bs4 import BeautifulSoup"
    reward, breakdown = compute_process_reward([code], {})
    assert reward == 0.05
    assert breakdown["bs4_imported"] is True

    # Tier 2: Import + soup
    code = """
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, 'html.parser')
"""
    reward, breakdown = compute_process_reward([code], {})
    assert reward == 0.15

    # Tier 3: Full process
    code = """
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, 'html.parser')
elem = soup.find('div', id='target')
print(elem.text)
"""
    reward, breakdown = compute_process_reward([code], {})
    assert reward == 0.40  # 0.05 + 0.10 + 0.15 + 0.10


def test_partial_credit_anti_hacking():
    """Verify AST detection prevents comment-based bypasses."""

    # Comment mentions BS4 but doesn't use it
    code = """
# from bs4 import BeautifulSoup
# soup = BeautifulSoup(HTML, 'html.parser')
import re
result = re.search(r'target="([^"]+)"', HTML)
print(result.group(1))
"""
    reward, breakdown = compute_process_reward([code], {})
    assert reward == 0.0  # No credit for comments
```

### Codex Review Findings

**HIGH SEVERITY:**

1. **Anti-hacking bypass** (`docs/bootstrap-plan.md:348`, `bs4_env/grading/rubric.py:662`)
   - Process reward gate only checks `base_reward == REWARD_WRONG`
   - Model could claim `status="limit"` on solvable tasks + run BS4 once = partial credit
   - **Fix**: Add explicit check: `if status == "limit" and solvable: return 0` before process credit

2. **AST checks don't ensure execution** (`docs/bootstrap-plan.md:282`, `:361`)
   - Code can put BS4 usage in dead code branches or parse dummy strings
   - Plan incorrectly claims "AST ensures actual code execution"
   - **Fix**: Require BS4 calls to use the `HTML` variable: check for `BeautifulSoup(HTML` pattern

**MEDIUM SEVERITY:**

3. **Tier dependencies not enforced** (`docs/bootstrap-plan.md:305`, `:369`)
   - `selection_method` credited independently of `soup_created`
   - Content access checks (`.text`, `.string`) too broad, hit non-BS4 objects
   - **Fix**: Chain tiers: only award tier N if tier N-1 was achieved

4. **Tier 1/4 not wired** (`docs/bootstrap-plan.md:237`, `rubric.py:24`)
   - `correct_type`/`partial_match` never applied in `compute_process_reward`
   - run_python → 0.05 not implemented
   - **Fix**: Wire all tiers or remove from spec

5. **run_python gate conditional** (`rubric.py:320`, `:400`)
   - `run_python_calls` can be `None`, allowing process credit without tool use
   - **Fix**: Require `run_python_calls is not None and run_python_calls > 0`

6. **Efficiency/penalty interaction unclear** (`rubric.py:424`, `:448`)
   - Process rewards could be scaled down by efficiency multiplier
   - BS4 penalty could double-count
   - **Fix**: Explicitly exclude process rewards from efficiency/BS4 adjustments

**0.55 CAP TOO HIGH:**
- Exceeds `REWARD_CORRECT_LIMIT` (0.5) - wrong answers could outscore limit responses
- Dwarfs `REWARD_PARTIAL_MAX` (0.1)
- **Recommendation**: Lower to **0.25-0.35**, keep below limit reward (0.5)

### Revised Implementation

```python
# Lower cap per Codex recommendation
PARTIAL_CREDIT_CAP = 0.30  # Below REWARD_CORRECT_LIMIT (0.5)

def compute_process_reward(code_samples: list[str], output: dict, status: str, solvable: bool) -> tuple[float, dict]:
    """Compute tiered partial credit with Codex-recommended safeguards."""

    # Gate 1: Don't reward limit-claiming on solvable tasks (anti-hacking)
    if status == "limit" and solvable:
        return 0.0, {"blocked": "limit_on_solvable"}

    if not PARTIAL_CREDIT_ENABLED or not code_samples:
        return 0.0, {}

    breakdown = {}
    reward = 0.0

    # Tier 2: Library usage with HTML variable check
    bs4_imported = any(_check_bs4_import_ast(c) for c in code_samples)
    soup_created_with_html = any(_check_soup_creation_with_html_ast(c) for c in code_samples)

    if bs4_imported:
        reward += TIER_REWARDS["bs4_imported"]
        breakdown["bs4_imported"] = True

    # Tier dependency: soup requires import
    if soup_created_with_html and bs4_imported:
        reward += TIER_REWARDS["soup_created"]
        breakdown["soup_created"] = True

        # Tier 3: DOM interaction (only if soup was created)
        selection_used = any(_check_selection_method_ast(c) for c in code_samples)
        if selection_used:
            reward += TIER_REWARDS["selection_method"]
            breakdown["selection_method"] = True

            # Content access requires selection
            content_accessed = any(_check_content_access_on_result_ast(c) for c in code_samples)
            if content_accessed:
                reward += TIER_REWARDS["content_access"]
                breakdown["content_access"] = True

    return min(reward, PARTIAL_CREDIT_CAP), breakdown


def _check_soup_creation_with_html_ast(code: str) -> bool:
    """AST check for BeautifulSoup(HTML, ...) - must use the HTML variable."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    class HTMLSoupVisitor(ast.NodeVisitor):
        def __init__(self):
            self.found = False

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "BeautifulSoup":
                # Check first argument is the HTML variable
                if node.args and isinstance(node.args[0], ast.Name):
                    if node.args[0].id == "HTML":
                        self.found = True
            self.generic_visit(node)

    visitor = HTMLSoupVisitor()
    visitor.visit(tree)
    return visitor.found
```

### Migration Plan (Revised)

1. **Phase 1**: Implement with Codex-recommended safeguards
   - Lower cap to 0.30
   - Enforce tier dependencies
   - Require `BeautifulSoup(HTML, ...)` pattern
2. **Phase 2**: Run Qwen3-4B and Llama 3.2-3B with partial credit
3. **Phase 3**: If 0% → >0%, success! If still 0%, add primer archetypes
4. **Benchmark runs**: Use `PARTIAL_CREDIT_ENABLED = False`

## Complexity Recommendation

**Strong recommendation: Both primer archetypes + complexity flag**

### Rationale

1. **Primer archetypes are essential** for teaching the action template:
   - 0% models don't know to write `from bs4 import BeautifulSoup; soup = BeautifulSoup(HTML, 'html.parser')`
   - Even with partial credit, they need examples that are trivially solvable
   - Ultra-banal HTML like `<span id="target">Hello</span>` removes all ambiguity

2. **Complexity flag enables smooth progression**:
   - Once models learn the template from primers, they need gradual complexity increase
   - Hard switching from primer → realistic is too jarring
   - `complexity="moderate"` bridges the gap with sparse boilerplate, minimal nesting

3. **The two work together**:
   ```
   Phase 0: difficulty="primer" (new archetypes, ultra-simple HTML)
            Models learn: import → parse → select → extract → return

   Phase 1: difficulty="easy" + complexity="low"
            Models learn: handle simple real-world structure

   Phase 2: difficulty="easy" + complexity="moderate"
            Models learn: filter noise, find target in context

   Phase 3: mode="tiered" (default realistic complexity)
            Full curriculum with difficulty weighting
   ```

4. **Alternative (just primers) is insufficient**:
   - Primer → realistic is too big a jump
   - Models would learn the template but fail on real HTML
   - Need intermediate complexity levels

5. **Alternative (just complexity flag) is insufficient**:
   - Even `complexity="low"` has structured HTML with real elements
   - 0% models still wouldn't know to write BS4 code
   - Need the ultra-trivial primer examples first

### Implementation Summary

| Component | Purpose | Example |
|-----------|---------|---------|
| `primer.py` archetypes | Teach action template | `<span id="target">Hello</span>` |
| `complexity="primer"` | Shorthand for ultra-simple | Single element, no chrome |
| `complexity="low"` | Simple structure | List of 3-5 items, no noise |
| `complexity="moderate"` | Real patterns, less noise | Nav + content, sparse chrome |
| `complexity="realistic"` | Default, full noise | Framework patterns, chrome |

## File Summary

| File | Changes |
|------|---------|
| `bs4_env/config.py` | Add `bootstrap: bool = False` and `complexity` options |
| `bs4_env/dataset.py` | Pass bootstrap flag to generators |
| `bs4_env/generators/base.py` | Modify `wrap_with_realistic_chrome` for bootstrap mode |
| `bs4_env/generators/primer.py` | NEW: Ultra-banal primer tasks |
| Multiple generators | Respect complexity flag |
| `bs4_env/grading/rubric.py` | Partial credit shaping (research needed) |
| `README.md` | Document bootstrap workflow |

## Open Questions

1. **Partial credit scope**: Verifiers supports continuous rewards 0.0-1.0. Should we implement structural partial credit (BS4 import → soup created → element found) or keep conservative binary rewards to avoid reward hacking?
2. **Mixing strategies**: Should we hard-switch between stages or mix with decaying weights?
3. **Primer task count**: How many ultra-banal primer archetypes do we need? (Proposed: 3-5 covering id, class, tag selectors)
4. **Complexity override**: Should `bootstrap=True` globally override complexity, or should it be per-archetype?

## Research Summary

| Topic | Finding | Source |
|-------|---------|--------|
| Continuous rewards | Verifiers natively supports 0.0-1.0 rewards | [verifiers docs](https://github.com/PrimeIntellect-ai/verifiers) |
| Partial credit | Built-in `partial_credit` function, weighted rubrics | verifiers environments.md |
| Online difficulty | Prime-RL has trainer-level task filtering | prime-rl orchestrator.toml |
| Dense rewards | ABC method distributes sparse reward via attention | [arxiv:2402.00782](https://arxiv.org/html/2402.00782v1) |
| Reward shaping | Key principles: bounded, rapid growth, centered | [arxiv:2502.18770](https://arxiv.org/html/2502.18770v1) |
| Process rewards | Turn-wise scoring acts like Q-function | [arxiv:2502.10325](https://arxiv.org/html/2502.10325v1) |
| Our rubric | Already has partial credit (max 0.1), efficiency multiplier, BS4 penalty | rubric.py:30-49 |

## Sources

### Prime/Verifiers Documentation (Primary)
- [Verifiers training.md](https://github.com/primeintellect-ai/verifiers/blob/main/docs/source/training.md)
- [Verifiers environments.md](https://github.com/primeintellect-ai/verifiers/blob/main/docs/source/environments.md)
- [Prime-RL GitHub](https://github.com/PrimeIntellect-ai/prime-rl)
- [Verifiers v0.1.7 Release](https://github.com/primeintellect-ai/verifiers/blob/main/docs/release/RELEASE_v0.1.7.md)

### Prime Training Blogs
- [INTELLECT-2 Training](https://www.primeintellect.ai/blog/intellect-2)
- [INTELLECT-3 Training](https://www.primeintellect.ai/blog/intellect-3)

### Academic Papers
- [h1: Bootstrapping LLMs for Long-Horizon Reasoning](https://arxiv.org/abs/2510.07312)
- [AdaRFT: Adaptive Curriculum RL Finetuning](https://arxiv.org/abs/2504.05520)
