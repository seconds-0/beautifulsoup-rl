# Bootstrap Feature Test Plan

This document outlines how to validate the bootstrap mechanism on tiny models before committing to RL training.

## Phase 1: Unit Test Validation (Complete)

All 340 unit tests pass, including:
- 42 tests for process partial credit
- 25 tests for primer archetypes
- Existing grading and archetype tests

## Phase 2: Tiny Model Smoke Tests

### Test Models (0% Baseline)

| Model | Size | Function Calling | Expected Behavior |
|-------|------|------------------|-------------------|
| Qwen3-0.6B | 0.6B | Yes | Struggles with tool use |
| Qwen3-1.7B | 1.7B | Yes | May attempt BS4 |
| Qwen3-4B | 4B | Yes | Should benefit from partial credit |
| Llama 3.2-1B | 1B | Yes | Struggles with tool use |
| Llama 3.2-3B | 3B | Yes | Should benefit from partial credit |

### Test 1: Primer Tasks Only

**Goal**: Verify models can solve ultra-simple HTML tasks.

```bash
# Run evaluation on primer-only tasks
uv run python -m bs4_env.scripts.eval_with_llm \
  --model qwen/qwen3-4b:free \
  --mode bootstrap \
  --difficulty primer \
  --num 50 \
  --verbose
```

**Expected Outcomes**:
- Models should achieve >0% on primer tasks
- If still 0%, the task is too hard OR model doesn't know the tool

**Success Criteria**:
- At least 1 correct answer out of 50
- Evidence of BeautifulSoup usage in generated code

### Test 2: Partial Credit Validation

**Goal**: Verify partial credit is awarded for correct tool-use patterns.

```bash
# Run evaluation with verbose output to check partial credit
uv run python -m bs4_env.scripts.eval_with_llm \
  --model qwen/qwen3-4b:free \
  --mode bootstrap \
  --num 50 \
  --verbose 2>&1 | grep "partial_credit"
```

**What to Check**:
1. Does the model generate code with `BeautifulSoup(HTML, ...)`?
2. Does it use selection methods (`.find()`, `.select()`)?
3. Does it access content (`.text`, `.get_text()`)?

**Expected Partial Credit Breakdown**:
- bs4_imported: +0.05 (should see in most attempts)
- soup_created_with_html: +0.10 (if model uses HTML variable)
- selection_method: +0.10 (if model calls .find/.select)
- content_access: +0.05 (if model uses .text/.get_text)

### Test 3: Compare Modes

**Goal**: Validate bootstrap mode provides easier tasks than standard modes.

```bash
# Compare pass rates across modes
for mode in primer bootstrap mvp; do
  echo "=== Mode: $mode ==="
  uv run python -m bs4_env.scripts.eval_with_llm \
    --model qwen/qwen3-4b:free \
    --mode $mode \
    --num 50 \
    --quiet
done
```

**Expected Ordering**:
- primer > bootstrap > mvp (pass rate)

### Test 4: Anti-Hacking Validation

**Goal**: Ensure partial credit can't be gamed.

**Manual Tests**:

1. **Limit-on-solvable blocked**:
   - Model claims `status: "limit"` on a primer task
   - Should receive 0.0 reward (not partial credit)

2. **String literal blocked**:
   - Model uses `BeautifulSoup("<html>dummy</html>", ...)`
   - Should NOT receive soup_created_with_html credit

3. **Comment bypass blocked**:
   - Model puts `# BeautifulSoup(HTML)` in comments
   - Should NOT receive any credit

**Test Script**:
```python
from bs4_env.grading.rubric import compute_process_partial_credit

# Test 1: Limit on solvable
reward, _ = compute_process_partial_credit(
    code_samples=['soup = BeautifulSoup(HTML, "html.parser")'],
    status="limit",
    solvable=True,
    run_python_calls=1,
)
assert reward == 0.0, "Should block limit on solvable"

# Test 2: String literal
reward, breakdown = compute_process_partial_credit(
    code_samples=['soup = BeautifulSoup("<html></html>", "html.parser")'],
    status="ok",
    solvable=True,
    run_python_calls=1,
)
assert "soup_created_with_html" not in breakdown, "Should not credit string literal"

# Test 3: Comment bypass
reward, breakdown = compute_process_partial_credit(
    code_samples=['# BeautifulSoup(HTML, "html.parser")\nx = 1'],
    status="ok",
    solvable=True,
    run_python_calls=1,
)
assert reward == 0.0, "Should not credit comments"

print("All anti-hacking tests passed!")
```

## Phase 3: Local Training Test (Optional)

If smoke tests pass, try a short training run:

```bash
# 100 steps on primer tasks only
prime env train seconds-0/beautiful-soup-env \
  -a '{"mode":"bootstrap","difficulty":"primer"}' \
  -m qwen/qwen3-4b \
  --steps 100 \
  --eval-interval 50
```

**Watch For**:
- Reward trend (should increase from 0)
- Loss convergence
- Code quality improvement

## Expected Results Summary

| Metric | Primer-Only | Bootstrap | MVp |
|--------|-------------|-----------|-----|
| 0% Model Pass Rate | 5-10% | 3-5% | 0% |
| Partial Credit Rate | 50%+ | 30-50% | 20-30% |
| Average Reward | 0.10-0.15 | 0.05-0.10 | 0.00-0.02 |

## Troubleshooting

### Model Still Scores 0%

1. Check if model generates ANY Python code
2. Check if model uses function calling correctly
3. Try even simpler prompts (direct instructions)
4. Consider SFT warmup before RL

### Partial Credit Not Awarded

1. Verify `code_samples` is passed to `compute_reward`
2. Check if model uses `HTML` variable (not string literal)
3. Inspect AST detection with verbose logging

### Model Games the System

1. Check if claims "limit" on solvable tasks
2. Check for dead code patterns (`if False: ...`)
3. Review breakdown dict for suspicious patterns

## Files Modified for Bootstrap

| File | Changes |
|------|---------|
| `bs4_env/config.py` | Added bootstrap mode, primer difficulty |
| `bs4_env/dataset.py` | Bootstrap mode handling |
| `bs4_env/generators/primer.py` | 5 new primer archetypes |
| `bs4_env/grading/rubric.py` | Process partial credit system |
| `bs4_env/registry.py` | Support primer difficulty |
| `bs4_env/auto_import.py` | Import primer module |
| `README.md` | Bootstrap documentation |
