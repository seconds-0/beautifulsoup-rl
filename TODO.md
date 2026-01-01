# BeautifulSoup RL Environment - TODO

Master list of bugs, improvements, and next steps.

---

## Bugs & Problems

### CRITICAL: Navigate Tool Was Missing (FIXED 2025-12-30)

**Status:** Fixed in commit `925d7cc`

The `navigate` tool was never exposed to models in `eval_with_llm.py`. This meant:
- All multi-step archetypes were **impossible to solve**
- Models scored 0% on: `link_chain`, `search_then_detail`, `compare_products`, `pagination_aggregate`, `list_extraction`
- Previous benchmark results for these 5 archetypes (100 examples, ~15% of test) are **invalid**

**Fix:** Added `NAVIGATE_TOOL_SCHEMA` to `get_tools()` and handle navigate calls in execution loop.

**Impact:** Re-running all benchmarks with fix. Expect significant score improvements.

---

### compare_products Fails Due to Missing Schema Keys in Prompt (FIXED 2025-12-30)

**Status:** Fixed in `prompt.py`

**Problem:** Models solved compare_products correctly (right values, right logic) but used wrong key names:
- Ground truth: `{"cheaper": "Elite Device", "price_difference": "$102.46"}`
- Model output: `{"cheaper_product": "Elite Device", "price_difference": "$102.46"}`

The `explain_schema()` function for object types just said "must be an object (dictionary)" without showing the required key names.

**Fix:** Enhanced `explain_schema()` to show exact required keys for object schemas:
```
Before: "The `answer` field must be an **object** (dictionary)"
After:  "The `answer` field must be an **object** with these exact keys: {"cheaper": string (required), "price_difference": string (required)}"
```

**Impact:** Should significantly improve compare_products scores when re-benchmarked.

---

### Log Storage Was Insufficient (FIXED 2025-12-30)

**Status:** Fixed in commit `925d7cc`

**Problem:** Results files were missing critical debugging data:
- `tool_history` not saved at all
- `final_output` truncated to 500 chars (only in verbose mode)
- `ground_truth` only saved in verbose mode
- `result` in tool_history truncated to 1000 chars

**Fix:** Now always stores:
- Full `tool_history` (all code executed and results)
- Full `final_output`
- `ground_truth`
- `query`

**Impact:** Future benchmarks will have full logs for debugging and analysis.

---

## Improvements

### High Priority

- [ ] **Add --compact flag for eval_with_llm.py** - Full logs can be large (HTML in tool results). Add option to reduce storage for production runs.

- [ ] **Validate multi-step tasks work end-to-end** - Run a few examples manually to verify navigate tool integration is correct.

- [ ] **Add test for navigate tool** - Unit test that verifies navigate tool is exposed and works correctly.

### Medium Priority

- [ ] **Add solvability verification tests** - For each archetype, verify at least one parser can solve the generated task.

- [ ] **Improve efficiency penalty** - Current -10% per extra tool call may be too aggressive for multi-step tasks that legitimately need many calls.

- [ ] **Add cost tracking** - Track $ cost per model for budgeting.

- [ ] **Parallel benchmark runs** - Run multiple models simultaneously on different runners.

### Low Priority

- [ ] **Add Claude/Anthropic model support** - Currently only OpenRouter, add direct Anthropic API option.

- [ ] **Web UI for results** - Dashboard to visualize benchmark results over time.

- [ ] **Streaming progress** - Show real-time benchmark progress in GitHub Actions logs.

---

## Research Questions

### Why is GPT-5.2 only ~8% better than Ministral 3 8B?

**Hypothesis:** The gap may widen significantly now that multi-step tasks work. Previous results had both models at 0% on 5 archetypes, hiding the real difference.

**To verify:** Compare new benchmark results when complete.

### What makes multi-step tasks hard?

**Observations from tool_history analysis needed:**
- Are models failing to use navigate?
- Are they navigating but not aggregating data?
- Is the output format wrong?
- Are they getting lost in the HTML?

**Action:** Analyze full logs from new benchmark runs.

### Is the efficiency penalty calibrated correctly?

**Current:** -10% per extra tool call beyond optimal, floor at 0.2x

**Questions:**
- Multi-step tasks legitimately need 5-10+ tool calls. Are we penalizing correct behavior?
- Should penalty be archetype-specific?

---

## Completed

- [x] Fix navigate tool exposure (2025-12-30)
- [x] Add full log storage (2025-12-30)
- [x] Re-trigger all benchmarks with fix (2025-12-30)
- [x] Document parser_required bug (2025-12-30) - Removed archetype entirely (2026-01-01)

---

## Benchmark Status

### Latest Results (2025-12-30, with navigate fix)

| Model | Pass Rate | Perfect Rate | Avg Reward | vs Old |
|-------|-----------|--------------|------------|--------|
| **GPT-5.2** | **75.6%** | 57.5% | 0.726 | +4.7% |
| Ministral 3 8B | 63.2% | 56.2% | 0.621 | +0.0% |
| Qwen3-8B | 43.1% | 26.6% | 0.401 | +3.5% |

### Multi-Step Archetype Breakdown

| Archetype | GPT-5.2 | Ministral 3 | Qwen3 |
|-----------|---------|-------------|-------|
| search_then_detail | **100%** | 45% | 5% |
| link_chain | **40%** | 0% | 5% |
| pagination_aggregate | **20%** | 15% | 0% |
| compare_products | 0%* | 0%* | 0%* |
| list_extraction | 5% | 0% | 5% |

*compare_products 0% due to schema key bug (now fixed) - needs re-benchmark

### Previous Results (INVALID - no navigate tool)

| Model | Pass Rate | Note |
|-------|-----------|------|
| GPT-5.2 | 70.9% | Multi-step archetypes were impossible |
| Ministral 3 8B | 63.2% | Multi-step archetypes were impossible |
| Qwen3-8B | 39.6% | Multi-step archetypes were impossible |

---

## File Reference

| File | Purpose |
|------|---------|
| `bs4_env/scripts/eval_with_llm.py` | LLM evaluation script |
| `bs4_env/generators/mvp_hard.py` | Hard archetypes |
| `bs4_env/generators/mvp_multistep.py` | Multi-step archetypes |
| `bs4_env/tools/tool_defs.py` | Tool definitions including navigate |
| `TEST_RECORDS.md` | Benchmark results history |
