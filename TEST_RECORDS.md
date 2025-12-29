# Test Records

Track evaluation progress and results. Update this file after each benchmark run.

## Target Models

Focus on **small/weak models** - they benefit most from RL training on this environment.

### Priority (Small Models)
- [x] `qwen/qwen3-8b` - Free tier available (**52.5% pass rate** - see below)
- [ ] `deepseek/deepseek-r1-0528-qwen3-8b` - Cheap
- [ ] `meta-llama/llama-3.1-8b-instruct` - Standard baseline
- [ ] `mistralai/mistral-7b-instruct` - Another baseline

### Blocked
- ~~`prime-intellect/intellect-3`~~ - OpenRouter function calling broken (Qwen parsing bug)
- ~~`x-ai/grok-4.1-fast`~~ - Rate limits too aggressive

### For Comparison Only (Strong Models)
- `anthropic/claude-3-5-haiku-latest` - Fast, capable
- `openai/gpt-4o` - Strong baseline

---

## Benchmark Runs

### 2025-12-28: Ministral 8B (Full - 200/200)

**Model:** `mistralai/ministral-8b`
**Config:** split=bench, mode=mvp, 200 examples
**Status:** Complete

| Archetype | Avg Reward | Perfect | Pass Rate |
|-----------|------------|---------|-----------|
| `mvp.string_returns_none` | 0.465 | 6/20 | 47% |
| `mvp.table_list_of_lists` | 0.425 | 6/20 | 43% |
| `mvp.class_reserved_word` | 0.420 | 5/20 | 42% |
| `mvp.extract_text_by_id` | 0.380 | 2/20 | 38% |
| `mvp.none_attribute_error` | 0.320 | 3/20 | 32% |
| `mvp.multivalue_class` | 0.250 | 1/20 | 25% |
| `mvp.extract_text_by_class` | 0.160 | 0/20 | 16% |
| `mvp.table_list_of_dicts` | 0.045 | 0/20 | 5% |
| `mvp.limit_js_required` | 0.000 | 0/20 | 0% |
| `mvp.limit_image_text` | 0.000 | 0/20 | 0% |
| **Total** | **0.246** | **23/200** | **27.5%** |

**Observations:**
- MUCH worse than Qwen3-8B (27.5% vs 52.5%)
- Heavy looping: 1,518 tool calls vs Qwen3's 246 (6x more)
- Burned 16.4M tokens (32x more than Qwen3's 518K)
- Tables were HARD (5-43%) unlike Qwen3 (100%)
- string_returns_none was BETTER (47% vs 20%) - opposite pattern

**Token Usage:** 16.4M input, 190K output (~16.6M total) - VERY expensive

---

### 2024-12-28: Qwen3-8B (Full - 200/200)

**Model:** `qwen/qwen3-8b`
**Config:** split=bench, mode=mvp, 200 examples
**Status:** Complete

| Archetype | Avg Reward | Perfect | Pass Rate |
|-----------|------------|---------|-----------|
| `mvp.table_list_of_dicts` | 1.000 | 20/20 | 100% |
| `mvp.table_list_of_lists` | 1.000 | 20/20 | 100% |
| `mvp.none_attribute_error` | 0.840 | 16/20 | 80% |
| `mvp.multivalue_class` | 0.650 | 13/20 | 65% |
| `mvp.extract_text_by_class` | 0.590 | 10/20 | 50% |
| `mvp.extract_text_by_id` | 0.435 | 7/20 | 35% |
| `mvp.class_reserved_word` | 0.430 | 7/20 | 35% |
| `mvp.string_returns_none` | 0.245 | 4/20 | 20% |
| `mvp.limit_js_required` | 0.070 | 0/20 | 0% |
| `mvp.limit_image_text` | 0.000 | 0/20 | 0% |
| **Total** | **0.526** | **97/200** | **52.5%** |

**Observations:**
- Tables are trivially easy (100%) - model excels at structured extraction
- Limitation tasks are VERY hard (0-7%) - model doesn't recognize when to abstain
- Text extraction moderate (35-50%) - confused by decoys
- Gotcha archetypes (`string_returns_none` 20%) show where RL would help most
- ~50% of failures had 0 tool calls (model did chain-of-thought but no action)

**Token Usage:** 518K input, 564K output (~1M total)

---

### 2024-12-28: Grok 4.1 Fast (Partial - 70/200)

**Model:** `x-ai/grok-4.1-fast`
**Config:** split=bench, mode=mvp, 200 examples requested
**Status:** Killed at 70 examples (rate limit hit)

| Archetype | Tested | Passed | Pass Rate |
|-----------|--------|--------|-----------|
| `mvp.extract_text_by_id` | 20 | 19 | 95% |
| `mvp.extract_text_by_class` | 20 | 16 | 80% |
| `mvp.table_list_of_dicts` | 20 | 20 | 100% |
| `mvp.table_list_of_lists` | 10 | 10 | 100% |
| **Total** | **70** | **65** | **93%** |

**Observations:**
- Class extraction is hardest (similar class names confusing model)
- Tables are easy (100%)
- All examples used tool correctly (1-2 calls)

**Issues:**
- Hit rate limit around example 71 (9521s timeout)

---

## Environment Changes Log

### 2024-12-28: Force Tool Usage
- **Change:** Removed HTML from prompt, stored in `info` dict instead
- **Why:** Models were "cheating" by reading HTML directly from prompt without using BS4
- **Files:** `prompt.py`, `base.py`, `verifiers_adapter.py`

### 2024-12-28: Schema Explanation Fix
- **Change:** Added `explain_schema()` to convert JSON schema to plain English
- **Why:** Models confused `{"type": "string"}` schema with answer format
- **Files:** `prompt.py`

### 2024-12-28: Removed response_format (conflict with tools)
- **Change:** Removed `response_format={"type": "json_object"}` from API calls
- **Why:** Conflicts with `tools` parameter on some providers (Fireworks/Qwen)
- **Files:** `eval_with_llm.py`

---

## Notes

- Environment has 24 archetypes total (13 phase 1, 8 phase 2, 3 i18n)
- Bench split uses seeds 110000-111000 (20 per archetype)
- Efficiency penalty: -10% per extra tool call, floor at 0.2, cutoff at 11+
