# Test Records

Track evaluation progress and results. Update this file after each benchmark run.

## Target Models

Focus on **small/weak models** - they benefit most from RL training on this environment.

### Priority (Small Models)
- [x] `mistralai/ministral-8b-2512` - **BEST: 71% pass rate** (Ministral 3, Dec 2025)
- [x] `qwen/qwen3-8b` - Free tier available (**52.5% pass rate**)
- [x] `mistralai/ministral-8b` - Old version (**27.5% pass rate** - loopy, expensive)

### Blocked
- ~~`deepseek/deepseek-r1-0528-qwen3-8b`~~ - No function calling support on OpenRouter
- ~~`google/gemma-3-*`~~ - No function calling support on OpenRouter
- ~~`prime-intellect/intellect-3`~~ - OpenRouter function calling broken (Qwen parsing bug)
- ~~`x-ai/grok-4.1-fast`~~ - Rate limits too aggressive

### For Comparison Only (Strong Models)
- [x] `openai/gpt-5.2` - **77.5% pass rate** (frontier baseline)
- `anthropic/claude-3-5-haiku-latest` - Fast, capable

---

## Benchmark Runs

### 2025-12-29: GPT-5.2 (Full - 200/200) - Frontier Baseline

**Model:** `openai/gpt-5.2`
**Config:** split=bench, mode=mvp, 200 examples
**Status:** Complete

| Archetype | Avg Reward | Perfect | Pass Rate |
|-----------|------------|---------|-----------|
| `mvp.table_list_of_lists` | 1.000 | 20/20 | 100% |
| `mvp.class_reserved_word` | 1.000 | 20/20 | 100% |
| `mvp.string_returns_none` | 1.000 | 20/20 | 100% |
| `mvp.table_list_of_dicts` | 0.985 | 18/20 | 100% |
| `mvp.extract_text_by_id` | 0.950 | 19/20 | 95% |
| `mvp.none_attribute_error` | 0.950 | 19/20 | 95% |
| `mvp.extract_text_by_class` | 0.850 | 17/20 | 85% |
| `mvp.multivalue_class` | 0.850 | 17/20 | 85% |
| `mvp.limit_js_required` | 0.273 | 0/20 | 15% |
| `mvp.limit_image_text` | 0.017 | 0/20 | 0% |
| **Total** | **0.787** | **150/200** | **77.5%** |

**Observations:**
- **string_returns_none: 100%** - GPT-5.2 fully understands the `.string` vs `get_text()` gotcha
- Extraction tasks excellent (85-100%)
- Limitations still challenging (0-15%) - even frontier models struggle here
- 288 tool calls (~1.4/example), efficient

**Token Usage:** 657K input, 127K output (~784K total)

---

### 2025-12-28: Ministral 3 8B (Full - 200/200) ⭐ BEST 8B

**Model:** `mistralai/ministral-8b-2512` (Ministral 3, December 2025)
**Config:** split=bench, mode=mvp, 200 examples
**Status:** Complete

| Archetype | Avg Reward | Perfect | Pass Rate |
|-----------|------------|---------|-----------|
| `mvp.table_list_of_dicts` | 1.000 | 20/20 | 100% |
| `mvp.table_list_of_lists` | 1.000 | 20/20 | 100% |
| `mvp.class_reserved_word` | 1.000 | 20/20 | 100% |
| `mvp.extract_text_by_id` | 0.950 | 19/20 | 95% |
| `mvp.none_attribute_error` | 0.950 | 19/20 | 95% |
| `mvp.extract_text_by_class` | 0.850 | 17/20 | 85% |
| `mvp.multivalue_class` | 0.850 | 17/20 | 85% |
| `mvp.string_returns_none` | 0.500 | 10/20 | 50% |
| `mvp.limit_js_required` | 0.127 | 0/20 | 13% |
| `mvp.limit_image_text` | 0.000 | 0/20 | 0% |
| **Total** | **0.723** | **142/200** | **71%** |

**Observations:**
- BEST 8B model tested: 71% vs Qwen3's 52.5% vs old Ministral's 27.5%
- Perfect on tables AND class_reserved_word (Ministral 3 knows BS4 well)
- Very efficient: 258 tool calls (1.3/example), ~587K tokens total
- Still struggles with limitations (0-13%) like all models
- string_returns_none still challenging (50%) - RL opportunity

**Token Usage:** 497K input, 91K output (~588K total) - cheap and efficient

---

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
