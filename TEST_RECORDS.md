# Test Records

Track evaluation progress and results. Update this file after each benchmark run.

---

## Baseline vs Trained Model Performance

Track RL training progress by comparing baseline (pre-training) to trained model performance.

| Model | Baseline | After Training | Improvement | Status |
|-------|----------|----------------|-------------|--------|
| Qwen3-4B | 0.0% | TBD | - | RL target (no function calling) |
| Llama 3.2-3B | 0.0% | TBD | - | RL target (no function calling) |
| Ministral 3B | 50.6% | TBD | - | Strong small model |
| Qwen3-8B | 43.1% | TBD | - | RL candidate |
| Ministral 8B | 68.4% | TBD | - | 8B ceiling |

*Last updated: 2026-01-01*

**Training config:** `split=train`, `mode=all`, `difficulty=mixed` (see README for mode options)

---

## Target Models

Focus on **small/weak models** - they benefit most from RL training on this environment.

### Tier 1: Small Models (RL Training Targets)
- [x] `qwen/qwen3-4b:free` - **0% pass rate** (20 tool calls, all fail - ideal RL target!)
- [x] `meta-llama/llama-3.2-3b-instruct` - **0% pass rate** (24 tool calls, all fail)
- [ ] `meta-llama/llama-3.2-1b-instruct` - ❌ No function calling support
- [ ] `google/gemma-3-4b-it` - ❌ No function calling support
- [ ] `google/gemma-3n-*` - ❌ No function calling support

### Tier 2: Medium Models (8B Validation)
- [x] `mistralai/ministral-8b-2512` - **BEST 8B: 68.4% pass rate** (Ministral 3, Jan 2026, 680 examples)
- [x] `mistralai/ministral-3b-2512` - **50.6% pass rate** (Ministral 3B, Jan 2026, 680 examples, $1.10)
- [x] `qwen/qwen3-8b` - **43.1% pass rate** (680 examples)
- [x] `mistralai/ministral-8b` - Old version (**27.5% pass rate** - loopy, expensive)

### Tier 3: Large Models (Ceiling Reference)
- [x] `moonshotai/kimi-k2` - **72.8% pass rate** (680 examples)
- [x] `z-ai/glm-4.7` - **74.7% pass rate** (680 examples)
- [ ] `minimax/minimax-m2.1` - Not yet tested

### Blocked (No Function Calling)
- ~~`meta-llama/llama-3.2-1b-instruct`~~ - No `tools` support on OpenRouter
- ~~`google/gemma-3-4b-it`~~ - No `tools` support on OpenRouter
- ~~`google/gemma-3n-*`~~ - No `tools` support on OpenRouter
- ~~`deepseek/deepseek-r1-0528-qwen3-8b`~~ - No function calling support
- ~~`prime-intellect/intellect-3`~~ - OpenRouter function calling broken
- ~~`x-ai/grok-4.1-fast`~~ - Rate limits too aggressive

---

## Benchmark Runs (52 Archetypes, 1040 Bench Examples)

### 2026-01-01: New Limitation Archetypes Baseline (Ministral 3B)

**Environment Update:** Added 5 limitation archetypes (PR #3):
- `limit_canvas_text` - Text rendered in HTML5 canvas
- `limit_svg_path_data` - Data only in SVG path/shape elements
- `limit_pdf_embed` - Content in embedded PDF documents
- `limit_js_required` - JavaScript-rendered content (existing)
- `limit_image_text` - Text only in images (existing)

**Model:** `mistralai/ministral-3b-2512`
**Config:** split=bench, limitation archetypes only (100 examples)

| Archetype | Pass Rate | Avg Reward | Notes |
|-----------|-----------|------------|-------|
| limit_pdf_embed | **40%** | 0.200 | Easiest to detect |
| limit_js_required | **30%** | 0.150 | Familiar pattern |
| limit_svg_path_data | **20%** | 0.100 | Path data opaque |
| limit_canvas_text | **5%** | 0.025 | Canvas API hidden |
| limit_image_text | **5%** | 0.025 | Hardest to recognize |

**Key Findings:**
1. **All limitation archetypes hard for 3B models** - 5-40% pass rates
2. **No perfect scores possible** - Max reward is 0.50 for correct limitation detection
3. **pdf_embed easiest** - PDF extension visible in source
4. **canvas/image hardest** - Require understanding of rendering techniques

**Solvable archetype sample (same model):**
- `extract_attribute`: 73% avg reward, 75% pass rate
- `extract_emoji_content`: 86.5% avg reward, 100% pass rate

This confirms limitation detection is a major RL training opportunity - models that learn to recognize unsolvable tasks gain significant reward.

---

### 2026-01-01: Small Model Baseline Testing (Qwen3-4B, Llama 3.2-3B)

Tested small models with function calling support to establish RL training baselines.

**Key Finding:** Both small models get **0% pass rate** despite making tool calls. This is ideal for RL training - there's massive room for improvement!

#### Qwen3-4B (qwen/qwen3-4b:free)

| Metric | Value |
|--------|-------|
| **Pass rate** | 0% |
| **Perfect rate** | 0% |
| **Average reward** | 0.000 |
| **Tool calls** | 20 (1.0/task) |
| **Tokens** | 31K in, 15K out |

#### Llama 3.2 3B (meta-llama/llama-3.2-3b-instruct)

| Metric | Value |
|--------|-------|
| **Pass rate** | 0% |
| **Perfect rate** | 0% |
| **Average reward** | 0.000 |
| **Tool calls** | 24 (1.2/task) |
| **Tokens** | 46K in, 2.4K out |

#### Function Calling Support Check

Verified via OpenRouter API (`supported_parameters` field):

| Model | Has `tools` | Can Benchmark? |
|-------|-------------|----------------|
| `qwen/qwen3-4b:free` | ✅ | Yes (0% pass) |
| `qwen/qwen3-8b` | ✅ | Yes (43% pass) |
| `meta-llama/llama-3.2-3b-instruct` | ✅ | Yes (0% pass) |
| `meta-llama/llama-3.2-1b-instruct` | ❌ | No |
| `google/gemma-3-4b-it` | ❌ | No |
| `google/gemma-3n-*` | ❌ | No |

**Benchmark Calibration Summary:**

| Tier | Models | Pass Rate | RL Opportunity |
|------|--------|-----------|----------------|
| Small (3-4B) | Qwen3-4B, Llama 3.2-3B | 0% | Huge! |
| Medium (8B) | Qwen3-8B, Ministral-8B | 43-68% | Moderate |
| Large | Kimi K2, GLM-4.7 | 72-75% | Ceiling |

---

### 2026-01-01: Ministral 8B Full Benchmark with Efficiency Guidelines

**Model:** `mistralai/ministral-8b-2512` (Ministral 3 8B)
**Config:** split=bench, mode=all, 680 examples (34 archetypes)
**Prompt:** With efficiency guidelines

| Metric | Value |
|--------|-------|
| **Pass rate (≥0.5)** | 68.4% |
| **Perfect rate (=1.0)** | 57.9% |
| **Average reward** | 0.665 |
| **Total tool calls** | 1,027 |
| **Total tokens** | ~4M |

**Perfect Archetypes (100%):**
- attribute_selector, class_reserved_word, deep_nesting_extraction
- extract_images, extract_multilingual, form_action_method
- remove_scripts_styles, select_options

**Hardest Archetypes (0% perfect):**
- list_extraction (0%) - **NOTE: Fix deployed but not in this benchmark**
- label_input_mapping (0%) - Complex form/label relationships

**Comparison with Ministral 3B:**

| Model | Pass Rate | Perfect Rate | Avg Reward |
|-------|-----------|--------------|------------|
| **Ministral 8B** | **68.4%** | **57.9%** | **0.665** |
| Ministral 3B | 50.6% | 19.7% | 0.453 |

**Key Finding:** 8B model shows +17.8% pass rate over 3B - significant scaling improvement. Both still struggle with multi-step and aggregation tasks.

---

### 2026-01-01: Ministral 3B Full Benchmark (680 examples)

**Model:** `mistralai/ministral-3b-2512` (Ministral 3B, with efficiency guidelines)
**Config:** split=bench, mode=all, 680 examples (34 archetypes)
**Cost:** $1.10

| Metric | Value |
|--------|-------|
| **Pass rate (≥0.5)** | 50.6% |
| **Perfect rate (=1.0)** | 19.7% |
| **Average reward** | 0.453 |
| **Total tool calls** | 3,117 |
| **Total tokens** | ~11M |

**Top Performing Archetypes:**

| Archetype | Avg Reward | Perfect |
|-----------|------------|---------|
| extract_images | 0.945 | 14/20 |
| select_options | 0.910 | 10/20 |
| extract_links | 0.905 | 14/20 |
| class_reserved_word | 0.875 | 9/20 |
| deep_nesting_extraction | 0.825 | 11/20 |
| direct_children | 0.785 | 0/20 |

**Hardest Archetypes (0% pass):**
- aggregation_min_max, semantic_decoy_extreme, count_elements
- relational_query, list_extraction, partial_data_extraction

**Comparison with 8B models:**

| Model | Pass Rate | Perfect Rate | Avg Reward |
|-------|-----------|--------------|------------|
| Ministral 3 8B | 63.2% | 56.2% | 0.621 |
| **Ministral 3B** | **50.6%** | **19.7%** | **0.453** |
| Qwen3-8B | 43.1% | 26.6% | 0.401 |

**Key Finding:** 3B model achieves 50.6% pass rate - better than Qwen3-8B (43.1%) despite being 2.7x smaller. Good RL training candidate.

---

### 2026-01-01: Weak Archetype Investigation

Investigated why certain archetypes have 0% pass rate on small models. Key findings:

#### 1. mvp.list_extraction (0% pass, marked "easy")

**Root Cause:** Ambiguous query + massive chrome pollution
- Query: "Extract all item texts from the list" - which list?
- HTML contains **292 `<li>` elements** from navigation chrome but only **4 target items** with class `list-item`
- Model uses `find_all('li')` instead of `find_all(class_='list-item')`, returning 292 items
- HTML size: 76KB (50KB+ chrome)

**Status:** ✅ FIXED - Query changed from "the list" to ".item-list element" to remove ambiguity while keeping realistic chrome difficulty. Commit: `cad186b`

#### 2. mvp.partial_data_extraction (5% pass)

**Root Cause:** Complex output schema + nullable fields
- Expects list of dicts with `name`, `price`, `description` keys
- Some fields are `null` (missing in HTML)
- Small models struggle with structured nullable output
- HTML size: 94KB

**Status:** Legitimate difficulty. Tests real-world scraping patterns where not all data is present.

#### 3. Multi-step archetypes (0% pass)

These require `navigate` tool for multi-page extraction:
- `search_then_detail`: Find item, navigate to detail page, extract data
- `link_chain`: Follow breadcrumb chain to destination
- `compare_products`: Visit multiple product pages, compare prices

**Status:** Legitimately hard for pre-RL models. Tests planning and multi-step execution. These archetypes are designed to challenge models.

#### 4. Computation archetypes (23-48% pass)

- `aggregation_min_max` (23%): Extract all prices, compute min/max
- `count_elements` (48%): Count products matching condition

**Status:** These are valid hard archetypes. Models must extract + compute, not just extract.

#### Summary

| Category | Archetypes | Issue | Fix Needed? |
|----------|------------|-------|-------------|
| Ambiguous query | list_extraction | Chrome pollution | ✅ Fixed (cad186b) |
| Complex schema | partial_data_extraction | Nullable fields | No - valid difficulty |
| Multi-step | search_then_detail, link_chain, compare_products | Navigate tool | No - designed hard |
| Computation | aggregation_min_max, count_elements | Extract + compute | No - valid signal |

**Recommendation:** These archetypes provide good RL training signal. The 0% pass rates on pre-RL small models indicate genuine capability gaps that training should address.

---

### 2025-12-31: Efficiency Guidelines Added (Prompt Improvement)

**Change:** Added explicit efficiency guidelines to system prompt in `bs4_env/prompt.py`:
```
## Efficiency
- Most tasks can be solved in 1-3 tool calls
- If your code produces valid output matching the expected format, finalize immediately
- If you've tried 3+ different approaches without success, provide your best answer
- Excessive tool calls (10+) result in zero reward regardless of correctness
```

**Test Results** (Ministral 3B 2512, 5 instances per archetype):

| Archetype | Before | After | Change |
|-----------|--------|-------|--------|
| mvp.extract_attribute | 60% | **80%** | +20% |
| mvp.extract_links | 80% | **100%** | +20% |
| mvp.extract_images | 100% | **100%** | same |
| mvp.direct_children | 60% | **80%** | +20% |
| mvp.descendants_filter | 60% | **100%** | +40% |
| mvp.table_column_by_header | 100% | 80% | -20% |
| mvp.remove_scripts_styles | 100% | **100%** | same |
| **Overall** | **80%** | **91.4%** | **+11.4%** |

**Cost:** ~$0.03-0.04 (35 API calls to Ministral 3B)

**Key Finding:** Efficiency guidelines reduced no-output loop failures significantly. The prompt helps models learn to finalize rather than looping indefinitely.

---

### 2025-12-31: New Archetypes Added (7 Core Extraction + Traversal)

**Completed PRD target of 50 archetypes.** Added 7 new archetypes to fill remaining gaps.

**Quick Test Results** (Ministral 3B 2512, 5 instances per archetype, BEFORE efficiency guidelines):

| Archetype | Pass Rate | Notes |
|-----------|-----------|-------|
| mvp.extract_attribute | 60% | 2 no-output loops |
| mvp.extract_links | 80% | Good |
| mvp.extract_images | **100%** | Perfect |
| mvp.direct_children | 60% | 2 no-output loops |
| mvp.descendants_filter | 60% | 1 order mismatch, 2 no-output |
| mvp.table_column_by_header | **100%** | Perfect |
| mvp.remove_scripts_styles | **100%** | Perfect |
| **Overall** | **80%** (28/35) | Solid performance |

**Failure Analysis:**
- **No-output loops** (7 cases): Model made 10 tool calls without final answer - fixed by efficiency guidelines
- **Order mismatch** (1 case): `descendants_filter` found correct items in wrong order - consider order-agnostic grading
- **Wrong data** (1 case): Model extracted text instead of `title` attribute - genuine skill gap

**Codex Review Findings (Fixed):**
- `direct_children`: Simplified HTML structure to avoid `get_text()` including nested text
- `table_column_by_header`: Added dynamic data generation to prevent reward hacking via memorization

---

### 2025-12-31: Chinese Models Benchmarked (Kimi K2, GLM-4.7)

Tested cheaper alternatives to GPT-5.2 from Chinese providers.

| Model | Pass Rate | Perfect Rate | Avg Reward | Cost | Runtime |
|-------|-----------|--------------|------------|------|---------|
| **GPT-5.2** | **75.6%** | **57.5%** | **0.726** | ~$1.20 | 2h49m |
| **GLM-4.7** | **74.7%** | 45.1% | 0.684 | ~$0.51 | 3h58m |
| **Kimi K2** | 72.8% | 45.0% | 0.666 | ~$0.28 | 5h45m |
| Ministral 3 8B | 63.2% | 56.2% | 0.621 | ~$0.07 | 1h19m |
| Qwen3-8B | 43.1% | 26.6% | 0.401 | ~$0.16 | 4h30m |

**Key Findings:**
1. **GLM-4.7 is best value**: Nearly GPT-5.2's pass rate at 43% of the cost
2. **Kimi K2** cheaper but slower, slightly worse pass rate
3. Both Chinese models have significantly lower **perfect rate** (45% vs 57.5%)

**Model-Specific Weaknesses:**

| Model | Strength | Weakness |
|-------|----------|----------|
| GLM-4.7 | Reliable, balanced | Multi-step nav (0-6% perfect) |
| Kimi K2 | Better multi-step nav (+15-18%) | `string_returns_none` 0% (BS4 gotcha blind) |

---

### 2025-12-31: All Models Re-benchmarked (Navigate Tool Fix)

**Critical Bug Fixed:** Previous benchmarks had the `navigate` tool missing - multi-step tasks were impossible to solve. See TODO.md for details.

| Model | Pass Rate | Perfect Rate | Avg Reward | Multi-Step |
|-------|-----------|--------------|------------|------------|
| **GPT-5.2** | **75.6%** | 57.5% | 0.726 | Can navigate |
| Ministral 3 8B | 63.2% | 56.2% | 0.621 | Struggles |
| Qwen3-8B | 43.1% | 26.6% | 0.401 | Struggles |

**Multi-Step Archetype Breakdown:**

| Archetype | GPT-5.2 | Ministral 3 | Qwen3 | Notes |
|-----------|---------|-------------|-------|-------|
| search_then_detail | **100%** | 45% | 5% | GPT-5.2 perfect! |
| link_chain | **40%** | 0% | 5% | Hard for 8B |
| pagination_aggregate | **20%** | 15% | 0% | Aggregation hard |
| compare_products | 0% | 0% | 0% | Schema bug* |
| list_extraction | 5% | 0% | 5% | Very hard |

*compare_products failed due to schema key names not shown in prompt (fixed in `prompt.py`, needs re-benchmark)

**Key Findings:**
1. GPT-5.2 clearly dominates multi-step tasks (100% on search_then_detail)
2. 8B models can navigate but struggle with multi-step reasoning
3. The gap between frontier and 8B models is now properly measured

---

### 2025-12-29: Ministral 3 8B (Full - 680/680) ⭐ BEST 8B

**Model:** `mistralai/ministral-8b-2512` (Ministral 3, December 2025)
**Config:** split=bench, mode=all, 680 examples (34 archetypes)
**Status:** Complete

| Category | Archetypes | Avg Reward | Pass Rate |
|----------|------------|------------|-----------|
| **Perfect (100%)** | deep_nesting, extract_text_by_id, class_reserved_word, extract_multilingual, extract_rtl, table_list_of_dicts | 1.000 | 100% |
| **Excellent (90%+)** | attribute_selector, multivalue_class, json_ld_array, none_attribute_error, css_combinator, parser_differences, extract_emoji, navigablestring_parent | 0.95 | 95%+ |
| **Good (80%+)** | extract_text_by_class, sibling_navigation | 0.80 | 83% |
| **Moderate (50-80%)** | relational_query, multi_hop_filter, json_ld_extraction, string_returns_none | 0.67 | 71% |
| **Challenging (20-50%)** | count_elements, structured_output, semantic_decoy, semantic_ambiguity, aggregation_min_max | 0.41 | 44% |
| **Hard Multi-Step (0-20%)** | limit_js_required, pagination_aggregate, partial_data, list_extraction, limit_image_text, search_then_detail, link_chain, compare_products | 0.04 | 0.4% |
| **Total** | 34 | **0.624** | **63.2%** |

**Observations:**
- Still BEST 8B: 63.2% on hard test (was 71% on easy test)
- **Perfect on 6 archetypes** including i18n tasks (multilingual, RTL)
- Multi-step navigation tasks are very hard (0-3% pass rate)
- Semantic challenges moderate difficulty (semantic_ambiguity: 45%, semantic_decoy: 50%)
- string_returns_none improved: 65% pass (was 50% on smaller test)

---

### 2025-12-29: Qwen3-8B (Full - 680/680)

**Model:** `qwen/qwen3-8b`
**Config:** split=bench, mode=all, 680 examples (34 archetypes)
**Status:** Complete

| Category | Archetypes | Avg Reward | Pass Rate |
|----------|------------|------------|-----------|
| **Excellent (90%+)** | table_list_of_lists | 0.99 | 100% |
| **Good (70-90%)** | json_ld_extraction, attribute_selector, none_attribute_error, class_reserved_word, parser_differences, whitespace_sibling | 0.76 | 80% |
| **Moderate (50-70%)** | extract_multilingual, table_list_of_dicts, navigablestring_parent, extract_rtl, deep_nesting, css_combinator, multivalue_class, extract_text_by_id, extract_text_by_class, extract_emoji | 0.52 | 56% |
| **Challenging (20-50%)** | string_returns_none, count_elements, semantic_ambiguity, aggregation_min_max, relational_query | 0.29 | 31% |
| **Very Hard (<20%)** | limit_js_required, partial_data, multi_hop_filter, semantic_decoy, structured_output, sibling_navigation, list_extraction, limit_image_text, search_then_detail, pagination_aggregate, link_chain, compare_products | 0.03 | 1.3% |
| **Total** | 34 | **0.376** | **39.6%** |

**Observations:**
- Significant drop from 52.5% (10 archetypes) to 39.6% (34 archetypes)
- Tables still strong (99-100%), but many new hard archetypes
- Multi-step tasks nearly impossible (0% on link_chain, compare_products, etc.)
- string_returns_none: 40% pass (was 20% - improvement with more data?)
- semantic_decoy_extreme is hardest semantic task (10% pass)

---

### 2025-12-30: GPT-5.2 (Full - 680/680) ⭐ FRONTIER BASELINE

**Model:** `openai/gpt-5.2`
**Config:** split=bench, mode=all, 680 examples (34 archetypes)
**Status:** Complete

| Category | Archetypes | Avg Reward | Pass Rate |
|----------|------------|------------|-----------|
| **Perfect (100%)** | deep_nesting, sibling_navigation, extract_text_by_id, string_returns_none, class_reserved_word, extract_multilingual, extract_rtl, extract_emoji, multivalue_class, table_list_of_dicts, whitespace_sibling | 1.000 | 100% |
| **Excellent (95%+)** | json_ld_extraction, table_list_of_lists, relational_query, attribute_selector, none_attribute_error, semantic_ambiguity | 0.97 | 97% |
| **Good (80-95%)** | parser_differences, semantic_decoy_extreme, extract_text_by_class | 0.88 | 88% |
| **Moderate (50-80%)** | css_combinator, navigablestring_parent, count_elements, multi_hop_filter, structured_output | 0.65 | 65% |
| **Challenging (20-50%)** | aggregation_min_max, limit_js_required | 0.27 | 27% |
| **Hard Multi-Step (0-20%)** | partial_data_extraction, pagination_aggregate, limit_image_text, list_extraction, search_then_detail, link_chain, compare_products | 0.03 | 3% |
| **Total** | 34 | **0.692** | **70.9%** |

**Observations:**
- **Frontier baseline**: 70.9% pass rate vs Ministral 3's 63.2% (+7.7%)
- **Perfect on 11 archetypes** (vs 6 for Ministral 3)
- **string_returns_none: 100%** - fully understands the `.string` vs `get_text()` gotcha
- **Semantic tasks excellent**: ambiguity 94.5%, decoy 89.5% (vs ~45-50% for 8B models)
- Multi-step navigation still very hard (0% on 4 archetypes)
- Efficient: 1,448 tool calls (~2.1/example)

**Token Usage:** 3.96M input, 851K output (~4.8M total)

---

## Phase 1 MVP Benchmark (13 Archetypes, 260 Examples)

### 2025-12-31: Ministral 3B v3 (Limit Prompt Guidance)

**Model:** `mistralai/ministral-3b`
**Config:** split=bench, mode=mvp, 260 examples (13 archetypes)
**Changes:** Added detailed limitation detection guidance to prompt

| Model | Pass Rate | Perfect Rate | Avg Reward | Change |
|-------|-----------|--------------|------------|--------|
| Ministral 3B v3 | **65.4%** | 51.5% | 0.630 | +3.1% |
| Ministral 3B v2 | 62.3% | 59.2% | 0.620 | baseline |

**Per-Archetype Changes:**

| Archetype | v2 | v3 | Change | Notes |
|-----------|-----|-----|--------|-------|
| table_list_of_lists | 15% | **96.5%** | +81.5% | 🎉 MASSIVE improvement |
| table_list_of_dicts | 75% | **81%** | +6% | Better |
| json_ld_extraction | 55% | **29.5%** | -25.5% | Regression? |
| navigablestring_parent | 32% | **29%** | -3% | Still hard |
| limit_js_required | 0% | **0%** | 0% | No improvement |
| limit_image_text | 0% | **0%** | 0% | No improvement |

**Key Finding:** Prompt guidance alone **does NOT help** small models recognize limitations. The limit archetypes remained at 0% despite detailed instructions. This confirms these tasks are a **core RL training opportunity** - models must learn through rewards, not prompting.

**File:** `results_ministral3_v3_limit_guidance.json`

---

### 2025-12-31: Full Model Comparison (Prompt & Efficiency Fix)

**Changes Made:**
1. Fixed efficiency penalty not applying to limit tasks (limit responses now get full 0.5)
2. Added limitation detection guidance to prompt (JS-rendered, image text)
3. Fixed `make_soup(HTML)` confusion in prompt

| Model | Pass Rate | Perfect Rate | Avg Reward | Notes |
|-------|-----------|--------------|------------|-------|
| **GLM-4.7** | **81.2%** | 78.8% | 0.832 | Frontier baseline |
| Ministral 3B | 62.3% | 59.2% | 0.620 | Best small model |
| Qwen3-8B | 55.0% | 35.4% | 0.515 | Struggles on basics |

**Per-Archetype Breakdown:**

| Archetype | GLM-4.7 | Ministral 3B | Qwen3-8B | Notes |
|-----------|---------|--------------|----------|-------|
| mvp.extract_text_by_id | **1.00** | 1.00 | 0.40 | Qwen struggles |
| mvp.extract_text_by_class | 0.85 | 0.80 | 0.46 | Decoy confusion |
| mvp.string_returns_none | 0.75 | 0.90 | 0.45 | BS4 gotcha |
| mvp.none_attribute_error | **0.95** | 0.95 | 0.61 | Good coverage |
| mvp.class_reserved_word | **1.00** | 1.00 | 0.55 | Qwen weaker |
| mvp.json_ld_extraction | **1.00** | 0.55 | 0.82 | Ministral struggles |
| mvp.limit_js_required | 0.28 | 0.00 | 0.13 | **Hard for all** |
| mvp.limit_image_text | 0.02 | 0.00 | 0.00 | **Very hard** |
| mvp.multivalue_class | **1.00** | 0.95 | 0.64 | Good |
| mvp.navigablestring_parent | **0.97** | 0.32 | 0.40 | Ministral weak |
| mvp.whitespace_sibling | **1.00** | 0.69 | 0.54 | Sibling gotcha |
| mvp.table_list_of_dicts | **0.99** | 0.75 | 0.80 | Tables OK |
| mvp.table_list_of_lists | **1.00** | 0.15 | 0.91 | Ministral parsing bug |

**Key Findings:**
1. **Limit archetypes fail on ALL models** (0-28%) - even GLM-4.7 only gets 28% on JS detection
2. **Ministral 3B beats Qwen3-8B** despite being 3B vs 8B - better at following instructions
3. **table_list_of_lists**: Ministral at 15% due to `get_text().split()` bug, Qwen at 91% - different parsing strategies
4. **navigablestring_parent**: GLM-4.7 at 97% vs Ministral at 32% - frontier models understand NavigableString

**Files:**
- `results_glm47_post_fix.json` - GLM-4.7 results
- `results_ministral3_v2.json` - Ministral 3B results
- `results_qwen3_post_fix.json` - Qwen3-8B results

---

## Legacy Benchmark Runs (10 Archetypes, 200 Examples)

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

- Environment has **53 archetypes** total (1060 bench examples)
  - Core extraction: 10 archetypes
  - Advanced/gotcha: 8 archetypes
  - Multi-step navigation: 5 archetypes (link_chain, search_then_detail, etc.)
  - Semantic challenges: 4 archetypes
  - i18n: 3 archetypes
  - **Limitations: 5 archetypes** (canvas_text, svg_path_data, pdf_embed, js_required, image_text)
  - Aggregation/counting: 2 archetypes
  - Forms: 6 archetypes
  - Tables: 4 archetypes
  - Error bait: 3 archetypes
  - Hard: 3 archetypes
- Bench split uses seeds 110000-111000 (20 per archetype)
- Efficiency penalty: -10% per extra tool call, floor at 0.2, cutoff at 11+
- Run benchmarks on cloud via `.github/workflows/bench.yml` (Namespace runners)

## Benchmark Calibration

| Tier | Model | Pass Rate | Perfect Rate | RL Target? |
|------|-------|-----------|--------------|------------|
| Small | Qwen3-4B | 0% | 0% | ✅ Ideal |
| Small | Llama 3.2-3B | 0% | 0% | ✅ Ideal |
| Medium | Qwen3-8B | 43.1% | 31.6% | ✅ Good |
| Medium | Ministral-8B | 68.4% | 57.9% | Validation |
| Large | Kimi K2 | 72.8% | 60.5% | Ceiling |
| Large | GLM-4.7 | 74.7% | 63.2% | Ceiling |

**Key Insight:** 0% → 43% → 75% progression shows clear learning signal for RL training.
