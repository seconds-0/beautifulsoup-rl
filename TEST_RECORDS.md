# Test Records

Track evaluation progress and results. Update this file after each benchmark run.

## Target Models

Focus on **small/weak models** - they benefit most from RL training on this environment.

### Priority (Small Models)
- [ ] `qwen/qwen3-8b` - Free tier available
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

### 2024-12-28: Structured JSON Output
- **Change:** Added `response_format={"type": "json_object"}` to API calls
- **Why:** Ensure valid JSON output from models
- **Files:** `eval_with_llm.py`

---

## Notes

- Environment has 24 archetypes total (13 phase 1, 8 phase 2, 3 i18n)
- Bench split uses seeds 110000-111000 (20 per archetype)
- Efficiency penalty: -10% per extra tool call, floor at 0.2, cutoff at 11+
