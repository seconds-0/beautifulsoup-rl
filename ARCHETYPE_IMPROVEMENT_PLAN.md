# Archetype Improvement Plan

Based on multi-agent analysis (Claude, Codex, Gemini) of the v4-resilient training run.

## Executive Summary

| Archetype | Reward | Root Cause | Fix Type |
|-----------|--------|------------|----------|
| mvp.css_combinator | 4% | Missing CSS selector guidance in prompt | Prompt |
| mvp.sibling_navigation | 0% | Whitespace trap + vague prompt | Generator + Prompt |
| mvp.json_ld_array | 7% | Curriculum bias (first-match heuristic) | Generator |
| mvp.honeypot_detection | 14% | CSS parsing too hard, no semantic cues | Generator |

---

## 1. mvp.css_combinator (4% reward)

### Root Cause
- **Model doesn't know CSS selector syntax exists** - the system prompt explains BS4 gotchas but never mentions `.select()` or the `>` combinator
- **Comparison**: `mvp.direct_children` (51% reward) hints at `recursive=False`, while `css_combinator` gives no syntax hints
- **Validation**: Ministral 3 8B achieved 95%+ after training, proving the task is learnable

### Proposed Fixes

#### Fix 1: Add CSS selector section to system prompt (RECOMMENDED)
**File**: `bs4_env/prompt.py`

Add a section explaining CSS selectors:
```python
## CSS Selectors

BeautifulSoup supports CSS selectors via `.select()` and `.select_one()`:
- `soup.select('div.classname')` - find all matching elements
- `soup.select_one('div#id')` - find first match
- `soup.select('.parent > .child')` - direct children only (not descendants)
- `soup.select('.sibling + .next')` - adjacent sibling
```

#### Fix 2: Update task query to use "direct children" terminology
**File**: `bs4_env/generators/mvp_advanced.py`

```python
# OLD
query = "Extract only the TOP-LEVEL menu link texts from the .main-menu element."

# NEW
query = "Extract only the menu link texts that are *direct children* of the .main-menu element. Do NOT include submenu items (which are nested inside other elements)."
```

#### Fix 3: Adjust partial credit for "superset" errors
**File**: `bs4_env/grading/rubric.py`

Allow higher partial credit (0.3-0.5) when model finds all correct items but includes extras.

---

## 2. mvp.sibling_navigation (0% success, 21% reward)

### Root Cause
- **Whitespace trap**: Generator inserts newlines between `<dt>` and `<dd>` elements
- **Model behavior**: Uses `.next_sibling` (returns whitespace NavigableString) instead of `.find_next_sibling()` (returns element)
- **The 21% reward = Process Partial Credit** (0.30) for using BS4 correctly but extracting wrong value

### Proposed Fixes

#### Fix 1: Remove whitespace trap (RECOMMENDED)
**File**: `bs4_env/generators/mvp_advanced.py`

```python
# OLD - creates whitespace nodes
body_content += f'  <dt class="info-label">{label}:</dt>\n'
body_content += f'  <dd class="info-value">{value}</dd>\n'

# NEW - no whitespace between siblings
body_content += f'  <dt class="info-label">{label}:</dt>'
body_content += f'<dd class="info-value">{value}</dd>\n'
```

**Rationale**: `mvp.whitespace_sibling` already tests this gotcha with a clear hint. No need to duplicate the trap without guidance.

#### Fix 2: Clarify prompt to mention elements (Alternative)
**File**: `bs4_env/generators/mvp_advanced.py`

```python
# OLD
query = f'Find the value that comes after the "{target_label}:" label. Use sibling navigation.'

# NEW
query = f'Find the "{target_label}:" label, then extract the text from the next sibling element (ignore whitespace).'
```

---

## 3. mvp.json_ld_array (7% reward)

### Root Cause
- **Curriculum bias**: `mvp.json_ld_extraction` (Medium) always places target script FIRST
- **Learned heuristic**: Model uses `soup.find()` (first match) instead of iterating
- **Shuffle breaks heuristic**: `mvp.json_ld_array` (Hard) shuffles scripts, so first-match only works ~20% of the time

### Proposed Fixes

#### Fix 1: Fix curriculum by randomizing script order in Medium task (RECOMMENDED)
**File**: `bs4_env/generators/mvp_json_ld.py`

```python
# In JsonLdExtractionGenerator.generate()
# OLD - target always first
html = html[:head_end] + json_ld_script + extra_scripts + "\n" + html[head_end:]

# NEW - randomize position
scripts_list = [json_ld_script] + extra_scripts
rng.shuffle(scripts_list)
all_scripts = "\n".join(scripts_list)
html = html[:head_end] + all_scripts + "\n" + html[head_end:]
```

#### Fix 2: Make Hard task prompt more explicit
**File**: `bs4_env/generators/mvp_json_ld.py`

```python
# In JsonLdArrayGenerator.generate()
query = (
    f'{query_template} from the JSON-LD block with @type="{target_type}". '
    f'You must iterate through all <script type="application/ld+json"> elements to find the correct one.'
)
```

---

## 4. mvp.honeypot_detection (14% reward, 48% partial)

### Root Cause
- **Model detects simple honeypots** (`type="hidden"`) but misses CSS-hidden ones
- **CSS parsing is hard**: Detecting `style="display:none"` or `style="opacity:0"` requires string parsing
- **No semantic cues**: CSS-hidden honeypots lack `tabindex="-1"` or `aria-hidden="true"` attributes

### Proposed Fixes

#### Fix 1: Add semantic attributes to CSS honeypots (RECOMMENDED)
**File**: `bs4_env/generators/mvp_forms.py`

```python
# OLD
field_parts.append(
    f'<input type="{input_type}" name="{name}" style="{style_attr}">'
)

# NEW - add tabindex and aria-hidden
field_parts.append(
    f'<input type="{input_type}" name="{name}" style="{style_attr}" tabindex="-1" aria-hidden="true">'
)
```

**Rationale**:
- Matches real-world accessibility best practices
- Provides learnable signal beyond CSS parsing
- Aligns with existing `safety.py` checks

#### Fix 2: Add honeypot detection hints to prompt (Alternative)
**File**: `bs4_env/prompt.py` or task query

```python
# Add to system prompt or task-specific query
"Honeypots can be identified by: type='hidden', style attributes like 'display:none' or 'opacity:0',
tabindex='-1', aria-hidden='true', or suspicious field names."
```

---

## Implementation Priority

| Priority | Archetype | Fix | Impact | Effort |
|----------|-----------|-----|--------|--------|
| P0 | sibling_navigation | Remove whitespace trap | 0% → 60%+ | Low |
| P0 | json_ld_array | Fix curriculum bias | 7% → 40%+ | Low |
| P1 | css_combinator | Add CSS selector prompt | 4% → 30%+ | Low |
| P1 | honeypot_detection | Add semantic attributes | 14% → 40%+ | Low |
| P2 | All | Adjust partial credit | Smoother training | Medium |

---

## Test Plan

### 1. Unit Tests (Pre-commit)
```bash
# Verify changes don't break existing tests
pytest tests/

# Add regression tests for each fix
pytest tests/test_css_combinator.py -v
pytest tests/test_sibling_navigation.py -v
pytest tests/test_json_ld_array.py -v
pytest tests/test_honeypot_detection.py -v
```

### 2. Local Evaluation (Pre-deploy)
```bash
# Evaluate base model on each archetype
for arch in css_combinator sibling_navigation json_ld_array honeypot_detection; do
    uv run python -m bs4_env.scripts.eval_with_llm \
        --model qwen/qwen3-8b \
        --num 50 \
        --archetype "mvp.$arch"
done
```

### 3. Prime Validation (Post-deploy)
```bash
# Benchmark on Prime infrastructure
prime env eval seconds-0/beautiful-soup-env \
    -a '{"split":"bench","mode":"mvp"}' \
    -m qwen/qwen3-8b \
    -n 200
```

### 4. Training Validation (Final)
```bash
# Run short training to verify learning signal
uv run rl @ configs/prime-rl/qwen3-8b-2xh100-v5.toml \
    --max-steps 50 \
    --ckpt
```

Compare reward curves on target archetypes before/after fixes.

---

## Success Criteria

| Archetype | Current | Target | Method |
|-----------|---------|--------|--------|
| css_combinator | 4% | 30%+ | Prompt + query fix |
| sibling_navigation | 0% | 60%+ | Remove whitespace trap |
| json_ld_array | 7% | 40%+ | Curriculum fix |
| honeypot_detection | 14% | 40%+ | Semantic attributes |

---

## References

- Gemini analysis: `/tmp/gemini_*.md`
- Claude analysis: `/tmp/claude_*.md`
- Codex analysis: `/tmp/codex_*.md`
- Training data: `archetype_analysis.json`
