# Review Response: BeautifulSoup RL Environment

This document responds to the comprehensive review of the BeautifulSoup RL environment. Each identified issue has been validated and addressed.

## Issues Addressed

### P0 - Critical Issues

#### P0.2: Sandbox Parameters Not Wired Through Config [FIXED]

**Original Issue:** README documented `docker_image`, `cpu_cores`, `memory_gb`, `timeout_minutes` parameters but they weren't wired through `EnvConfig` or `load_environment()` to the executor.

**Fix:**
- Added fields to `EnvConfig` in `bs4_env/config.py`
- Added parameters to `load_environment()` in `beautiful_soup_env.py`
- Updated `get_executor()` in `bs4_env/tools/executor.py` to accept and pass sandbox params
- Updated `verifiers_adapter.py` to pass all params from config to executor

**Files Modified:**
- `bs4_env/config.py`
- `beautiful_soup_env.py`
- `bs4_env/tools/executor.py`
- `bs4_env/adapters/verifiers_adapter.py`

---

#### P0.3: SVG Limitation Archetype Actually Solvable [FIXED]

**Original Issue:** `mvp.limit_svg_path_data` was marked as `solvable=False` but the data WAS recoverable:
- Bar charts: `height = value * 2` (divide by 2 to get exact value)
- Line charts: `y = 180 - value * 1.5` (reverse formula to get exact value)

**Fix:** Converted to a hard extraction task `mvp.extract_svg_geometry`:
- Set `solvable=True`
- Provide ground truth = the original data values list
- Updated query to explain the extraction task
- Added `INT_LIST_SCHEMA` for answer validation
- Included transformation hints in the query

**Files Modified:**
- `bs4_env/generators/mvp_limitations.py` (removed old archetype)
- `bs4_env/generators/mvp_hard.py` (added new extraction archetype)
- `bs4_env/config.py` (added `INT_LIST_SCHEMA`)
- `bs4_env/data/bench_manifest.json` (updated archetype IDs)
- `tests/test_limitation_archetypes.py` (removed SVG tests)

---

### P1 - Important Issues

#### P1.1: Prompt/Rubric Mismatch for Tool-Call Limits [FIXED]

**Original Issue:** Prompt said "10+ tool calls = zero reward regardless of correctness" but rubric exempts `status: "limit"` responses.

**Fix:** Updated prompt in `bs4_env/prompt.py` line 145 to:
```
Excessive tool calls (10+) result in zero reward for extraction tasks (limitation responses are exempt to allow exploration before abstaining)
```

---

#### P1.2: Inefficient Seed Selection [FIXED]

**Original Issue:** Used `shuffle(list(range(...)))` instead of more efficient `rng.sample(range(...))`.

**Fix:** Changed `bs4_env/dataset.py` `_select_seeds_for_archetype()` to use:
```python
available = range(seed_range[0], seed_range[1])
return list(rng.sample(available, min(num_examples, len(available))))
```

---

### P2 - Polish

#### P2.1: Model Training Results Section [ADDED]

**Added to README.md:**
- Benchmark calibration table showing 0% → 50% → 75% progression
- Archetype difficulty distribution breakdown
- Link to TEST_RECORDS.md for full details

---

## Verification

All 273 tests pass after changes:
```
pytest tests/ -v
# ======================= 273 passed, 1 warning in 36.07s ========================
```

Dataset generation verified:
```
python -m bs4_env.scripts.preview_dataset --num 3
# Dataset size: 28000 (working correctly)
```

---

## Final Checklist

- [x] P0.2: Sandbox params wired through EnvConfig → load_environment → get_executor → PrimeSandboxExecutor
- [x] P0.3: SVG archetype converted to `mvp.extract_svg_geometry` with `solvable=True` and ground truth
- [x] P1.1: Prompt updated to clarify limit responses exempt from tool-call penalty
- [x] P1.2: Seed selection uses `rng.sample()` instead of shuffle
- [x] P2.1: README has "Model Training Results" section with baseline scores
- [x] All tests pass (273/273)
- [x] TEST_RECORDS.md updated to reflect archetype count change (52 archetypes)

---

## Summary of Changes

| Category | Files Changed | Lines Added/Removed |
|----------|---------------|---------------------|
| P0.2 Config | 4 files | +40/-5 |
| P0.3 SVG | 5 files | +140/-155 |
| P1.1 Prompt | 1 file | +1/-1 |
| P1.2 Seed | 1 file | +2/-2 |
| P2.1 README | 2 files | +30/-3 |

**Total:** 8 unique files modified
