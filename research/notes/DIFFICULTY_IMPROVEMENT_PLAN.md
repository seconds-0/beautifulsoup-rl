# BeautifulSoup RL Environment: Difficulty Research & Improvement Plan

## Executive Summary

Research validates the analyst's claims. Pass rates are high because:
1. Only 4/24 archetypes are "hard" (17%), and only 2 are limitation tasks (8%)
2. Many queries specify exact IDs/classes, enabling single `find()` solutions
3. BS4 usage isn't enforced - regex/string parsing can pass
4. Uniform per-archetype sampling keeps the mix easy

---

## Part 1: Current State Analysis

### Archetype Distribution
- **Total**: 24 archetypes
- **Easy**: 6 (25%)
- **Medium**: 14 (58%)
- **Hard**: 4 (17%)
- **Limitation tasks**: 2 (8%)

### Why Pass Rates Are High
1. Queries provide exact selectors → single `find()` solutions work
2. BS4 not enforced → regex passes
3. Chrome adds noise but not semantic difficulty
4. Uniform sampling keeps mix easy

---

## Part 2: Target Architecture

### Difficulty Tiers (Expanded to ~30 archetypes)
```
Tier 1 (Easy): 6 archetypes     - Basic extraction, simple gotchas
Tier 2 (Medium): 10 archetypes  - Tables, JSON-LD, multi-class, i18n
Tier 3 (Hard): 8 archetypes     - Relational queries, semantic decoys
Tier 4 (Very Hard): 6 archetypes - Multi-hop, aggregation, complex tables
```

### Target Pass Rates (Stratified)
| Difficulty | Target Pass Rate | 8B Model Target | Frontier Target |
|------------|------------------|-----------------|-----------------|
| Easy | 60-80% | ~70% | ~95% (near saturation) |
| Medium | 30-60% | ~40% | ~70% |
| Hard | 10-30% | ~15% | ~40% |
| Very Hard | 5-15% | ~5% | ~20% |

---

## Part 3: New Archetypes to Add

### Single-Step Hard Archetypes
1. **`mvp.relational_query`** - "Extract price from row labeled 'Shipping'"
2. **`mvp.multi_hop_filter`** - Filter entities → navigate → extract
3. **`mvp.aggregation_min_max`** - "Find the lowest price among all products"
4. **`mvp.structured_output`** - Extract {name, price, sku, url} consistently
5. **`mvp.table_rowspan`** - Complex tables with row/column spans
6. **`mvp.json_ld_array`** - Multiple JSON-LD blocks, select by @type
7. **`mvp.semantic_decoy_extreme`** - 5+ near-identical elements
8. **`mvp.parser_required`** - HTML needing specific parser

### Multi-Step Archetypes
1. **`mvp.search_then_detail`** - Find item in list → extract from detail page
2. **`mvp.pagination_aggregate`** - Navigate pages → aggregate results
3. **`mvp.link_chain`** - Follow breadcrumb/link chain → extract at destination

---

## Part 4: Anti-Hacking Measures

### BS4 Enforcement: Penalize, Don't Reject
- Apply -0.15 reward penalty for pure regex/string solutions
- Creates gradient toward BS4 usage without artificial constraints

### Evidence Validation
- Require semantic understanding for limitation tasks
- Multiple valid evidences, must identify correct one

### Parser-Dependent Tasks
- HTML that parses differently across `lxml`/`html.parser`/`html5lib`

---

## Part 5: Implementation Phases

### Phase 0: Setup ✓
- Create worktree: `../beautifulsoup-rl-hard`
- Branch: `feature/hard-archetypes`

### Phase 1: Quick Wins
- Add BS4 usage detection with penalty
- Create `mode="hard_only"` and `mode="tiered"` presets
- Implement difficulty-weighted sampling

### Phase 2: New Hard Archetypes
- Add 6-8 single-step hard archetypes

### Phase 3: Sampling & Curriculum
- Adaptive sampling based on pass rates
- Progressive difficulty option

### Phase 4: Multi-Step Workflows
- Navigate tool with pre-generated HTML snapshots
- Multi-turn conversation support

---

## Part 6: Files to Modify

| File | Changes |
|------|---------|
| `bs4_env/tools/harness.py` | BS4 usage detection |
| `bs4_env/grading/rubric.py` | Penalty for non-BS4 |
| `bs4_env/dataset.py` | Difficulty-weighted sampling |
| `bs4_env/config.py` | New mode options |
| `bs4_env/generators/mvp_advanced.py` | New hard archetypes |
| `bs4_env/generators/mvp_tables.py` | Complex table archetypes |
| `bs4_env/generators/mvp_json_ld.py` | Multi-block JSON-LD |
| `bs4_env/tools/navigate.py` (new) | Navigate tool |
| `bs4_env/generators/mvp_multistep.py` (new) | Multi-step archetypes |

---

## Part 7: Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Hard archetypes | 4 (17%) | 14 (47%) |
| Limitation archetypes | 2 (8%) | 4-6 (13-20%) |
| Frontier model saturation | ~80%+ | <60% on hard |
| 8B model signal | Unknown | 15-40% on medium/hard |
| Semantic reasoning tasks | 1 | 8+ |
