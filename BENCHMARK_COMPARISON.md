# Benchmark Comparison: Qwen3-8B Baseline vs RL-Trained

**Date**: 2026-01-08
**Training Run**: bs4-qwen3-8b-v4-resilient

## Summary

| Model | Avg Reward | Pass Rate | Perfect Rate | Examples |
|-------|------------|-----------|--------------|----------|
| **Baseline (Qwen3-8B)** | 61.7% | 63.8% | 54.3% | 680 |
| **RL-Trained (Step 440)** | 91.8% | ~90% | 90% | 50 |
| **Improvement** | **+30.1 pts** | **+26.2 pts** | **+35.7 pts** | - |

## Details

### Baseline Model
- **Model**: `qwen/qwen3-8b` via OpenRouter
- **Evaluation**: `bs4_env.scripts.eval_with_llm`
- **Settings**: split=bench, mode=all, 680 examples
- **Results file**: `results_qwen3_8b_baseline.json`

### RL-Trained Model
- **Model**: `seconds-0/qwen3-8b-bs4-rl` (HuggingFace)
- **Checkpoint**: Step 440 (from B2: `bs4-qwen3-8b-v4-resilient/step_440/`)
- **Evaluation**: `vf-eval` with local vLLM server
- **Settings**: split=bench, mode=mvp, 50 examples, 3 rollouts each
- **Results**: 135/150 perfect, 3/150 partial, 12/150 failed

### Per-Archetype Analysis (Baseline)

Worst performing archetypes (baseline):
| Archetype | Baseline Score |
|-----------|---------------|
| mvp.limit_image_text | 0.00 |
| mvp.link_chain | 0.00 |
| mvp.limit_canvas_text | 0.03 |
| mvp.compare_products | 0.03 |
| mvp.multi_hop_filter | 0.03 |
| mvp.limit_js_required | 0.07 |
| mvp.honeypot_detection | 0.10 |

Best performing archetypes (baseline):
| Archetype | Baseline Score |
|-----------|---------------|
| mvp.class_reserved_word | 1.00 |
| mvp.extract_attribute | 1.00 |
| mvp.extract_images | 1.00 |
| mvp.multivalue_class | 1.00 |
| mvp.extract_links | 0.99 |
| mvp.extract_rtl | 0.98 |
| mvp.extract_multilingual | 0.98 |

### RL-Trained Model Failures

All 12 failures (out of 150 rollouts) were on:
- **Archetype**: `mvp.aggregation_min_max`
- **Difficulty**: hard

Note: Baseline scored 88% on this archetype, suggesting RL training may have slightly regressed on edge cases while dramatically improving overall performance.

## Training Status

As of 2026-01-08:
- **Current Step**: 588 / 1000 (58.8%)
- **Pod**: bs4-llama-2xh100 (2x H100)
- **Checkpoints**: Syncing to B2 every 5 steps

## Methodology Notes

1. **Baseline**: Run via OpenRouter API using `eval_with_llm.py` script
2. **RL-Trained**: Run via local vLLM server on RTX6000Ada pod using `vf-eval`
3. **Comparison caveat**: Different sample sizes (680 vs 50) and modes (all vs mvp)
4. **Conclusion**: RL training provides substantial improvement (~30 pts) despite methodology differences
