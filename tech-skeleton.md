I created a complete, highly-commented reference skeleton repository and packaged it as a downloadable zip.

[Download the reference implementation skeleton (zip)](sandbox:/mnt/data/beautiful_soup_env_skeleton.zip)

---

## High-level file structure

This is the intended structure (it is exactly what’s inside the zip), and each directory exists for a specific engineering reason.

```text
beautiful_soup_env_skeleton/
  beautiful_soup_env.py              # Environments Hub entrypoint: load_environment(...)
  README.md                          # How to use + safety posture + what to read first
  pyproject.toml                     # Packaging, dependencies, dev tools
  .gitignore

  bs4_env/
    __init__.py
    config.py                        # EnvConfig (split/mode/difficulty/executor/timeouts)
    registry.py                      # Archetype registry + specs
    auto_import.py                   # Central place to import generator modules to populate registry
    dataset.py                       # Build dataset rows for train/eval/bench
    bench_manifest.json              # Frozen (archetype_id, seed, difficulty) for Bench Mode
    prompt.py                        # Prompt formatting (no label leakage)

    generators/
      __init__.py
      base.py                        # Generator protocol, TaskInstance, RNG utilities, HTML noise helpers
      mvp_core_extraction.py         # Example implemented MVP archetypes (templates for your Top-50)
      mvp_tables.py                  # Example implemented table archetype + pattern
      mvp_limitations.py             # Example implemented unsolvable/limitation archetype + evidence rules
      phase2_placeholder.py          # Where Phase 2 generators go

    grading/
      __init__.py
      schema.py                      # JSON Schema validation (jsonschema)
      normalize.py                   # Deterministic normalization for strings/lists/dicts
      safety.py                      # Safety checks (token patterns, forbidden patterns)
      rubric.py                      # Deterministic reward computation (correct vs limit vs safety)

    tools/
      __init__.py
      executor.py                    # LocalSubprocessExecutor + PrimeSandboxExecutor stub
      harness.py                     # Runner script builder + globals injection + bs4 usage enforcement
      tool_defs.py                   # Tool entrypoints: run_python_tool(...)

    adapters/
      __init__.py
      verifiers_adapter.py           # One place to wire real Verifiers env + tools; includes fallback env

  bs4_env/scripts/
    __init__.py
    preview_dataset.py               # Print a few prompts + info sanity checks
    smoke_eval_local.py              # End-to-end local loop: tool call -> final JSON -> grade

  scripts/
    bench_models.py                  # Placeholder for `prime env eval` multi-model bench harness

  tests/
    test_determinism.py              # Generator determinism checks
    test_grading.py                  # Rubric correctness checks (ok + limit anti-hack)
    test_executor_local.py           # Local tool execution works
    test_env_smoke.py                # Smoke env construction + step loop
```

---

## How to read the skeleton in the “right order”

If you want an implementer (human or AI) to onboard quickly, the best reading order is:

1. `beautiful_soup_env.py` to see the public entrypoint and config surface.
2. `bs4_env/dataset.py` to see how tasks become dataset rows and how bench/train/eval splits work.
3. `bs4_env/generators/base.py` to understand how new tasks are authored safely and deterministically.
4. `bs4_env/grading/rubric.py` to understand reward computation and anti-hacking logic.
5. `bs4_env/tools/harness.py` and `bs4_env/tools/executor.py` to understand tool execution and enforcement.
6. `bs4_env/adapters/verifiers_adapter.py` to see where the actual Verifiers integration must be wired.

---

## What’s “fully implemented” vs “intentionally stubbed”

This skeleton is designed to be immediately runnable in a local Python environment *without Prime access* and still provide real confidence that your task generation + grading logic is sane.

### Implemented

* Deterministic procedural generation utilities (`stable_int_seed`, `make_rng`, noise/malformation helpers).
* A few fully working archetypes:

  * `mvp.extract_text_by_id`
  * `mvp.extract_all_links`
  * `mvp.table_list_of_dicts`
  * `mvp.limit_js_required` (unsolvable limitation task with evidence verification)
* Strict JSON-schema grading + deterministic normalization.
* Safety check module (minimal token patterns; easy to expand).
* Local tool execution via subprocess (good for tests and development).
* Local smoke-eval loop demonstrating: tool run → parse tool output → final JSON → grade.

### Stubbed (with very explicit TODO boundaries)

* Prime sandbox execution backend (`PrimeSandboxExecutor`) is left as a stub, because it must be wired to your actual Prime/Verifiers sandbox APIs.
* The Verifiers integration adapter is intentionally minimal and isolated in `bs4_env/adapters/verifiers_adapter.py`.

  * If `verifiers` is not installed, it returns a tiny `MinimalEnv` so tests work.
  * If `verifiers` *is* installed, it raises `NotImplementedError` so you do not accidentally think you shipped real integration.

This design prevents the common failure mode where a repo “kind of runs” but does not actually satisfy the bounty requirement of real sandbox integration.

---

## Notes on expanding this skeleton into your full Top‑50 MVP

You now have a clean “generator authoring pattern” with three working templates:

* A classic solvable extraction task.
* A structured extraction returning a list of dicts.
* A deliberately unsolvable “limitation required” task that gives partial credit only with evidence.

To implement the remaining MVP archetypes, you can follow this pattern:

* Add one generator module per theme (`mvp_forms.py`, `mvp_traversal.py`, `mvp_malformed.py`, `mvp_output_normalization.py`, etc.).
* Register each archetype in `registry.py` using the `@register(...)` decorator pattern shown.
* Add the new module import to `auto_import.py` so the registry is populated reliably.
* Add at least:

  * One determinism unit test per new module (or expand `test_determinism.py`).
  * One grading test that exercises the tricky failure mode (for example, `.string` returns `None`, reserved word `class`, missing attribute).

---

## Where implementers will wire Prime + Verifiers

Two files are the “official wiring points,” and the rest of the codebase stays stable:

1. `bs4_env/tools/executor.py`

   * Implement `PrimeSandboxExecutor.run(...)` using Prime’s sandbox infrastructure (or Verifiers’ sandbox utilities, if you’re using those).
   * Respect: timeout, output caps, and default network disabled.

2. `bs4_env/adapters/verifiers_adapter.py`

   * Replace `build_verifiers_environment(...)` so it returns a real `vf.Environment`.
   * Register tools so the model can call `run_python` and receive structured stdout/stderr/exit_code.

---

If you want, I can also generate a **second zip** that includes a “filled-in Top‑50 module layout” (all archetype IDs stubbed with TODOs and docstrings), so implementers can work through them methodically without inventing names, file placement, or registration boilerplate.
