Below is a full rewrite of your PRD with the concrete modifications we discussed, including (a) a sharply specified environment contract, (b) reward anti-hacking rules, (c) a curated **Top‑50 MVP scenario set**, and (d) a **Phase 2** plan for the remaining archetypes.

After the PRD, you will find a step‑by‑step technical design and implementation plan that is intended to be sufficient for an implementing engineer or implementing AI agent to execute end‑to‑end: scaffolding, toolchain selection, environment configuration, architecture/dataflow, testing strategy, development verification loops, benchmark mode and cross‑model benchmarking, and finally the MVP milestone where you can run reinforcement learning (RL) training and show meaningful improvement.

---

# BeautifulSoup RL Environment

## Product Requirements Document (Rewritten)

**Author:** Alex
**Status:** Planning → Implementation
**Target:** Prime Intellect Environments Hub Bounty (Software Library Evals: BeautifulSoup)
**Last Updated:** December 12, 2025

---

## 1. Executive Summary

### 1.1 What We’re Building

We are building a reinforcement learning environment that trains and evaluates agents on **Beautiful Soup** (BeautifulSoup4), a Python library used for parsing and navigating HyperText Markup Language (HTML) and Extensible Markup Language (XML). The environment focuses on realistic scraping tasks that require both **technical proficiency** with BeautifulSoup and **judgment** about what is extractable from static HTML and what is not.

The environment will be published as a standalone Python package on Prime Intellect’s Environments Hub and will be compatible with the Verifiers environment interface, which expects a dataset, rollout protocol, and rubric-based reward computation. ([Prime Intellect Docs][1])

### 1.2 Why This Matters

Most scraping benchmarks use tidy, tutorial-style HTML. Production HTML is frequently malformed, noisy, misleading, or intentionally adversarial. Agents trained only on clean HTML will fail in the real world.

This environment is designed to train agents that:

* Extract data correctly from messy HTML using BeautifulSoup primitives.
* Avoid common BeautifulSoup gotchas and API traps.
* Recognize static parsing limitations (for example, JavaScript-rendered content, text embedded in images).
* Respect safety boundaries (for example, not extracting credentials or session tokens).

### 1.3 Success Criteria

This project is considered successful when all of the following are true:

1. The environment contains **at least 50 nontrivial task archetypes** with procedural variation and deterministic ground-truth verification.
2. Each episode supports **actual code execution** via a sandboxed Python tool interface, with network access disabled by default. ([Prime Intellect Docs][2])
3. Rewards are **verifiable and deterministic**, including a safe partial-credit path for correctly identifying BeautifulSoup’s limitations that is not reward-hackable.
4. We demonstrate **successful reinforcement learning training** using Prime’s RL infrastructure (prime‑rl), including before/after evaluation curves on held-out seeds. ([Prime Intellect Docs][3])
5. The repository exhibits “professional engineering quality,” including tests, reproducibility, and clear documentation, and it passes Hub Actions (automated build + test on push). ([Prime Intellect Docs][4])

---

## 2. Scope and Non‑Goals

### 2.1 In‑Scope

* Static HTML parsing and extraction tasks solvable with BeautifulSoup (and common parsers such as lxml and html5lib).
* Tool-based execution in a sandboxed Python environment.
* Deterministic reward computation from ground truth.
* Safe, structured “limitation detection” tasks where the correct behavior is to abstain and explain why static parsing cannot succeed.

### 2.2 Explicit Non‑Goals

* We will not train or evaluate bypass behavior for CAPTCHAs, paywalls, logins, or bot protection. The expected behavior in these cases is detection and safe abstention.
* We will not make live network requests to real websites during training or evaluation, and the default sandbox will disable network access. ([Prime Intellect Docs][2])
* We will not extract or output secrets, credentials, or session/authentication tokens, and we will explicitly penalize doing so.

---

## 3. Key Product Principles

### 3.1 Verifiable Rewards

All core rewards are computed deterministically (exact match or normalized match). We avoid LLM judges for correctness.

### 3.2 Procedural Generation With Reproducibility

Every archetype is a parameterized generator that maps a seed into:

* An HTML string.
* A natural-language extraction request.
* A ground-truth expected output.
* Metadata for grading (answer schema, normalization, solvable flag, etc.).

### 3.3 “Judgment” Is First‑Class, Not An Afterthought

Some tasks are intentionally impossible with static BeautifulSoup. The environment must reward correct abstention **only when abstention is objectively warranted**.

---

## 4. Environment Contract

### 4.1 Environment Type and Tool Protocol

This will be a **tool-using environment** built with Verifiers. Tool-using environments in Verifiers use a loop where the model can call stateless tools and the rollout terminates when the model responds without tool calls. ([GitHub][5])

Because sandbox execution requires per-episode state (sandbox handles), we will implement this as a **StatefulToolEnv** or as a specialized sandbox wrapper (SandboxEnv/PythonEnv pattern). Verifiers explicitly documents the StatefulToolEnv state injection approach and notes that SandboxEnv/PythonEnv packages that pattern for Prime sandboxes. ([Prime Intellect Docs][1])

### 4.2 Tool Schema

We provide one primary tool and two optional helper tools.

#### Primary Tool: `run_python`

* **Purpose:** Execute Python code in an isolated sandbox that has BeautifulSoup and the supported parser backends installed.
* **Arguments:**

  * `code: str` — Python code to execute.
* **Environment-provided globals:**

  * `HTML: str` — The HTML content for the episode.
  * `QUERY: str` — The natural-language task request.
  * `CONSTRAINTS: dict` — A dictionary including required output schema and safety rules.
* **Returns:**

  * `stdout: str`
  * `stderr: str`
  * `exit_code: int`
  * `runtime_ms: int` (if available)

#### Optional Tool: `get_task_metadata`

* **Purpose:** Return the grading schema and allowed output forms (but never the ground-truth answer).
* **Returns:** The output JSON schema, `solvable` flag (optional), and allowed limitation reasons (optional).

#### Optional Tool: `lint_json`

* **Purpose:** Validate candidate JSON output against the schema before finalizing.
* **Returns:** Validation errors, if any.

### 4.3 Episode Termination

An episode ends when the assistant returns a final response with **no tool calls**, consistent with ToolEnv semantics. ([GitHub][5])

### 4.4 Final Answer Format

The final assistant message must be a single JSON object. The top-level schema is:

```json
{
  "status": "ok" | "limit",
  "answer": null | string | number | object | array,
  "limit": {
    "reason": string,
    "evidence": string
  }
}
```

Rules:

* If `status` is `"ok"`, then `answer` must match the task’s answer schema and `limit` must be null or omitted.
* If `status` is `"limit"`, then `limit.reason` must be one of the allowed reasons for the task, and `limit.evidence` must be a literal substring or regex-verified signature found in the HTML.

---

## 5. Reward Structure and Anti‑Hacking Rules

### 5.1 Reward Values

* A correct extraction yields **+1.0** reward.
* A correct limitation identification yields **+0.5** reward.
* An incorrect answer yields **0.0** reward.
* A safety violation yields **−0.5** reward (or stronger if needed).

### 5.2 Reward Anti‑Hacking Constraints

To prevent “always abstain” reward hacking, each task instance includes a `solvable: bool` label in its hidden `info`.

* If `solvable = true` and the model outputs `"limit"`, then the reward is **0.0**, and repeated abstention can optionally incur a small penalty.
* If `solvable = false` and the model outputs `"limit"`, then the reward is **+0.5 only if**:

  * The reason is correct for that task, and
  * The evidence is verifiably present in the HTML (string containment or regex match).
* If `solvable = false` and the model outputs `"ok"`, then the reward is **0.0**, because it should not fabricate.

### 5.3 Safety Penalties

The environment assigns a negative reward if the output includes:

* Any value from `<input type="password">` fields.
* Any session/authentication token patterns intentionally placed as traps.
* Any explicitly marked sensitive fields (task metadata will include redaction rules).

---

## 6. MVP Scenario Set: The 50 Best Archetypes

### 6.1 Rationale for MVP Selection

The MVP set is designed to be:

* Unambiguously about **BeautifulSoup mastery**, not generic scraping.
* Rich in the “gotchas” that cause real bugs.
* Diverse enough to produce meaningful RL training signal.
* Practical to implement quickly with deterministic graders.

### 6.2 The Curated Top‑50 List (MVP)

Each scenario below is an archetype that produces unlimited seeded instances.

**Core Extraction (10 archetypes)**

1. The agent extracts visible text from a specific tag while ignoring nested scripts and styles.
2. The agent extracts text by CSS class using BeautifulSoup’s `class_` conventions rather than reserved words.
3. The agent extracts text by unique `id` and safely handles missing elements.
4. The agent extracts a single attribute value such as `href`, `src`, or `data-*` and returns it in the required type.
5. The agent extracts all links as a list of `{text, href}` objects and normalizes whitespace in anchor text.
6. The agent extracts all images as a list of `{src, alt}` objects while handling missing `alt` values deterministically.
7. The agent extracts multiple matching elements with `find_all()` and returns a stable ordering rule (document order).
8. The agent uses multi-criteria selection (tag + class + attribute) and avoids false matches from near-duplicates.
9. The agent uses CSS selectors (`select` / `select_one`) to target a nested element precisely.
10. The agent resolves ambiguous repeated structures by anchoring to surrounding semantic cues (for example, nearest heading).

**Table Parsing (8 archetypes)**
11. The agent converts a simple `<table>` into a list of lists with row/column normalization.
12. The agent converts a headered table into a list of dicts using detected headers as keys.
13. The agent extracts a specific column by header name and handles header whitespace differences.
14. The agent extracts a specific row by matching a row label and returns the associated value.
15. The agent handles missing cells by filling a specified sentinel value (for example, null).
16. The agent handles nested tags inside cells (for example, `<span>` wrappers) without returning markup.
17. The agent handles malformed table structure (for example, missing `<tbody>` or inconsistent row lengths) deterministically.
18. The agent handles a table where `rowspan` or `colspan` affects alignment using a defined expansion rule.

**Traversal and Navigation (8 archetypes)**
19. The agent finds a node and returns a parent element’s attribute (for example, nearest `<a>` around a `<span>`).
20. The agent enumerates direct children and filters only immediate children, not descendants.
21. The agent searches descendants with a conditional filter function (for example, tags containing a substring).
22. The agent navigates to the next sibling matching criteria and ignores whitespace-only siblings.
23. The agent navigates to the previous sibling matching criteria and handles missing sibling cases safely.
24. The agent uses document-order traversal (`find_next`/`find_previous`) for cases where tree structure is misleading.
25. The agent finds the closest ancestor using a CSS selector (“closest”) when required by the task. ([Crummy][6])
26. The agent extracts breadcrumb navigation into a normalized list representing the path.

**Forms and Inputs (6 archetypes)**
27. The agent enumerates all form fields and returns `{name, type, value, required}` objects.
28. The agent extracts `<select>` options including labels and values and preserves display order.
29. The agent extracts form `action` and `method` and returns defaults if omitted.
30. The agent maps `<label>` text to the correct `<input>` using `for` and `id` association rules.
31. The agent detects login forms and explicitly refuses to extract credential values.
32. The agent detects “honeypot” hidden fields and does not include them in the extracted results.

**Malformed HTML and Parser Differences (6 archetypes)**
33. The agent handles unclosed tags and still extracts the correct target text.
34. The agent handles improperly nested tags where DOM repair changes expected tree shape.
35. The agent handles missing quotes on attributes and still extracts required values.
36. The agent handles whitespace-only text nodes by stripping and normalizing output.
37. The agent selects an appropriate parser backend when the task explicitly requires it and explains failures otherwise.
38. The agent parses XML using an XML-capable parser setting when the input is XML-like and avoids HTML assumptions.

**Output Normalization and Encoding (4 archetypes)**
39. The agent returns text with a specified separator and strip behavior and passes strict normalization.
40. The agent normalizes HTML entities and produces the expected Unicode output.
41. The agent handles mojibake-like cases where encoding declarations conflict and returns the specified normalized form.
42. The agent correctly handles multi-valued attributes (for example, `class` as a list) and returns a stable representation.

**Search Filters and Tree Modification (4 archetypes)**
43. The agent uses regex-based tag or attribute matching and avoids overmatching.
44. The agent uses a custom predicate function to match tags when static selectors are insufficient.
45. The agent removes `<script>` and `<style>` nodes before extraction to avoid contamination.
46. The agent decomposes advertisement or navigation blocks and extracts the article body cleanly.

**Error Bait and “Read the Warning” (4 archetypes)**
47. The agent avoids passing reserved words (for example, `class`) incorrectly and uses `class_` or `attrs` safely.
48. The agent correctly handles `.string` returning null and uses a safer extraction method (`get_text`).
49. The agent avoids `NoneType` attribute errors by checking for missing tags before indexing attributes.
50. The agent recognizes “this input looks like a URL, not markup” warnings and treats the string as invalid HTML input rather than fabricating.

---

## 7. Phase 2 Scenario Set: The Remaining Archetypes (Expansion)

### 7.1 Purpose of Phase 2

Phase 2 expands breadth and realism once we have:

* A stable sandbox toolchain.
* A trained model that improves measurably on held-out seeds.
* A proven grader suite that is hard to hack.

### 7.2 Expansion Themes

Phase 2 will add the remaining archetypes from your taxonomy, grouped into themes such as:

* Bot protection and WAF page *detection* (not bypass).
* CSS visual reordering and obfuscation patterns (as detection + abstention unless explicitly solvable from DOM).
* Advanced anti-scraping traps (invisible markers, decoys, randomized class names).
* Internationalization edge cases (RTL pages, CJK, locale date/price formats).
* Hydration shells and Shadow DOM detection.
* Cloudflare-specific signatures and safe reporting (for example, Ray ID extraction) as limitation tasks.
* Pagination traps and loop detection as classification tasks.

Phase 2 is explicitly framed as “classification and abstention” for dynamic or blocked content, which keeps the project aligned with safety and avoids any interpretation of bypass training.

---

## 8. Toolchain and Platform Integration

### 8.1 Environment Packaging and Hub Publishing

We will create a Verifiers environment module that exports `load_environment`, includes a `README.md` for the Hub, and a `pyproject.toml` that pins dependencies and configures default evaluation parameters. ([Prime Intellect Docs][1])

We will publish versions via `prime env push`. ([Prime Intellect Docs][7])

Each push triggers an automated Hub Action that builds the environment in a clean container, installs it, and runs its tests. ([Prime Intellect Docs][4])

### 8.2 Evaluation Interfaces for Bench Mode

We support two evaluation flows:

* `vf-eval` (Verifiers CLI) for local development loops.
* `prime env eval` (Prime CLI) for benchmarking against multiple hosted models and for creating public benchmark context. ([Prime Intellect Docs][8])

### 8.3 Sandbox Requirements

We execute Python in an isolated sandbox, and we disable network access by default to prevent live scraping and to maximize determinism. ([Prime Intellect Docs][2])

---

## 9. Training Demonstration Plan

### 9.1 Baseline (Pre‑RL)

We run a fixed “Bench Mode” evaluation suite on held-out seeds and record:

* Overall pass rate.
* Pass rate by archetype.
* Failure mode breakdown (format errors, extraction errors, limitation misclassification).

### 9.2 RL Training (prime‑rl)

We run prime‑rl training with:

* Training dataset distribution that begins easy and increases difficulty knobs.
* Held-out evaluation every N steps.
* Logged reward curves and evaluation curves.

Prime’s documentation describes using prime‑rl with verifiers environments and shows the three config files (trainer/orchestrator/inference) launched by a single `rl` entrypoint. ([Prime Intellect Docs][3])

### 9.3 Proof of Improvement

We consider the training demonstration successful when the trained model:

* Beats the baseline model on held-out seeds by a meaningful margin (for example, +10–20 points absolute on the MVP bench).
* Demonstrates reduced reward hacking (lower false “limit” rate on solvable tasks).
* Demonstrates improved robustness on malformed HTML and error-bait tasks.

---

## 10. Implementation Phases

### Phase 0: Scaffolding and CI Readiness

* We generate a template environment module using `vf-init` or the Prime template, and we ensure `load_environment` returns a `vf.Environment` object. ([Prime Intellect Docs][1])
* We add a minimal test suite so Hub Actions passes from the first push. ([Prime Intellect Docs][4])

### Phase 1: MVP Generators + Verifiers + Bench Mode (Top‑50)

* We implement the Top‑50 archetype generators with deterministic seeds and stable ground truth.
* We implement the JSON output contract and a strict grader.
* We implement sandbox execution tools and a safe default (no network).
* We implement Bench Mode and baseline evaluations across several models using `prime env eval`. ([Prime Intellect Docs][8])

### Phase 2: RL Training Demonstration

* We run a baseline evaluation.
* We train using prime‑rl and produce training curves and held-out eval curves. ([Prime Intellect Docs][3])

### Phase 3: Phase‑2 Scenario Expansion (Remaining Archetypes)

* We add the remaining archetypes in prioritized batches.
* We keep deterministic grading and expand bench coverage.

### Phase 4: Polish and Submission

* We write full docs, add examples, and publish stable versions to the Hub.
* We finalize the bounty submission with clear evidence artifacts.

---

# Technical Design and Implementation Plan

## A step‑by‑step build plan for an implementing engineer or implementing AI agent

This section is intentionally operational. If someone follows it, they should be able to build, test, benchmark, train, and publish the environment end-to-end.

---

## A. Toolchain Selection

### A.1 Core Libraries

* We will use **Verifiers** as the environment framework, because it standardizes datasets, rollouts, rubrics (reward functions), and tool-based interaction protocols. ([GitHub][5])
* We will use **BeautifulSoup4** (beautifulsoup4) as the evaluated library, with parser backends:

  * `lxml`
  * `html5lib`
  * Python’s built-in `html.parser`
* We will include **Soup Sieve** (soupsieve) explicitly because modern CSS selector support and `closest` integration rely on it. ([Faceless User][9])
* We will use **pytest** for tests.

### A.2 Prime Intellect Platform Tools

* We will use **Prime CLI** to push environments to the Hub (`prime env push`) and to run cross-model evaluations (`prime env eval`). ([Prime Intellect Docs][7])
* We will rely on Hub Actions to validate packaging and tests on push. ([Prime Intellect Docs][4])
* For RL training, we will use **prime‑rl**. ([GitHub][10])

### A.3 Execution Sandbox Strategy

* The environment will support a **sandboxed code execution** tool.
* The default will be **network disabled**, because Prime sandboxes support disabling network access and we want deterministic behavior. ([Prime Intellect Docs][2])
* We will implement a **dual-backend** executor so unit tests can run without Prime credentials:

  * A “local subprocess executor” for CI and unit tests.
  * A “Prime sandbox executor” for the real environment mode and for the bounty requirement.

---

## B. Repository and Package Structure

### B.1 Scaffold the Environment Module

You will create the environment as a standalone package that exports `load_environment`. Verifiers documents the required structure and the `vf-init` generator output. ([Prime Intellect Docs][1])

Recommended structure:

```
environments/beautiful_soup_env/
  beautiful_soup_env.py        # load_environment + env classes
  bs4_env/
    __init__.py
    dataset.py                 # dataset construction, splits
    generators/
      __init__.py
      base.py                  # generator interface + RNG utilities
      mvp_*.py                 # MVP archetype generators
      phase2_*.py              # expansion generators
    grading/
      schema.py                # JSON schema + normalization rules
      normalize.py             # canonicalization functions
      rubric.py                # reward functions
    tools/
      executor.py              # local vs prime sandbox executor
      harness.py               # injected globals + safe runner
  pyproject.toml
  README.md
  tests/
    test_determinism.py
    test_generators_mvp.py
    test_grading.py
    test_executor_local.py
    test_env_smoke.py
```

### B.2 `pyproject.toml` Requirements

Your `pyproject.toml` must:

* Pin or bound dependencies.
* Include `pyproject.toml` in the build output.
* Provide default evaluation parameters for `vf-eval` if desired. Verifiers documents `[tool.verifiers.eval]` defaults and inclusion rules. ([Prime Intellect Docs][1])

You should also include enough metadata for Hub display (name, description, tags, version). ([Prime Intellect Docs][1])

---

## C. System Architecture and Dataflow

### C.1 High-Level Dataflow

1. A dataset builder samples archetypes and seeds to construct a Hugging Face Dataset with a `prompt` column and an `info` column. ([GitHub][5])
2. The environment presents one prompt (HTML + query + output contract) to the model.
3. The model optionally calls tools (primarily `run_python`) to execute parsing code in a sandbox.
4. The model emits a final JSON response.
5. The rubric parses the JSON, validates schema, checks safety constraints, and compares normalized output to ground truth.
6. The rubric produces a scalar reward plus metrics for analysis.

### C.2 Why This Matches Prime’s Expectations

* Verifiers environments are built from datasets + rollout logic + rubrics. ([GitHub][5])
* Tool-based environments terminate when the model stops calling tools, and tools must be stateless/idempotent. ([GitHub][5])
* StatefulToolEnv/SandboxEnv patterns exist for injecting sandbox handles into tool calls. ([Prime Intellect Docs][1])

---

## D. Environment Interaction Design

### D.1 Prompt Template (User Message)

Each episode prompt should include:

* A short instruction that the model must solve using BeautifulSoup code.
* The HTML content, clearly delimited.
* The extraction query.
* The output schema and restrictions, including safety rules and limitation output rules.

A recommended prompt structure:

* A short “Task” header that states the extraction goal.
* A short “Constraints” header that states:

  * The required JSON output schema.
  * That the model should call `run_python` to test code.
  * That the sandbox has network disabled.
  * That it must never output credentials/tokens if present.
* A delimited `HTML` block.
* A delimited `QUERY` line.

### D.2 Tool Calling Strategy

We want the optimal policy to look like:

* Write code, run it, inspect errors, refine code, then output final JSON.
* Avoid calling tools once a correct result is known.
* Avoid unsafe extraction targets even if visible.

---

## E. Dataset Design

### E.1 Row Schema

Each dataset row will contain:

* `prompt: List[ChatMessage]` where the last message is user content describing the task. ([GitHub][5])
* `info: dict` containing:

  * `archetype_id: str`
  * `seed: int`
  * `solvable: bool`
  * `answer_schema: dict` (a JSON schema-like structure for the answer field)
  * `normalization: dict` (flags for whitespace, unicode normalization, ordering rules)
  * `ground_truth: any` (the expected answer, hidden from the model)
  * `limit: {allowed_reasons: [...], signatures: {...}}`
  * `safety: {forbidden_patterns: [...], forbidden_fields: [...]}`

### E.2 Deterministic Procedural Generation

Each generator must:

* Use a stable RNG seeded from `(split, archetype_id, seed)` using SHA‑256 or another stable hash, not Python’s salted `hash()`.
* Build a structured representation of the “true content” first (ground truth), then render to HTML with noise/malformation so the label is not “whatever BeautifulSoup returns.”

### E.3 Train / Eval / Bench Splits

* **Train split** uses a large set of seeds per archetype.
* **Eval split** uses disjoint seeds, never used in training.
* **Bench split** is a fixed, published list of `(archetype_id, seed)` pairs that never changes for a given environment version, so benchmarks remain comparable.

---

## F. Grading and Reward Implementation

### F.1 Output Parsing

Implement a strict JSON parse:

* If JSON parsing fails, assign a strong format penalty and correctness reward of 0.0.
* If JSON parses but violates schema, assign correctness reward of 0.0 and track a “schema_violation” metric.

### F.2 Correctness Grading

Correctness grading uses per-task rules:

* Strings get whitespace normalization and Unicode normalization as configured.
* Lists can be order-sensitive or order-insensitive depending on the task.
* Dict keys are normalized if specified.

### F.3 Limitation Grading

Limitation grading is only available if:

* `solvable == false`,
* `status == "limit"`,
* `reason` is correct and allowed, and
* `evidence` matches a signature present in the HTML.

### F.4 Safety Grading

Implement explicit checks for:

* Password fields and credential extraction.
* Token-like patterns.
* Any task-specific redaction rules.

Safety violations override correctness and produce negative reward.

---

## G. Sandbox Execution Tooling

### G.1 Executor Abstraction

Create an interface:

* `Executor.run(code: str, globals: dict, timeout_s: float) -> ExecResult`

Provide two implementations:

1. `LocalSubprocessExecutor` that runs `python` in a subprocess with a timeout and controlled environment variables.
2. `PrimeSandboxExecutor` that runs code in Prime’s sandbox infrastructure (preferred for the bounty).

### G.2 Network and Determinism

* The sandbox should run with network disabled by default. ([Prime Intellect Docs][2])
* The harness should avoid nondeterministic dependencies (time, randomness) unless explicitly needed.

### G.3 Harness Design

In the executed code context, predefine:

* `HTML`, `QUERY`, and `CONSTRAINTS`.
* A helper function `make_soup(parser: str)` that constructs `BeautifulSoup(HTML, parser)`.

This reduces prompt friction and encourages correct library usage.

---

## H. Environment Implementation (Verifiers)

### H.1 `load_environment` Entry Point

Your environment package must export `load_environment` returning a `vf.Environment` instance. ([Prime Intellect Docs][1])

Arguments recommended:

* `split: str = "train" | "eval" | "bench"`
* `mode: str = "mvp" | "phase2" | "all"`
* `difficulty: str = "easy" | "medium" | "hard" | "mixed"`
* `num_examples: int | None`
* `seed: int`
* `executor_backend: str = "local" | "prime"`
* `network_access: bool = False`

### H.2 Choosing the Base Class

* Use `vf.ToolEnv` if tools are truly stateless and require no per-episode state.
* Use `vf.StatefulToolEnv` if you need to inject sandbox IDs or handles into tool calls. ([Prime Intellect Docs][1])
* Prefer the SandboxEnv/PythonEnv pattern when integrating Prime sandboxes, because it is the standard documented approach. ([Prime Intellect Docs][1])

---

## I. Testing Strategy

### I.1 Unit Tests

You should implement unit tests that run fast and deterministically in Hub Actions.

1. **Determinism tests** must confirm that a generator produces identical outputs for the same seed and different outputs for different seeds.
2. **Schema tests** must confirm that graders reject malformed JSON and accept valid JSON.
3. **Normalization tests** must confirm that whitespace and ordering rules behave as expected.
4. **Safety tests** must confirm that leaking forbidden strings triggers negative reward.
5. **Archetype coverage tests** must confirm that all Top‑50 generators register and can produce at least one instance.

### I.2 Integration Tests

1. **Local executor integration** must run a small set of episodes end-to-end without requiring external credentials.
2. **Prime sandbox integration** should be marked as optional and run only when credentials are present, so Hub Actions does not fail.

### I.3 Environment Smoke Tests

* A smoke test must instantiate the environment via `load_environment`, generate a few examples, and run grading on a known “golden” completion.

### I.4 Hub Actions Compatibility

Every `prime env push` triggers an action that builds and runs tests in a clean container, so your test suite must pass without requiring manual setup. ([Prime Intellect Docs][4])

---

## J. Development Verification Loops

### J.1 “Golden Episode” Debugging

Maintain a small set of golden episodes (one per major category) with:

* Stored HTML.
* Stored expected answer.
* Stored one correct reference solution (BeautifulSoup code).
* Stored one incorrect solution demonstrating a common bug.

This allows rapid regression detection when you modify normalization or reward functions.

### J.2 Instrumentation and Metrics

In addition to scalar reward, log metrics:

* `format_ok`
* `schema_ok`
* `correct_ok`
* `limit_ok`
* `safety_violation`
* `tool_calls_count`
* `runtime_ms` (if available)

These metrics help distinguish whether RL is improving extraction or merely improving formatting.

---

## K. Bench Mode and Cross‑Model Benchmarking

### K.1 Bench Mode Definition

Bench Mode is a fixed dataset split with:

* Frozen seeds.
* Fixed difficulty distribution.
* No training leakage.
* Stable scoring rules.

Bench Mode is used for:

* Baseline measurement before RL.
* Cross-model comparisons.
* Regression testing across environment versions.

### K.2 Running Benchmarks

You can benchmark using Prime’s evaluation tooling:

* `prime env eval` is designed to evaluate an installed environment against a chosen model with configurable rollouts, examples, and environment args. ([Prime Intellect Docs][8])
* Hosted evaluations are available in the Hub UI for convenience and for sharing results. ([Prime Intellect Docs][11])

In your repository, add a script `scripts/bench_models.py` that:

* Accepts a list of models.
* Runs `prime env eval` with `--env-args` pointing to `{"split":"bench","mode":"mvp"}`.
* Saves JSON outputs and aggregates pass rates by archetype.

This produces the “context” evidence reviewers expect: how hard is this environment for common LLMs today.

---

## L. RL Training MVP: When You Should Expect Meaningful Improvement

### L.1 MVP Definition

You have reached MVP when:

* The Top‑50 archetypes exist with deterministic grading.
* Bench Mode exists and produces stable baseline numbers.
* Sandbox execution works (at minimum via local executor, and ideally via Prime sandbox for the real demonstration).
* You can run a short RL job and see a measurable lift on held-out seeds.

### L.2 Minimal Training Evidence Plan

1. Run Bench Mode evaluation on a baseline model and record metrics.
2. Run prime‑rl training using environment `split=train` with a curriculum schedule implemented by sampling difficulty knobs.
3. Re-run Bench Mode evaluation on the trained checkpoint and compare.
4. Report both overall and per-archetype improvements, with special attention to:

   * Reduced format errors.
   * Reduced `.string` / None mistakes.
   * Improved parser choice when specified.
   * Improved abstention correctness on unsolvable tasks.

Prime’s RL documentation shows the three-component architecture (trainer, orchestrator, inference) and the single `rl` entrypoint that takes TOML configs. ([Prime Intellect Docs][3])

### L.3 Performance and Throughput Sanity Checks

prime‑rl supports a `--bench` flag to benchmark the RL stack and identify throughput bottlenecks. ([Prime Intellect Docs][12])
You should run this once early to ensure code execution tooling does not catastrophically slow training.

---

## M. Publishing and Submission Workflow

### M.1 Local Development Commands

* Install editable for development.
* Run `pytest`.
* Run `vf-eval` locally for quick checks.
* Run `prime env eval` for multi-model baselines. ([Prime Intellect Docs][8])

### M.2 Push to Hub

* Increment version in `pyproject.toml`.
* Run `prime env push`. ([Prime Intellect Docs][7])
* Confirm Hub Actions pass (build + test). ([Prime Intellect Docs][4])

---

## N. Phase 2 Implementation Plan (Remaining Archetypes)

Phase 2 should be executed in batches that preserve grading integrity:

1. Add limitation/detection archetypes first (hydration shells, Shadow DOM indicators, cross-origin iframes) because they are deterministic and low-risk.
2. Add anti-scraping trap archetypes next (invisible spans, honeypots, decoys) because they remain static HTML problems.
3. Add WAF signature detection archetypes as abstention-only classification tasks (not bypass).
4. Add internationalization and locale normalization tasks last because they can expand normalization complexity and should not destabilize MVP scoring.

---

## What I would add beyond your original PRD, as “engineering reality” notes

* You should treat any change to normalization or grading rules as a breaking change that can invalidate benchmark comparability, so you should version Bench Mode seeds per environment version and keep a changelog.
* You should keep sandbox integration “optional in tests” but “required in the RL demo,” because Hub Actions must pass without secrets while your bounty submission must prove sandboxed execution.
* You should explicitly document how you prevent label leakage (ground truth from structured data, not derived by BeautifulSoup parsing), because reviewers will look for that.



[1]: https://docs.primeintellect.ai/verifiers/environments "Environments - Prime Intellect Docs"
[2]: https://docs.primeintellect.ai/sandboxes/overview "Sandboxes Overview - Prime Intellect Docs"
[3]: https://primeintellect.mintlify.app/reinforcement-fine-tuning "Reinforcement Fine-Tuning (Beta) - Prime Intellect Docs"
[4]: https://docs.primeintellect.ai/tutorials-environments/environment-actions "Environment Actions - Prime Intellect Docs"
[5]: https://github.com/PrimeIntellect-ai/verifiers "GitHub - PrimeIntellect-ai/verifiers: Environments for LLM Reinforcement Learning"
[6]: https://www.crummy.com/software/BeautifulSoup/bs4/doc/?utm_source=chatgpt.com "Beautiful Soup 4.13.0 documentation - Crummy"
[7]: https://docs.primeintellect.ai/tutorials-environments/create "Create & Upload Environment - Prime Intellect Docs"
[8]: https://docs.primeintellect.ai/tutorials-environments/evaluating "Evaluating Environments - Prime Intellect Docs"
[9]: https://facelessuser.github.io/soupsieve/?utm_source=chatgpt.com "Quick Start - Soup Sieve"
[10]: https://github.com/PrimeIntellect-ai/prime-rl "GitHub - PrimeIntellect-ai/prime-rl: Async RL Training at Scale"
[11]: https://docs.primeintellect.ai/tutorials-environments/hosted-evaluations "Hosted Evaluations - Prime Intellect Docs"
[12]: https://docs.primeintellect.ai/prime-rl/benchmarking "Benchmarking - Prime Intellect Docs"
