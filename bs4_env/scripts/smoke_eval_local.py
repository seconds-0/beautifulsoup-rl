#!/usr/bin/env python3
"""End-to-end local smoke test for the BeautifulSoup RL environment.

This script demonstrates the full pipeline:
1. Load environment and get an example
2. Create tool registry for the example
3. Run code through the executor
4. Grade the output

Usage:
    python -m bs4_env.scripts.smoke_eval_local
    python -m bs4_env.scripts.smoke_eval_local --num 5
"""

from __future__ import annotations

import argparse
import json
import sys

from beautiful_soup_env import load_environment


def create_solution_code(info: dict) -> str:
    """Generate solution code based on the task type.

    This is a simple heuristic solver for demonstration purposes.
    A real agent would use an LLM to generate this code.
    """
    archetype = info.get("archetype_id", "")

    # For limitation tasks, we can't solve - must detect and abstain
    if not info.get("solvable", True):
        limit_info = info.get("limit_info", {})
        allowed_reasons = limit_info.get("allowed_reasons", ["unknown"])

        # Detect limitation type based on allowed reasons
        is_image_task = any(
            r in allowed_reasons for r in ["image_text", "ocr_required", "text_in_image"]
        )
        is_js_task = any(
            r in allowed_reasons for r in ["js_required", "javascript_required", "dynamic_content"]
        )

        return f"""
import json

# This is a limitation task - detect and abstain
soup = make_soup()
evidence = None
reason = None

is_image_task = {is_image_task}
is_js_task = {is_js_task}

if is_image_task:
    # Look for image evidence
    img = soup.find("img")
    if img:
        evidence = str(img)[:50]
        reason = "image_text"

if not evidence and is_js_task:
    # Look for JS evidence
    scripts = soup.find_all("script")
    for script in scripts:
        if script.string:
            text = script.string
            if "getElementById" in text or "innerHTML" in text or "ReactDOM" in text:
                evidence = text[:50]
                reason = "js_required"
                break

if evidence and reason:
    result = {{"status": "limit", "answer": None, "limit": {{"reason": reason, "evidence": evidence}}}}
else:
    # Fallback using first allowed reason
    allowed = {allowed_reasons}
    fallback_reason = allowed[0] if allowed else "unknown"
    result = {{"status": "limit", "answer": None, "limit": {{"reason": fallback_reason, "evidence": "<"}}}}

print(json.dumps(result))
"""

    # For solvable tasks, generate extraction code based on archetype
    if "extract_text_by_id" in archetype:
        return """
import json
import re
soup = make_soup()

# Parse target ID from QUERY
match = re.search(r'id="([^"]+)"', QUERY)
target_id = match.group(1) if match else None

if target_id:
    element = soup.find(id=target_id)
    if element:
        answer = element.get_text(strip=True)
    else:
        answer = None
else:
    answer = None
result = {"status": "ok", "answer": answer}
print(json.dumps(result))
"""

    if "extract_text_by_class" in archetype:
        return """
import json
import re
soup = make_soup()

# Parse target class from QUERY
match = re.search(r'class="([^"]+)"', QUERY)
target_class = match.group(1) if match else None

if target_class:
    element = soup.find(class_=target_class)
    if element:
        answer = element.get_text(strip=True)
    else:
        answer = None
else:
    answer = None
result = {"status": "ok", "answer": answer}
print(json.dumps(result))
"""

    if "table_list_of_dicts" in archetype:
        return """
import json
soup = make_soup()
table = soup.find("table")
if not table:
    print(json.dumps({"status": "ok", "answer": []}))
else:
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for tr in table.find_all("tr")[1:]:  # Skip header row
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    result = {"status": "ok", "answer": rows}
    print(json.dumps(result))
"""

    if "table_list_of_lists" in archetype:
        return """
import json
soup = make_soup()
table = soup.find("table")
if not table:
    print(json.dumps({"status": "ok", "answer": []}))
else:
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)
    result = {"status": "ok", "answer": rows}
    print(json.dumps(result))
"""

    if "string_returns_none" in archetype:
        return """
import json, re
soup = make_soup()
match = re.search(r'id="([^"]+)"', QUERY)
target_id = match.group(1) if match else None
element = soup.find(id=target_id) if target_id else None
if element:
    # Use get_text with separator - .string returns None for multiple children
    # separator=' ' preserves spacing between child elements
    answer = element.get_text(separator=' ', strip=True)
else:
    answer = None
result = {"status": "ok", "answer": answer}
print(json.dumps(result))
"""

    if "none_attribute_error" in archetype or "class_reserved_word" in archetype:
        return """
import json, re
soup = make_soup()
match = re.search(r'class="([^"]+)"', QUERY)
target_class = match.group(1) if match else None
# Use class_ parameter, not class (reserved word)
element = soup.find(class_=target_class) if target_class else None
if element:
    answer = element.get_text(strip=True)
else:
    answer = None
result = {"status": "ok", "answer": answer}
print(json.dumps(result))
"""

    # Generic fallback
    return """
import json
soup = make_soup()
# Generic extraction attempt
body = soup.find("body") or soup
answer = body.get_text(strip=True)[:200] if body else None
result = {"status": "ok", "answer": answer}
print(json.dumps(result))
"""


def run_smoke_test(env, idx: int) -> dict:
    """Run a single smoke test on an example."""
    example = env.get_example(idx)
    info = example["info"]

    # Create tool registry
    tool_registry = env.create_tool_registry(example)

    # Generate solution code
    code = create_solution_code(info)

    # Execute the code
    exec_result_str = tool_registry.call("run_python", {"code": code})

    # Parse the tool response
    try:
        exec_result = json.loads(exec_result_str)
    except json.JSONDecodeError:
        exec_result = {"stdout": exec_result_str, "stderr": ""}

    # Parse the output
    stdout = exec_result.get("stdout", "")
    stderr = exec_result.get("stderr", "")

    # Grade the output
    reward, metrics = env.grade(stdout, example)

    return {
        "idx": idx,
        "archetype_id": info.get("archetype_id"),
        "solvable": info.get("solvable"),
        "seed": info.get("seed"),
        "reward": reward,
        "metrics": metrics,
        "stdout": stdout[:500],
        "stderr": stderr[:200] if stderr else None,
        "ground_truth": info.get("ground_truth"),
    }


def main():
    parser = argparse.ArgumentParser(description="Local smoke test for BS4 environment")
    parser.add_argument("--split", default="train", choices=["train", "eval", "bench"])
    parser.add_argument("--mode", default="mvp", choices=["mvp", "phase2", "all"])
    parser.add_argument("--num", type=int, default=10, help="Number of examples to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    print(f"Loading environment: split={args.split}, mode={args.mode}")
    env = load_environment(split=args.split, mode=args.mode)
    print(f"Dataset size: {len(env)}")

    results = []
    total_reward = 0.0

    num_to_test = min(args.num, len(env))

    for i in range(num_to_test):
        print(f"\n--- Example {i} ---")
        result = run_smoke_test(env, i)
        results.append(result)
        total_reward += result["reward"]

        print(f"Archetype: {result['archetype_id']}")
        print(f"Solvable: {result['solvable']}")
        print(f"Reward: {result['reward']}")

        if args.verbose:
            print(f"Ground truth: {result['ground_truth']}")
            print(f"Output: {result['stdout'][:200]}")
            if result["stderr"]:
                print(f"Stderr: {result['stderr']}")

        if result["reward"] < 0.5:
            print(f"  ⚠️  Low reward - check metrics: {result['metrics']}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total examples: {num_to_test}")
    print(f"Average reward: {total_reward / num_to_test:.3f}")

    # Breakdown by archetype
    archetype_rewards = {}
    for r in results:
        arch = r["archetype_id"]
        if arch not in archetype_rewards:
            archetype_rewards[arch] = []
        archetype_rewards[arch].append(r["reward"])

    print("\nBy archetype:")
    for arch, rewards in sorted(archetype_rewards.items()):
        avg = sum(rewards) / len(rewards)
        print(f"  {arch}: {avg:.3f} ({len(rewards)} examples)")

    # Failures
    failures = [r for r in results if r["reward"] < 0.5]
    if failures:
        print(f"\n⚠️  {len(failures)} examples with low reward:")
        for f in failures[:5]:
            print(f"  - {f['archetype_id']} (idx={f['idx']}): {f['reward']}")
            if f.get("metrics", {}).get("parse_error"):
                print(f"    Parse error: {f['metrics']['parse_error']}")

    return 0 if total_reward / num_to_test > 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
