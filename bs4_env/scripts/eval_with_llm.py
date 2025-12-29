#!/usr/bin/env python3
"""Evaluate BeautifulSoup environment with real LLMs via OpenRouter.

This script tests the environment using proper OpenAI function calling API,
matching exactly how Prime/Verifiers runs evaluations.

**Critical**: We test how we measure. This uses the same tool schemas and
function calling flow that Prime uses in production.

Usage:
    # Set your OpenRouter API key
    export OPENROUTER_API_KEY=sk-or-...

    # Run with defaults (gpt-4o-mini, 20 examples)
    uv run python -m bs4_env.scripts.eval_with_llm

    # Run with specific model
    uv run python -m bs4_env.scripts.eval_with_llm --model deepseek/deepseek-r1-0528-qwen3-8b

    # Run more examples
    uv run python -m bs4_env.scripts.eval_with_llm --num 100 --model qwen/qwen3-8b

Available cheap models on OpenRouter:
    - openai/gpt-4o-mini (~$0.00015/1K input, $0.0006/1K output)
    - deepseek/deepseek-r1-0528-qwen3-8b (~$0.02/1M tokens)
    - qwen/qwen3-8b (~$0.028/1M tokens)
    - anthropic/claude-3-haiku (~$0.00025/1K input)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from openai import OpenAI

from beautiful_soup_env import load_environment
from bs4_env.tools.harness import RUN_PYTHON_TOOL_SCHEMA


# OpenRouter base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def create_openrouter_client() -> OpenAI:
    """Create OpenAI client configured for OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        # Try regular OpenAI key as fallback
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key and not api_key.startswith("sk-or-"):
            # Using actual OpenAI, not OpenRouter
            return OpenAI(api_key=api_key)

    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable")
        sys.exit(1)

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )


def get_tools() -> list[dict]:
    """Get tool definitions in OpenAI format.

    Uses the same schema defined in harness.py for consistency with Prime.
    """
    return [{
        "type": "function",
        "function": RUN_PYTHON_TOOL_SCHEMA
    }]


def run_llm_agent(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tool_registry: Any,
    max_turns: int = 10,
) -> tuple[str, list[dict], dict]:
    """Run an LLM agent on a task using proper function calling.

    This matches how Prime/Verifiers executes agents:
    1. Send messages with tools parameter
    2. If model returns tool_calls, execute them
    3. Send tool results back as tool role messages
    4. Repeat until model stops calling tools
    5. Return the final non-tool response

    Args:
        client: OpenAI client.
        model: Model name (e.g., "openai/gpt-4o-mini").
        messages: Initial messages (system + user).
        tool_registry: Tool registry for executing code.
        max_turns: Maximum conversation turns.

    Returns:
        Tuple of (final_output, tool_history, stats).
    """
    tools = get_tools()
    conversation = list(messages)  # Copy to avoid mutation
    tool_history = []
    stats = {"input_tokens": 0, "output_tokens": 0, "turns": 0, "tool_calls": 0}
    final_output = ""

    for turn in range(max_turns):
        stats["turns"] += 1

        try:
            response = client.chat.completions.create(
                model=model,
                messages=conversation,
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=4000,
                # Note: response_format not used - conflicts with tools on some providers
            )
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)}), tool_history, stats

        # Track token usage
        if response.usage:
            stats["input_tokens"] += response.usage.prompt_tokens
            stats["output_tokens"] += response.usage.completion_tokens

        message = response.choices[0].message

        # Check if model wants to call tools
        if message.tool_calls:
            # Add assistant message with tool calls to conversation
            conversation.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Execute each tool call
            for tool_call in message.tool_calls:
                stats["tool_calls"] += 1
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                # Execute the tool
                if func_name == "run_python":
                    result = tool_registry.call("run_python", func_args)
                else:
                    result = json.dumps({"error": f"Unknown tool: {func_name}"})

                tool_history.append({
                    "tool": func_name,
                    "args": func_args,
                    "result": result[:1000]  # Truncate for history
                })

                # Add tool result to conversation
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            # Model is done calling tools - this is the final response
            final_output = message.content or ""
            break

    return final_output, tool_history, stats


def run_evaluation(
    model: str,
    num_examples: int,
    split: str = "bench",
    mode: str = "mvp",
    verbose: bool = False,
) -> dict:
    """Run full evaluation.

    Args:
        model: Model name.
        num_examples: Number of examples to evaluate.
        split: Dataset split.
        mode: Archetype mode.
        verbose: Print detailed output.

    Returns:
        Evaluation results dict.
    """
    print(f"Loading environment: split={split}, mode={mode}")
    env = load_environment(split=split, mode=mode)
    print(f"Dataset size: {len(env)}")

    client = create_openrouter_client()
    print(f"Model: {model}")
    print(f"Using OpenAI function calling API (production-match)")
    print(f"Evaluating {num_examples} examples...\n")

    results = []
    total_reward = 0.0
    total_tokens = {"input": 0, "output": 0}
    total_tool_calls = 0

    num_to_test = min(num_examples, len(env))

    for i in range(num_to_test):
        example = env.get_example(i)
        info = example["info"]

        print(f"[{i+1}/{num_to_test}] {info.get('archetype_id')} (seed={info.get('seed')})...", end=" ", flush=True)

        start_time = time.time()

        # Create tool registry for this example
        tool_registry = env.create_tool_registry(example)

        # Run the LLM agent with proper function calling
        final_output, tool_history, stats = run_llm_agent(
            client=client,
            model=model,
            messages=example["prompt"],
            tool_registry=tool_registry,
        )

        elapsed = time.time() - start_time

        # Grade the output (include tool call count for efficiency penalty)
        reward, metrics = env.grade(final_output, example, tool_call_count=stats["tool_calls"])

        results.append({
            "idx": i,
            "archetype_id": info.get("archetype_id"),
            "seed": info.get("seed"),
            "solvable": info.get("solvable"),
            "reward": reward,
            "metrics": metrics,
            "turns": stats["turns"],
            "tool_calls": stats["tool_calls"],
            "input_tokens": stats["input_tokens"],
            "output_tokens": stats["output_tokens"],
            "elapsed_s": elapsed,
            "final_output": final_output[:500] if verbose else None,
            "ground_truth": info.get("ground_truth") if verbose else None,
        })

        total_reward += reward
        total_tokens["input"] += stats["input_tokens"]
        total_tokens["output"] += stats["output_tokens"]
        total_tool_calls += stats["tool_calls"]

        status = "PASS" if reward >= 0.5 else "FAIL"
        print(f"{status} (r={reward:.2f}, {elapsed:.1f}s, {stats['tool_calls']} calls)")

        if verbose and reward < 0.5:
            print(f"  Ground truth: {info.get('ground_truth')}")
            print(f"  Output: {final_output[:200]}")
            if metrics.get("parse_error"):
                print(f"  Parse error: {metrics['parse_error']}")

    # Summary
    avg_reward = total_reward / num_to_test

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Examples: {num_to_test}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Pass rate (r>=0.5): {sum(1 for r in results if r['reward'] >= 0.5) / num_to_test:.1%}")
    print(f"Perfect rate (r=1.0): {sum(1 for r in results if r['reward'] == 1.0) / num_to_test:.1%}")
    print(f"Total tool calls: {total_tool_calls}")
    print(f"Total tokens: {total_tokens['input']:,} input, {total_tokens['output']:,} output")

    # Breakdown by archetype
    archetype_results = {}
    for r in results:
        arch = r["archetype_id"]
        if arch not in archetype_results:
            archetype_results[arch] = []
        archetype_results[arch].append(r["reward"])

    print(f"\nBy archetype:")
    for arch, rewards in sorted(archetype_results.items()):
        avg = sum(rewards) / len(rewards)
        perfect = sum(1 for r in rewards if r == 1.0)
        print(f"  {arch}: {avg:.3f} avg, {perfect}/{len(rewards)} perfect")

    # Failures
    failures = [r for r in results if r["reward"] < 0.5]
    if failures:
        print(f"\nFailed examples ({len(failures)}):")
        for f in failures[:10]:
            print(f"  - idx={f['idx']} {f['archetype_id']}: r={f['reward']:.2f}")

    return {
        "model": model,
        "num_examples": num_to_test,
        "avg_reward": avg_reward,
        "pass_rate": sum(1 for r in results if r["reward"] >= 0.5) / num_to_test,
        "perfect_rate": sum(1 for r in results if r["reward"] == 1.0) / num_to_test,
        "total_tool_calls": total_tool_calls,
        "total_tokens": total_tokens,
        "by_archetype": {k: sum(v)/len(v) for k, v in archetype_results.items()},
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BS4 environment with LLMs (production-match)")
    parser.add_argument("--model", "-m", default="openai/gpt-4o-mini",
                       help="Model name (OpenRouter format, e.g., openai/gpt-4o-mini)")
    parser.add_argument("--num", "-n", type=int, default=20,
                       help="Number of examples to evaluate")
    parser.add_argument("--split", default="bench", choices=["train", "eval", "bench"])
    parser.add_argument("--mode", default="mvp", choices=["mvp", "phase2", "all"])
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output for failures")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Save results to JSON file")
    args = parser.parse_args()

    results = run_evaluation(
        model=args.model,
        num_examples=args.num,
        split=args.split,
        mode=args.mode,
        verbose=args.verbose,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    # Return exit code based on performance
    return 0 if results["avg_reward"] >= 0.3 else 1


if __name__ == "__main__":
    sys.exit(main())
