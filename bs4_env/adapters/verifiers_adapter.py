from __future__ import annotations

"""Verifiers framework integration for BeautifulSoup RL environment.

This module provides the adapter that wires our environment to Verifiers,
Prime's RL environment framework.

Two modes:
1. If verifiers is installed: Wire to real vf.Environment (TODO: implement)
2. If verifiers is not installed: Return MinimalEnv for local testing
"""

import json
from typing import Any, Callable

from bs4_env.config import EnvConfig, TaskConstraints
from bs4_env.dataset import build_dataset
from bs4_env.grading.rubric import compute_reward
from bs4_env.tools.executor import get_executor
from bs4_env.tools.tool_defs import create_tool_registry


def build_verifiers_environment(config: EnvConfig) -> Any:
    """Build a Verifiers-compatible environment.

    Args:
        config: Environment configuration.

    Returns:
        A vf.Environment if verifiers is installed, otherwise MinimalEnv.
    """
    # Try to import verifiers
    try:
        import verifiers as vf

        return _build_real_verifiers_env(config, vf)
    except ImportError:
        # Verifiers not installed, return minimal env for local testing
        return MinimalEnv(config)


def _build_real_verifiers_env(config: EnvConfig, vf: Any) -> Any:
    """Build a real Verifiers environment.

    Args:
        config: Environment configuration.
        vf: The verifiers module.

    Returns:
        A vf.Environment instance.

    Raises:
        NotImplementedError: This needs to be wired to actual Verifiers APIs.
    """
    # TODO: Implement actual Verifiers integration
    # This is the wiring point for the real environment.
    #
    # The implementation should:
    # 1. Create a dataset using build_dataset(config)
    # 2. Define tools using our tool schemas
    # 3. Wire up the rubric using compute_reward
    # 4. Return a vf.ToolEnv or vf.StatefulToolEnv
    #
    # Example structure (pseudocode):
    #
    # dataset = build_dataset(config)
    #
    # def reward_fn(output: str, info: dict) -> tuple[float, dict]:
    #     return compute_reward(output, info)
    #
    # tools = [run_python_tool, get_metadata_tool, lint_json_tool]
    #
    # return vf.ToolEnv(
    #     dataset=dataset,
    #     tools=tools,
    #     reward_fn=reward_fn,
    # )

    raise NotImplementedError(
        "Real Verifiers integration is not yet implemented.\n"
        "\n"
        "To implement:\n"
        "1. Review Verifiers documentation for ToolEnv/StatefulToolEnv patterns\n"
        "2. Wire build_dataset() to create the HF dataset\n"
        "3. Wire tool schemas from bs4_env.tools.tool_defs\n"
        "4. Wire compute_reward() as the reward function\n"
        "5. Handle sandbox state if using StatefulToolEnv\n"
        "\n"
        "For now, use MinimalEnv for local development:\n"
        "  env = MinimalEnv(config)"
    )


class MinimalEnv:
    """Minimal environment for local testing without Verifiers.

    This provides just enough interface to test the full pipeline locally.
    """

    def __init__(self, config: EnvConfig):
        """Initialize the minimal environment.

        Args:
            config: Environment configuration.
        """
        self.config = config
        self._dataset = None
        self._executor = None
        self._current_idx = 0

    @property
    def dataset(self):
        """Lazily build and cache the dataset."""
        if self._dataset is None:
            self._dataset = build_dataset(self.config)
        return self._dataset

    @property
    def executor(self):
        """Lazily create and cache the executor."""
        if self._executor is None:
            self._executor = get_executor(
                backend=self.config.executor_backend,
                max_output_chars=self.config.max_output_chars,
            )
        return self._executor

    def __len__(self) -> int:
        """Return number of examples in dataset."""
        return len(self.dataset)

    def __iter__(self):
        """Iterate over examples."""
        self._current_idx = 0
        return self

    def __next__(self) -> dict:
        """Get next example."""
        if self._current_idx >= len(self.dataset):
            raise StopIteration
        example = self.get_example(self._current_idx)
        self._current_idx += 1
        return example

    def get_example(self, idx: int = 0) -> dict:
        """Get a specific example.

        Args:
            idx: Example index.

        Returns:
            Dictionary with 'prompt', 'info', 'html', 'query'.
        """
        row = self.dataset[idx]
        info = json.loads(row["info"]) if isinstance(row["info"], str) else row["info"]

        # Extract HTML and query from the prompt
        # (they're embedded in the user message)
        prompt = row["prompt"]
        user_msg = next(m for m in prompt if m["role"] == "user")

        # Parse HTML from the prompt
        html = self._extract_html_from_prompt(user_msg["content"])
        query = self._extract_query_from_prompt(user_msg["content"])

        return {
            "prompt": prompt,
            "info": info,
            "html": html,
            "query": query,
            "idx": idx,
        }

    def _extract_html_from_prompt(self, content: str) -> str:
        """Extract HTML from prompt content."""
        # HTML is between ```html and ```
        start = content.find("```html")
        if start < 0:
            return ""
        start += 7  # len("```html")
        end = content.find("```", start)
        if end < 0:
            return content[start:].strip()
        return content[start:end].strip()

    def _extract_query_from_prompt(self, content: str) -> str:
        """Extract query from prompt content."""
        # Query is after "## Task" and before the next ##
        start = content.find("## Task")
        if start < 0:
            return ""
        start += 7  # len("## Task")
        end = content.find("##", start)
        if end < 0:
            return content[start:].strip()
        return content[start:end].strip()

    def create_tool_registry(self, example: dict) -> Any:
        """Create a tool registry for an example.

        Args:
            example: Example from get_example().

        Returns:
            ToolRegistry with run_python and optional tools configured.
        """
        from bs4_env.tools.tool_defs import create_tool_registry

        constraints = TaskConstraints(
            output_schema=example["info"].get("answer_schema", {}),
            allowed_limit_reasons=example["info"].get("limit_info", {}).get(
                "allowed_reasons", []
            ),
        )

        return create_tool_registry(
            executor=self.executor,
            html=example["html"],
            query=example["query"],
            constraints=constraints.__dict__,
            task_info=example["info"],
            timeout_s=self.config.timeout_s,
        )

    def grade(self, output: str, example: dict) -> tuple[float, dict]:
        """Grade a model output.

        Args:
            output: The raw model output string.
            example: The example dictionary.

        Returns:
            Tuple of (reward, metrics).
        """
        return compute_reward(
            raw_output=output,
            task_info=example["info"],
            html=example["html"],
        )

    def run_episode(
        self,
        agent_fn: Callable[[list[dict], Any], tuple[str, list[dict]]],
        idx: int = 0,
    ) -> dict:
        """Run a complete episode with an agent.

        Args:
            agent_fn: Function that takes (messages, tool_registry) and returns
                (final_output, tool_call_history).
            idx: Example index to run.

        Returns:
            Episode result dictionary.
        """
        example = self.get_example(idx)
        tool_registry = self.create_tool_registry(example)

        # Run agent
        final_output, tool_history = agent_fn(example["prompt"], tool_registry)

        # Grade output
        reward, metrics = self.grade(final_output, example)

        return {
            "idx": idx,
            "archetype_id": example["info"].get("archetype_id"),
            "seed": example["info"].get("seed"),
            "reward": reward,
            "metrics": metrics,
            "final_output": final_output,
            "tool_calls": len(tool_history),
            "ground_truth": example["info"].get("ground_truth"),
        }


def create_simple_agent(model_name: str = "gpt-4") -> Callable:
    """Create a simple agent for testing.

    This is a placeholder that would integrate with an LLM API.

    Args:
        model_name: The model to use.

    Returns:
        Agent function compatible with MinimalEnv.run_episode().
    """

    def agent_fn(
        messages: list[dict],
        tool_registry: Any,
    ) -> tuple[str, list[dict]]:
        """Simple agent that just returns a placeholder response."""
        # In a real implementation, this would:
        # 1. Send messages to the LLM
        # 2. Handle tool calls in a loop
        # 3. Return final output when model stops calling tools

        # Placeholder: just return an error response
        return '{"status": "ok", "answer": "placeholder"}', []

    return agent_fn
