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
        A vf.StatefulToolEnv instance.
    """
    from bs4_env.tools.executor import get_executor
    from bs4_env.tools.harness import build_runner_script, build_tool_response

    # Build the HuggingFace dataset
    dataset = build_dataset(config)

    # Create executor for code execution
    executor = get_executor(
        backend=config.executor_backend,
        max_output_chars=config.max_output_chars,
    )

    # Define the run_python tool
    # Note: html, query, constraints are injected via update_tool_args and skipped from schema
    def run_python(
        code: str,
        html: str = "",
        query: str = "",
        constraints_json: str = "{}",
    ) -> str:
        """Execute Python code with BeautifulSoup to parse the HTML.

        The following globals are available:
        - HTML: The HTML content to parse
        - QUERY: The task description
        - CONSTRAINTS: Task constraints and output schema
        - make_soup(parser='html.parser'): Helper to create BeautifulSoup object

        Args:
            code: Python code to execute. Must print JSON output.

        Returns:
            Execution result with stdout, stderr, and exit code.
        """
        constraints = json.loads(constraints_json) if constraints_json else {}

        globals_dict = {
            "HTML": html,
            "QUERY": query,
            "CONSTRAINTS": constraints,
        }

        result = executor.run(code, globals_dict, timeout_s=config.timeout_s)
        return build_tool_response(result.to_dict())

    # Define reward function
    def bs4_reward(completion, state: dict, info: dict = None, **kwargs) -> float:
        """Compute reward for BeautifulSoup task completion.

        Args:
            completion: The model's completion - either a string or list of message dicts.
            state: Task state containing html and other context.
            info: Task info dict containing ground_truth, archetype_id, etc.
            **kwargs: Additional arguments passed by verifiers (prompt, answer, etc.).

        Returns:
            Reward value: 1.0 correct (efficiency-adjusted), 0.5 valid limitation,
            0.0 wrong or too many tool calls, -0.5 safety violation.
        """
        # Get task_info from info kwarg (preferred) or state
        task_info = info or state.get("info", {}) or state.get("task_info", {})
        html = state.get("html", "")

        # Count tool calls from completion history for efficiency penalty
        tool_call_count = 0
        if isinstance(completion, list):
            for msg in completion:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        tool_call_count += len(tool_calls)

        # Extract string content from completion
        if isinstance(completion, list):
            # completion is a list of message dicts - get last assistant message content
            raw_output = ""
            for msg in reversed(completion):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        raw_output = content
                        break
            if not raw_output and completion:
                # Fallback: get content from last message
                last_msg = completion[-1]
                if isinstance(last_msg, dict):
                    raw_output = str(last_msg.get("content", ""))
                else:
                    raw_output = str(last_msg)
        else:
            raw_output = str(completion)

        reward, metrics = compute_reward(
            raw_output=raw_output,
            task_info=task_info,
            html=html,
            tool_call_count=tool_call_count if tool_call_count > 0 else None,
        )
        return reward

    # Create the rubric
    rubric = vf.Rubric(funcs=[bs4_reward], weights=[1.0])

    # Create StatefulToolEnv subclass to handle per-task state
    class BeautifulSoupEnv(vf.StatefulToolEnv):
        """BeautifulSoup RL environment with per-task state."""

        async def setup_state(self, state: dict, **kwargs) -> dict:
            """Set up task-specific state from the dataset row."""
            state = await super().setup_state(state, **kwargs)

            # Get the current example's info
            row = kwargs.get("row", {})
            info = row.get("info", {})
            if isinstance(info, str):
                info = json.loads(info)

            # Extract HTML and query from prompt
            prompt = row.get("prompt", [])
            user_msg = next((m for m in prompt if m.get("role") == "user"), {})
            content = user_msg.get("content", "")

            # Parse HTML from prompt
            html = ""
            start = content.find("```html")
            if start >= 0:
                start += 7
                end = content.find("```", start)
                html = content[start:end].strip() if end > start else content[start:].strip()

            # Parse query from prompt
            query = ""
            start = content.find("## Task")
            if start >= 0:
                start += 7
                end = content.find("##", start)
                query = content[start:end].strip() if end > start else content[start:].strip()

            # Build constraints
            constraints = {
                "output_schema": info.get("answer_schema", {}),
                "allowed_limit_reasons": info.get("limit_info", {}).get("allowed_reasons", []),
            }

            # Store in state
            state["html"] = html
            state["query"] = query
            state["constraints"] = constraints
            state["task_info"] = info

            return state

        def update_tool_args(
            self, tool_name: str, tool_args: dict, messages: Any, state: dict, **kwargs
        ) -> dict:
            """Inject state into tool arguments."""
            if tool_name == "run_python":
                return {
                    **tool_args,
                    "html": state.get("html", ""),
                    "query": state.get("query", ""),
                    "constraints_json": json.dumps(state.get("constraints", {})),
                }
            return tool_args

    # Create environment and add tool with skipped args
    env = BeautifulSoupEnv(
        dataset=dataset,
        tools=[],  # Add tools after construction
        max_turns=10,
        rubric=rubric,
    )
    # Add the run_python tool, skipping state-injected args from schema
    env.add_tool(run_python, args_to_skip=["html", "query", "constraints_json"])

    return env


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
