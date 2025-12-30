from __future__ import annotations

"""Verifiers framework integration for BeautifulSoup RL environment.

This module provides the adapter that wires our environment to Verifiers,
Prime's RL environment framework.

Two modes:
1. If verifiers is installed: Wire to real vf.Environment (TODO: implement)
2. If verifiers is not installed: Return MinimalEnv for local testing
"""

import contextlib
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
    from bs4_env.tools.harness import (
        NAVIGATE_SUCCESS_MARKER,
        build_tool_response,
    )

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

    # Define the navigate tool for multi-step tasks
    # Note: pages_json is injected via update_tool_args and skipped from schema
    def navigate(
        href: str,
        pages_json: str = "{}",
    ) -> str:
        """Navigate to a different page by following a link.

        Use this tool to follow links found in the HTML. After navigating,
        the HTML global in run_python will contain the new page content.

        Args:
            href: The href attribute of the link to follow (e.g., '/products/123').

        Returns:
            Success message if navigation succeeded, error message otherwise.
        """
        pages = json.loads(pages_json) if pages_json else {}

        if not href:
            return "Error: No href provided"

        if not pages:
            return "Error: This task does not support navigation. Use run_python to parse the current page."

        # Normalize href for lookup
        normalized = _normalize_href(href, pages)

        if normalized not in pages:
            return f"Error: Page '{href}' not found. Check the HTML for valid links."

        # Return structured marker + user message
        # env_response parses the marker to update state["html"]
        return (
            f"{NAVIGATE_SUCCESS_MARKER}{normalized}\n\n"
            f"Successfully navigated. Use run_python to extract data from the new page."
        )

    def _normalize_href(href: str, pages: dict) -> str:
        """Normalize an href for lookup in pages dict."""
        href = href.strip()

        # Try exact match first
        if href in pages:
            return href

        # Try with/without leading slash
        if href.startswith("/"):
            without_slash = href[1:]
            if without_slash in pages:
                return without_slash
        else:
            with_slash = "/" + href
            if with_slash in pages:
                return with_slash

        # Try without query string
        if "?" in href:
            base = href.split("?")[0]
            if base in pages:
                return base

        # Try without fragment
        if "#" in href:
            base = href.split("#")[0]
            if base in pages:
                return base

        return href

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

        # Count tool calls and extract code samples from completion history
        tool_call_count = 0
        code_samples = []
        if isinstance(completion, list):
            for msg in completion:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        tool_call_count += len(tool_calls)
                        # Extract code from run_python tool calls
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                func = tc.get("function", {})
                                if func.get("name") == "run_python":
                                    args = func.get("arguments", "{}")
                                    try:
                                        args_dict = (
                                            json.loads(args) if isinstance(args, str) else args
                                        )
                                        code = args_dict.get("code", "")
                                        if code:
                                            code_samples.append(code)
                                    except json.JSONDecodeError:
                                        pass

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
            code_samples=code_samples if code_samples else None,
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

            # HTML and query are stored in info (not in prompt)
            html = info.get("html", "")
            query = info.get("query", "")

            # Build constraints
            constraints = {
                "output_schema": info.get("answer_schema", {}),
                "allowed_limit_reasons": info.get("limit_info", {}).get("allowed_reasons", []),
            }

            # Store pages for multi-step tasks
            pages = info.get("pages", {})

            # Store in state
            state["html"] = html
            state["query"] = query
            state["constraints"] = constraints
            state["task_info"] = info
            state["pages"] = pages
            state["navigation_history"] = []

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
            elif tool_name == "navigate":
                return {
                    **tool_args,
                    "pages_json": json.dumps(state.get("pages", {})),
                }
            return tool_args

        async def env_response(self, messages: list, state: dict, **kwargs):
            """Process environment response after tool execution.

            Detects successful navigate calls and updates state["html"] accordingly.
            """
            # Call parent implementation first
            response, state = await super().env_response(messages, state, **kwargs)

            # Check for navigate success in recent messages
            pages = state.get("pages", {})
            if pages:
                for msg in reversed(messages):
                    if not isinstance(msg, dict):
                        continue

                    # Look for tool results (role: "tool" messages contain tool output)
                    if msg.get("role") == "tool":
                        content = msg.get("content", "")
                        # Use structured marker for robust detection
                        if isinstance(content, str) and content.startswith(NAVIGATE_SUCCESS_MARKER):
                            # Extract href from marker: "NAVIGATE_OK:href\n..."
                            marker_content = content[len(NAVIGATE_SUCCESS_MARKER) :]
                            normalized_href = marker_content.split("\n")[0].strip()
                            if normalized_href in pages:
                                state["html"] = pages[normalized_href]
                                state["navigation_history"].append(normalized_href)
                            # Only process the most recent navigate
                            break

            return response, state

        def __len__(self) -> int:
            """Return number of examples in dataset."""
            return len(self.dataset)

        def get_example(self, idx: int = 0) -> dict:
            """Get a specific example for eval scripts.

            Args:
                idx: Example index.

            Returns:
                Dictionary with 'prompt', 'info', 'html', 'query'.
            """
            row = self.dataset[idx]
            info = json.loads(row["info"]) if isinstance(row["info"], str) else row["info"]

            return {
                "prompt": row["prompt"],
                "info": info,
                "html": info.get("html", ""),
                "query": info.get("query", ""),
                "idx": idx,
            }

        def create_tool_registry(self, example: dict) -> Any:
            """Create a tool registry for an example (for eval scripts).

            Args:
                example: Example from get_example().

            Returns:
                ToolRegistry with run_python and optional tools configured.
            """
            from bs4_env.config import TaskConstraints
            from bs4_env.tools.tool_defs import create_tool_registry

            constraints = TaskConstraints(
                output_schema=example["info"].get("answer_schema", {}),
                allowed_limit_reasons=example["info"].get("limit_info", {}).get("allowed_reasons", []),
            )

            pages = example["info"].get("pages", {})

            return create_tool_registry(
                executor=get_executor(
                    backend="local",
                    max_output_chars=10000,
                ),
                html=example["html"],
                query=example["query"],
                constraints=constraints.__dict__,
                task_info=example["info"],
                timeout_s=30.0,
                pages=pages if pages else None,
            )

        def grade(
            self,
            output: str,
            example: dict,
            tool_call_count: int | None = None,
            code_samples: list[str] | None = None,
        ) -> tuple[float, dict]:
            """Grade a model output (for eval scripts).

            Args:
                output: The raw model output string.
                example: The example dictionary.
                tool_call_count: Number of tool calls made.
                code_samples: List of code strings executed.

            Returns:
                Tuple of (reward, metrics).
            """
            from bs4_env.grading.rubric import compute_reward

            return compute_reward(
                raw_output=output,
                task_info=example["info"],
                html=example["html"],
                tool_call_count=tool_call_count,
                code_samples=code_samples,
            )

    # Create environment and add tools
    env = BeautifulSoupEnv(
        dataset=dataset,
        tools=[],  # Add tools after construction
        max_turns=10,
        rubric=rubric,
    )

    # Add the run_python tool, skipping state-injected args from schema
    env.add_tool(run_python, args_to_skip=["html", "query", "constraints_json"])

    # Add navigate tool, skipping state-injected args from schema
    # Note: navigate is always available, but only works if pages exist in the task
    env.add_tool(navigate, args_to_skip=["pages_json"])

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

        # HTML and query are stored in info (not in prompt)
        html = self._extract_html_from_info(info)
        query = self._extract_query_from_info(info)

        return {
            "prompt": row["prompt"],
            "info": info,
            "html": html,
            "query": query,
            "idx": idx,
        }

    def _extract_html_from_info(self, info: dict) -> str:
        """Extract HTML from info dict."""
        return info.get("html", "")

    def _extract_query_from_info(self, info: dict) -> str:
        """Extract query from info dict."""
        return info.get("query", "")

    def create_tool_registry(self, example: dict) -> Any:
        """Create a tool registry for an example.

        Args:
            example: Example from get_example().

        Returns:
            ToolRegistry with run_python and optional tools configured.
            For multi-step tasks, includes navigate tool.
        """

        constraints = TaskConstraints(
            output_schema=example["info"].get("answer_schema", {}),
            allowed_limit_reasons=example["info"].get("limit_info", {}).get("allowed_reasons", []),
        )

        # Get pages for multi-step tasks
        pages = example["info"].get("pages", {})

        return create_tool_registry(
            executor=self.executor,
            html=example["html"],
            query=example["query"],
            constraints=constraints.__dict__,
            task_info=example["info"],
            timeout_s=self.config.timeout_s,
            pages=pages if pages else None,
        )

    def grade(
        self,
        output: str,
        example: dict,
        tool_call_count: int | None = None,
        code_samples: list[str] | None = None,
    ) -> tuple[float, dict]:
        """Grade a model output.

        Args:
            output: The raw model output string.
            example: The example dictionary.
            tool_call_count: Number of tool calls made (for efficiency penalty).
            code_samples: List of code strings executed (for BS4 usage penalty).

        Returns:
            Tuple of (reward, metrics).
        """
        return compute_reward(
            raw_output=output,
            task_info=example["info"],
            html=example["html"],
            tool_call_count=tool_call_count,
            code_samples=code_samples,
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

        # Extract code samples from tool history for BS4 penalty
        code_samples = []
        for call in tool_history:
            if isinstance(call, dict):
                # Direct code key (our format)
                if "code" in call:
                    code_samples.append(call["code"])
                # OpenAI-style tool call format
                elif "arguments" in call:
                    args = call.get("arguments", {})
                    if isinstance(args, str):
                        with contextlib.suppress(json.JSONDecodeError, TypeError):
                            args = json.loads(args)
                    if isinstance(args, dict) and "code" in args:
                        code_samples.append(args["code"])

        # Grade output with tool call metrics for efficiency and BS4 penalties
        reward, metrics = self.grade(
            final_output,
            example,
            tool_call_count=len(tool_history),
            code_samples=code_samples if code_samples else None,
        )

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
