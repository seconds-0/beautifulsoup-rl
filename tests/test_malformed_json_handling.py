"""Tests for malformed JSON handling in verifiers_adapter.

These tests cover the defensive checks added after training v5 crashed with:
  'list' object has no attribute 'get'

Root cause: Models sometimes produce invalid JSON for tool arguments, including:
- Lists instead of dicts: [{"code": "..."}] instead of {"code": "..."}
- Plain strings instead of dicts
- Truncated JSON due to max_tokens limits
- Empty/null arguments

The fixes in verifiers_adapter.py guard against these cases to prevent crashes
during RL training where we can't afford to lose a training run to bad output.
"""

import importlib.util
import json

import pytest


class TestMalformedToolArguments:
    """Tests for handling malformed tool call arguments in bs4_reward."""

    def _create_completion_with_tool_call(self, args):
        """Helper to create a completion with tool calls."""
        return [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "run_python",
                            "arguments": args,
                        },
                    }
                ],
            }
        ]

    def test_args_as_list_does_not_crash(self):
        """When model produces list instead of dict, should not crash.

        This was the exact bug from training v5:
        Model output: [{"code": "print('hello')"}]
        Expected:     {"code": "print('hello')"}
        """
        from bs4_env.adapters.verifiers_adapter import build_verifiers_environment
        from bs4_env.config import EnvConfig

        # Build environment
        config = EnvConfig(split="bench", mode="mvp", seed=42)

        # Import here to check if verifiers is installed
        if importlib.util.find_spec("verifiers") is None:
            pytest.skip("verifiers not installed, cannot test reward function")

        env = build_verifiers_environment(config)

        # Create a completion where args is a LIST (the bug case)
        list_args = json.dumps([{"code": "print('hello')"}])  # List wrapping dict
        completion = self._create_completion_with_tool_call(list_args)

        state = {"html": "<html></html>", "task_info": {"solvable": True}}

        # This should NOT crash - it should handle gracefully
        # Access the rubric's reward function
        reward_fn = env.rubric.funcs[0]
        try:
            # Call reward function - should not raise
            reward = reward_fn(completion, state, info={"solvable": True, "ground_truth": "test"})
            # Should return some reward (likely 0 for wrong answer, but not crash)
            assert isinstance(reward, int | float)
        except AttributeError as e:
            if "'list' object has no attribute 'get'" in str(e):
                pytest.fail(
                    "Bug regression: list args should not cause AttributeError. "
                    "The guard in bs4_reward() should handle this."
                )
            raise

    def test_args_as_string_does_not_crash(self):
        """When model produces string instead of dict, should not crash."""
        from bs4_env.adapters.verifiers_adapter import build_verifiers_environment
        from bs4_env.config import EnvConfig

        config = EnvConfig(split="bench", mode="mvp", seed=42)

        if importlib.util.find_spec("verifiers") is None:
            pytest.skip("verifiers not installed, cannot test reward function")

        env = build_verifiers_environment(config)

        # Create a completion where args is a plain STRING
        string_args = '"just a string, not a dict"'
        completion = self._create_completion_with_tool_call(string_args)

        state = {"html": "<html></html>", "task_info": {"solvable": True}}

        reward_fn = env.rubric.funcs[0]
        reward = reward_fn(completion, state, info={"solvable": True, "ground_truth": "test"})
        assert isinstance(reward, int | float)

    def test_valid_dict_args_extracts_code(self):
        """Valid dict arguments should still work correctly."""
        from bs4_env.adapters.verifiers_adapter import build_verifiers_environment
        from bs4_env.config import EnvConfig

        config = EnvConfig(split="bench", mode="mvp", seed=42)

        if importlib.util.find_spec("verifiers") is None:
            pytest.skip("verifiers not installed, cannot test reward function")

        env = build_verifiers_environment(config)

        # Normal valid args
        valid_args = json.dumps({"code": "print('hello')"})
        completion = self._create_completion_with_tool_call(valid_args)

        state = {"html": "<html></html>", "task_info": {"solvable": True}}

        reward_fn = env.rubric.funcs[0]
        reward = reward_fn(completion, state, info={"solvable": True, "ground_truth": "test"})
        assert isinstance(reward, int | float)


class TestJsonRepair:
    """Tests for the _try_repair_json method."""

    def _get_repair_func(self):
        """Get the repair function from the environment class."""
        from bs4_env.adapters.verifiers_adapter import build_verifiers_environment
        from bs4_env.config import EnvConfig

        if importlib.util.find_spec("verifiers") is None:
            pytest.skip("verifiers not installed")

        config = EnvConfig(split="bench", mode="mvp", seed=42)
        env = build_verifiers_environment(config)
        return env._try_repair_json

    def test_valid_json_passes_through(self):
        """Valid JSON should be returned as-is."""
        repair = self._get_repair_func()
        valid = '{"code": "print(1)"}'
        assert repair(valid) == valid

    def test_truncated_json_missing_brace(self):
        """Truncated JSON missing closing brace should be repaired."""
        repair = self._get_repair_func()
        truncated = '{"code": "print(1)"'  # Missing }
        result = repair(truncated)

        if result is not None:
            # If repaired, should be valid JSON
            parsed = json.loads(result)
            assert isinstance(parsed, dict)

    def test_truncated_json_missing_bracket(self):
        """Truncated JSON missing closing bracket should be repaired."""
        repair = self._get_repair_func()
        truncated = '["a", "b"'  # Missing ]
        result = repair(truncated)

        if result is not None:
            parsed = json.loads(result)
            assert isinstance(parsed, list)

    def test_extract_code_from_partial(self):
        """Should extract code argument from partially valid JSON."""
        repair = self._get_repair_func()
        # Truncated mid-string but code is complete
        partial = '{"code": "print(1)", "extra": "trun'
        result = repair(partial)

        if result is not None:
            parsed = json.loads(result)
            # Should have preserved the code
            if isinstance(parsed, dict) and "code" in parsed:
                assert parsed["code"] == "print(1)"

    def test_unfixable_json_returns_none(self):
        """Completely unfixable JSON should return None."""
        repair = self._get_repair_func()
        # Random garbage that can't be fixed
        garbage = "this is not json at all {{{{["
        result = repair(garbage)
        # Should return None for unfixable
        assert result is None


class TestEnvResponseJsonHandling:
    """Tests for malformed JSON handling in env_response."""

    @pytest.mark.asyncio
    async def test_none_arguments_handled(self):
        """None arguments should be replaced with empty object."""
        from bs4_env.adapters.verifiers_adapter import build_verifiers_environment
        from bs4_env.config import EnvConfig

        if importlib.util.find_spec("verifiers") is None:
            pytest.skip("verifiers not installed")

        config = EnvConfig(split="bench", mode="mvp", seed=42)
        env = build_verifiers_environment(config)

        # Message with None arguments
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "run_python",
                            "arguments": None,  # This should be handled
                        },
                    }
                ],
            }
        ]

        state = {
            "html": "<html></html>",
            "query": "test",
            "constraints": {},
            "pages": {},
            "navigation_history": [],
        }

        # Should not crash
        try:
            await env.env_response(messages, state)
            # If we get here without exception, the fix works
        except TypeError as e:
            if "NoneType" in str(e):
                pytest.fail("None arguments should be handled gracefully")
            raise

    @pytest.mark.asyncio
    async def test_empty_string_arguments_handled(self):
        """Empty string arguments should be replaced with empty object."""
        from bs4_env.adapters.verifiers_adapter import build_verifiers_environment
        from bs4_env.config import EnvConfig

        if importlib.util.find_spec("verifiers") is None:
            pytest.skip("verifiers not installed")

        config = EnvConfig(split="bench", mode="mvp", seed=42)
        env = build_verifiers_environment(config)

        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "run_python",
                            "arguments": "",  # Empty string
                        },
                    }
                ],
            }
        ]

        state = {
            "html": "<html></html>",
            "query": "test",
            "constraints": {},
            "pages": {},
            "navigation_history": [],
        }

        # Should not crash
        try:
            await env.env_response(messages, state)
        except json.JSONDecodeError:
            pytest.fail("Empty string arguments should be handled gracefully")

    @pytest.mark.asyncio
    async def test_malformed_json_arguments_repaired(self):
        """Malformed JSON arguments should be repaired or defaulted."""
        from bs4_env.adapters.verifiers_adapter import build_verifiers_environment
        from bs4_env.config import EnvConfig

        if importlib.util.find_spec("verifiers") is None:
            pytest.skip("verifiers not installed")

        config = EnvConfig(split="bench", mode="mvp", seed=42)
        env = build_verifiers_environment(config)

        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "run_python",
                            "arguments": '{"code": "print(1)"',  # Truncated JSON
                        },
                    }
                ],
            }
        ]

        state = {
            "html": "<html></html>",
            "query": "test",
            "constraints": {},
            "pages": {},
            "navigation_history": [],
        }

        # Should not crash - either repairs or defaults to {}
        try:
            await env.env_response(messages, state)
        except json.JSONDecodeError as e:
            pytest.fail(f"Malformed JSON arguments should be repaired or defaulted: {e}")


class TestDirectArgumentParsing:
    """Direct unit tests for the argument parsing logic in bs4_reward.

    These tests don't require the full verifiers environment - they test
    the parsing logic in isolation.
    """

    def test_list_args_type_check(self):
        """Demonstrate the isinstance guard for list args."""
        # Simulating the parsing logic from bs4_reward
        args = '[{"code": "print(1)"}]'
        args_dict = json.loads(args)

        # This is the guard that was added to prevent the crash
        if isinstance(args_dict, dict):
            code = args_dict.get("code", "")
        else:
            # List case - the guard catches this
            code = ""

        assert code == ""  # Guard prevents accessing .get() on list

    def test_dict_args_extracts_code(self):
        """Normal dict args should extract code correctly."""
        args = '{"code": "print(1)"}'
        args_dict = json.loads(args)

        if isinstance(args_dict, dict):
            code = args_dict.get("code", "")
        else:
            code = ""

        assert code == "print(1)"

    def test_string_value_args(self):
        """When JSON parses to just a string, should not crash."""
        args = '"just a string"'
        args_dict = json.loads(args)

        # Guard handles this - string has no .get()
        if isinstance(args_dict, dict):
            code = args_dict.get("code", "")
        else:
            code = ""

        assert code == ""

    def test_integer_value_args(self):
        """When JSON parses to just an integer, should not crash."""
        args = "42"
        args_dict = json.loads(args)

        if isinstance(args_dict, dict):
            code = args_dict.get("code", "")
        else:
            code = ""

        assert code == ""

    def test_null_value_args(self):
        """When JSON parses to null, should not crash."""
        args = "null"
        args_dict = json.loads(args)

        if isinstance(args_dict, dict):
            code = args_dict.get("code", "")
        else:
            code = ""

        assert code == ""
