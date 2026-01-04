"""Tests for efficiency-based reward adjustments.

These tests verify that:
1. Efficiency multiplier calculates correctly at boundaries
2. Hard cutoff works (>10 calls = 0.0)
3. Wrong answers aren't affected by efficiency
4. Integration with compute_reward works
5. Per-task max_tool_calls constraints are honored
"""

import pytest

from bs4_env.grading.rubric import (
    DEFAULT_MAX_TOOL_CALLS,
    EFFICIENCY_FLOOR,
    EFFICIENCY_PENALTY_PER_CALL,
    MAX_TOOL_CALLS,
    compute_efficiency_multiplier,
    compute_reward,
    get_max_tool_calls,
)


class TestEfficiencyMultiplier:
    """Tests for compute_efficiency_multiplier function."""

    def test_one_call_full_credit(self):
        """Single tool call gets full credit."""
        assert compute_efficiency_multiplier(1) == 1.0

    def test_zero_calls_full_credit(self):
        """Zero tool calls gets full credit."""
        assert compute_efficiency_multiplier(0) == 1.0

    def test_two_calls(self):
        """Two calls gets 0.9."""
        assert compute_efficiency_multiplier(2) == 0.9

    def test_three_calls(self):
        """Three calls gets 0.8."""
        assert compute_efficiency_multiplier(3) == 0.8

    def test_linear_decay(self):
        """Verify linear decay pattern."""
        # Each additional call after first costs 0.1
        for n in range(1, 9):
            expected = max(EFFICIENCY_FLOOR, 1.0 - EFFICIENCY_PENALTY_PER_CALL * (n - 1))
            assert compute_efficiency_multiplier(n) == pytest.approx(expected)

    def test_floor_at_nine_calls(self):
        """Nine calls hits the floor."""
        assert compute_efficiency_multiplier(9) == EFFICIENCY_FLOOR

    def test_floor_at_ten_calls(self):
        """Ten calls is still at floor (not cutoff yet)."""
        assert compute_efficiency_multiplier(10) == EFFICIENCY_FLOOR

    def test_hard_cutoff_eleven_calls(self):
        """Eleven calls exceeds limit - returns 0.0."""
        assert compute_efficiency_multiplier(11) == 0.0

    def test_hard_cutoff_fifty_calls(self):
        """Fifty calls is way over - returns 0.0."""
        assert compute_efficiency_multiplier(50) == 0.0

    def test_hard_cutoff_boundary(self):
        """Verify exact boundary of MAX_TOOL_CALLS."""
        assert compute_efficiency_multiplier(MAX_TOOL_CALLS) == EFFICIENCY_FLOOR
        assert compute_efficiency_multiplier(MAX_TOOL_CALLS + 1) == 0.0


class TestEfficiencyInReward:
    """Tests for efficiency integration with compute_reward."""

    @pytest.fixture
    def correct_output(self):
        """A correct JSON output."""
        return '{"status": "ok", "answer": "test value"}'

    @pytest.fixture
    def wrong_output(self):
        """A wrong JSON output."""
        return '{"status": "ok", "answer": "wrong value"}'

    @pytest.fixture
    def task_info(self):
        """Task info with ground truth."""
        return {
            "ground_truth": "test value",
            "solvable": True,
            "answer_schema": {"type": "string"},
        }

    def test_correct_one_call_full_reward(self, correct_output, task_info):
        """Correct answer with 1 call gets full reward."""
        reward, metrics = compute_reward(correct_output, task_info, tool_call_count=1)
        assert reward == 1.0
        assert metrics["efficiency_multiplier"] == 1.0
        assert metrics["tool_calls"] == 1

    def test_correct_two_calls_reduced(self, correct_output, task_info):
        """Correct answer with 2 calls gets 0.9 reward."""
        reward, metrics = compute_reward(correct_output, task_info, tool_call_count=2)
        assert reward == pytest.approx(0.9)
        assert metrics["efficiency_multiplier"] == 0.9

    def test_correct_five_calls_reduced(self, correct_output, task_info):
        """Correct answer with 5 calls gets 0.6 reward."""
        reward, metrics = compute_reward(correct_output, task_info, tool_call_count=5)
        assert reward == pytest.approx(0.6)
        assert metrics["efficiency_multiplier"] == 0.6

    def test_correct_ten_calls_floor(self, correct_output, task_info):
        """Correct answer with 10 calls gets floor reward."""
        reward, metrics = compute_reward(correct_output, task_info, tool_call_count=10)
        assert reward == pytest.approx(EFFICIENCY_FLOOR)
        assert metrics["efficiency_multiplier"] == EFFICIENCY_FLOOR

    def test_correct_over_limit_fails(self, correct_output, task_info):
        """Correct answer with >10 calls is treated as failure."""
        reward, metrics = compute_reward(correct_output, task_info, tool_call_count=11)
        assert reward == 0.0
        assert metrics["efficiency_multiplier"] == 0.0
        assert metrics["correct"] is False
        assert any("Exceeded max tool calls" in e for e in metrics["errors"])

    def test_wrong_answer_unaffected(self, wrong_output, task_info):
        """Wrong answer stays at 0.0 regardless of efficiency."""
        # Wrong with 1 call
        reward1, _ = compute_reward(wrong_output, task_info, tool_call_count=1)
        assert reward1 == 0.0

        # Wrong with 5 calls - still 0.0, not negative
        reward5, _ = compute_reward(wrong_output, task_info, tool_call_count=5)
        assert reward5 == 0.0

    def test_no_tool_count_skips_efficiency(self, correct_output, task_info):
        """When tool_call_count is None, efficiency is not applied."""
        reward, metrics = compute_reward(correct_output, task_info, tool_call_count=None)
        assert reward == 1.0
        assert metrics["efficiency_multiplier"] is None
        assert metrics["tool_calls"] is None

    def test_metrics_track_tool_calls(self, correct_output, task_info):
        """Metrics include tool call count."""
        _, metrics = compute_reward(correct_output, task_info, tool_call_count=7)
        assert metrics["tool_calls"] == 7


class TestEfficiencyEdgeCases:
    """Edge case tests for efficiency calculation."""

    def test_negative_calls_treated_as_zero(self):
        """Negative call count is treated as zero/one (full credit)."""
        # Implementation detail: negative values should not crash
        assert compute_efficiency_multiplier(-1) == 1.0

    def test_very_large_call_count(self):
        """Very large call counts return 0.0."""
        assert compute_efficiency_multiplier(1000) == 0.0
        assert compute_efficiency_multiplier(999999) == 0.0


class TestRawVsWeightedToolCounts:
    """Tests for raw vs weighted tool count semantics."""

    @pytest.fixture
    def correct_output(self):
        return '{"status": "ok", "answer": "test"}'

    @pytest.fixture
    def task_info(self):
        return {
            "ground_truth": "test",
            "solvable": True,
            "answer_schema": {"type": "string"},
        }

    def test_raw_count_used_for_hard_cap(self, correct_output, task_info):
        """Hard cap uses raw count (11 raw calls = zero reward)."""
        # 11 raw calls but only 5 weighted (imagine many navigate calls)
        reward, metrics = compute_reward(
            correct_output,
            task_info,
            tool_call_count=5.0,  # Weighted: low
            tool_call_count_raw=11,  # Raw: over limit
        )
        assert reward == 0.0
        assert metrics["errors"]
        assert "Exceeded max tool calls" in str(metrics["errors"])

    def test_weighted_count_used_for_soft_penalty(self, correct_output, task_info):
        """Soft efficiency penalty uses weighted count."""
        # 8 raw calls but only 3.0 weighted (many navigate calls)
        reward, metrics = compute_reward(
            correct_output,
            task_info,
            tool_call_count=3.0,  # Weighted
            tool_call_count_raw=8,  # Raw
        )
        # Should use weighted count (3.0) for efficiency: 1.0 - 0.1 * 2 = 0.8
        assert reward == pytest.approx(0.8)
        assert metrics["efficiency_multiplier"] == pytest.approx(0.8)

    def test_falls_back_to_weighted_when_raw_not_provided(self, correct_output, task_info):
        """When tool_call_count_raw is None, uses tool_call_count for hard cap."""
        # 12 weighted calls, no raw count provided
        reward, metrics = compute_reward(
            correct_output,
            task_info,
            tool_call_count=12.0,
            tool_call_count_raw=None,
        )
        # Should use weighted count for hard cap too
        assert reward == 0.0
        assert "Exceeded max tool calls" in str(metrics["errors"])

    def test_metrics_track_both_counts(self, correct_output, task_info):
        """Metrics include both raw and weighted counts."""
        _, metrics = compute_reward(
            correct_output,
            task_info,
            tool_call_count=3.0,
            tool_call_count_raw=5,
        )
        assert metrics["tool_calls"] == 3.0
        assert metrics["tool_calls_raw"] == 5


class TestGetMaxToolCalls:
    """Tests for get_max_tool_calls helper function."""

    def test_returns_default_for_none(self):
        """None task_info returns default."""
        assert get_max_tool_calls(None) == DEFAULT_MAX_TOOL_CALLS

    def test_returns_default_for_empty_dict(self):
        """Empty task_info returns default."""
        assert get_max_tool_calls({}) == DEFAULT_MAX_TOOL_CALLS

    def test_returns_default_when_no_constraints(self):
        """Task without constraints returns default."""
        task_info = {"metadata": {"some_key": "value"}}
        assert get_max_tool_calls(task_info) == DEFAULT_MAX_TOOL_CALLS

    def test_reads_from_metadata_constraints(self):
        """Reads max_tool_calls from metadata.constraints."""
        task_info = {
            "metadata": {"constraints": {"max_tool_calls": 15}}
        }
        assert get_max_tool_calls(task_info) == 15

    def test_handles_json_serialized_metadata(self):
        """Handles metadata as JSON string (from dataset)."""
        import json
        task_info = {
            "metadata": json.dumps({"constraints": {"max_tool_calls": 20}})
        }
        assert get_max_tool_calls(task_info) == 20

    def test_returns_default_for_invalid_json_metadata(self):
        """Returns default when metadata is invalid JSON."""
        task_info = {"metadata": "not valid json {"}
        assert get_max_tool_calls(task_info) == DEFAULT_MAX_TOOL_CALLS


class TestArchetypeAwareEfficiency:
    """Tests for per-task max_tool_calls constraints."""

    @pytest.fixture
    def correct_output(self):
        return '{"status": "ok", "answer": "test"}'

    @pytest.fixture
    def base_task_info(self):
        return {
            "ground_truth": "test",
            "solvable": True,
            "answer_schema": {"type": "string"},
        }

    def test_default_limit_applied(self, correct_output, base_task_info):
        """Default limit is applied when no constraints specified."""
        # 11 calls exceeds default of 10
        reward, metrics = compute_reward(
            correct_output,
            base_task_info,
            tool_call_count=11,
            tool_call_count_raw=11,
        )
        assert reward == 0.0
        assert metrics["max_tool_calls"] == DEFAULT_MAX_TOOL_CALLS
        assert "Exceeded max tool calls" in str(metrics["errors"])

    def test_custom_limit_allows_more_calls(self, correct_output, base_task_info):
        """Custom higher limit allows more tool calls."""
        base_task_info["metadata"] = {"constraints": {"max_tool_calls": 15}}

        # 12 calls is within custom limit of 15
        reward, metrics = compute_reward(
            correct_output,
            base_task_info,
            tool_call_count=12,
            tool_call_count_raw=12,
        )
        assert reward > 0  # Should not be zero
        assert metrics["max_tool_calls"] == 15
        assert "Exceeded max tool calls" not in str(metrics.get("errors", []))

    def test_custom_limit_enforced(self, correct_output, base_task_info):
        """Custom limit is enforced when exceeded."""
        base_task_info["metadata"] = {"constraints": {"max_tool_calls": 15}}

        # 16 calls exceeds custom limit of 15
        reward, metrics = compute_reward(
            correct_output,
            base_task_info,
            tool_call_count=16,
            tool_call_count_raw=16,
        )
        assert reward == 0.0
        assert metrics["max_tool_calls"] == 15
        assert "16 > 15" in str(metrics["errors"])

    def test_stricter_custom_limit(self, correct_output, base_task_info):
        """Custom stricter limit is enforced."""
        base_task_info["metadata"] = {"constraints": {"max_tool_calls": 5}}

        # 6 calls exceeds stricter limit of 5
        reward, metrics = compute_reward(
            correct_output,
            base_task_info,
            tool_call_count=6,
            tool_call_count_raw=6,
        )
        assert reward == 0.0
        assert metrics["max_tool_calls"] == 5

    def test_metrics_include_max_tool_calls(self, correct_output, base_task_info):
        """Metrics always include the max_tool_calls limit used."""
        _, metrics = compute_reward(correct_output, base_task_info, tool_call_count=1)
        assert "max_tool_calls" in metrics
        assert metrics["max_tool_calls"] == DEFAULT_MAX_TOOL_CALLS

    def test_json_serialized_metadata_constraints(self, correct_output, base_task_info):
        """Handles JSON-serialized metadata from datasets."""
        import json
        base_task_info["metadata"] = json.dumps({"constraints": {"max_tool_calls": 20}})

        # 15 calls is within limit of 20
        reward, metrics = compute_reward(
            correct_output,
            base_task_info,
            tool_call_count=15,
            tool_call_count_raw=15,
        )
        assert reward > 0
        assert metrics["max_tool_calls"] == 20
