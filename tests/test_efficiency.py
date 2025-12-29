"""Tests for efficiency-based reward adjustments.

These tests verify that:
1. Efficiency multiplier calculates correctly at boundaries
2. Hard cutoff works (>10 calls = 0.0)
3. Wrong answers aren't affected by efficiency
4. Integration with compute_reward works
"""

import pytest

from bs4_env.grading.rubric import (
    EFFICIENCY_FLOOR,
    EFFICIENCY_PENALTY_PER_CALL,
    MAX_TOOL_CALLS,
    compute_efficiency_multiplier,
    compute_reward,
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
