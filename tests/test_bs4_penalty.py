"""Tests for BS4 usage penalty.

This tests the gradient toward BeautifulSoup usage:
- Solutions using BS4 get full credit
- Solutions using pure regex/string get a penalty (but still pass)
"""

import pytest

from bs4_env.grading.rubric import (
    check_bs4_usage,
    compute_bs4_penalty,
    compute_reward,
    BS4_USAGE_PENALTY,
    REWARD_CORRECT,
)


class TestBS4UsageDetection:
    """Tests for detecting BS4 usage in code."""

    def test_detects_beautifulsoup_import(self):
        """BeautifulSoup class usage is detected."""
        code = 'soup = BeautifulSoup(HTML, "html.parser")'
        assert check_bs4_usage([code]) is True

    def test_detects_make_soup_helper(self):
        """make_soup() helper usage is detected."""
        code = "soup = make_soup()"
        assert check_bs4_usage([code]) is True

    def test_detects_find_all_method(self):
        """soup.find_all() method is detected (BS4-specific)."""
        code = 'elements = soup.find_all("a")'
        assert check_bs4_usage([code]) is True

    def test_find_alone_not_detected(self):
        """Plain .find() is not detected (could be string method)."""
        # .find() alone is ambiguous - could be string.find()
        code = 'result = soup.find("div")'
        # This should NOT be detected without BeautifulSoup/make_soup
        assert check_bs4_usage([code]) is False

    def test_find_with_beautifulsoup_detected(self):
        """soup.find() with BeautifulSoup import is detected."""
        code = 'soup = BeautifulSoup(HTML, "html.parser")\nresult = soup.find("div")'
        assert check_bs4_usage([code]) is True

    def test_detects_select_method(self):
        """soup.select() CSS selector method is detected."""
        code = 'elements = soup.select("div.target > span")'
        assert check_bs4_usage([code]) is True

    def test_detects_get_text_method(self):
        """element.get_text() method is detected."""
        code = 'text = element.get_text(strip=True)'
        assert check_bs4_usage([code]) is True

    def test_detects_sibling_navigation(self):
        """Sibling navigation is detected."""
        code = "sibling = element.next_sibling"
        assert check_bs4_usage([code]) is True

    def test_pure_regex_not_detected(self):
        """Pure regex solution is not detected as BS4."""
        code = '''
import re
match = re.search(r'<div class="target">(.*?)</div>', HTML)
answer = match.group(1) if match else None
print(json.dumps({"status": "ok", "answer": answer}))
'''
        assert check_bs4_usage([code]) is False

    def test_pure_string_not_detected(self):
        """Pure string manipulation is not detected as BS4."""
        code = '''
start = HTML.find('<div class="target">')
end = HTML.find('</div>', start)
answer = HTML[start:end].split(">")[1]
print(json.dumps({"status": "ok", "answer": answer}))
'''
        assert check_bs4_usage([code]) is False

    def test_empty_code_list_returns_true(self):
        """Empty code list means no penalty (can't detect usage)."""
        assert check_bs4_usage([]) is True

    def test_multiple_code_samples_any_bs4(self):
        """If any code sample uses BS4, returns True."""
        code1 = 're.search(r"pattern", HTML)'  # No BS4
        code2 = 'soup = make_soup()'  # Uses BS4
        assert check_bs4_usage([code1, code2]) is True

    def test_multiple_code_samples_no_bs4(self):
        """If no code sample uses BS4, returns False."""
        code1 = 're.search(r"pattern", HTML)'
        code2 = 'HTML.find("<div>")'
        assert check_bs4_usage([code1, code2]) is False


class TestBS4Penalty:
    """Tests for BS4 penalty computation."""

    def test_bs4_used_no_penalty(self):
        """Using BS4 results in no penalty."""
        code = 'soup = make_soup()'
        penalty, used = compute_bs4_penalty([code])
        assert penalty == 0.0
        assert used is True

    def test_no_bs4_penalty_applied(self):
        """Not using BS4 results in penalty."""
        code = 're.search(r"pattern", HTML)'
        penalty, used = compute_bs4_penalty([code])
        assert penalty == BS4_USAGE_PENALTY
        assert used is False

    def test_none_samples_no_penalty(self):
        """None code samples means no penalty (can't detect)."""
        penalty, used = compute_bs4_penalty(None)
        assert penalty == 0.0
        assert used is True


class TestBS4PenaltyInReward:
    """Tests for BS4 penalty integrated with compute_reward."""

    @pytest.fixture
    def task_info(self):
        """Basic task info for testing."""
        return {
            "ground_truth": "expected_answer",
            "solvable": True,
            "answer_schema": {"type": "string"},
        }

    def test_bs4_solution_full_reward(self, task_info):
        """BS4 solution gets full reward."""
        output = '{"status": "ok", "answer": "expected_answer"}'
        code = 'soup = make_soup()\nresult = soup.find("div").text'

        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            code_samples=[code],
        )

        assert reward == REWARD_CORRECT
        assert metrics["bs4_used"] is True
        assert metrics["bs4_penalty"] == 0.0
        assert metrics["correct"] is True

    def test_regex_solution_penalized(self, task_info):
        """Regex solution gets penalized but still passes."""
        output = '{"status": "ok", "answer": "expected_answer"}'
        code = 're.search(r"<div>(.*?)</div>", HTML).group(1)'

        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            code_samples=[code],
        )

        expected_reward = REWARD_CORRECT - BS4_USAGE_PENALTY
        assert reward == pytest.approx(expected_reward)
        assert metrics["bs4_used"] is False
        assert metrics["bs4_penalty"] == BS4_USAGE_PENALTY
        assert metrics["correct"] is True  # Still correct!
        assert len(metrics["warnings"]) > 0
        assert "BS4 not detected" in metrics["warnings"][0]

    def test_wrong_answer_no_bs4_penalty(self, task_info):
        """Wrong answers don't get BS4 penalty (base reward is 0)."""
        output = '{"status": "ok", "answer": "wrong_answer"}'
        code = 're.search(r"<div>(.*?)</div>", HTML).group(1)'

        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            code_samples=[code],
        )

        assert reward == 0.0  # Wrong answer, no additional penalty
        assert metrics["correct"] is False

    def test_no_code_samples_no_penalty(self, task_info):
        """If no code samples provided, no penalty applied."""
        output = '{"status": "ok", "answer": "expected_answer"}'

        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            code_samples=None,
        )

        assert reward == REWARD_CORRECT
        assert metrics["bs4_used"] is None
        assert metrics["bs4_penalty"] is None

    def test_limitation_task_no_bs4_penalty(self):
        """Limitation tasks don't get BS4 penalty."""
        task_info = {
            "solvable": False,
            "limit_info": {
                "allowed_reasons": ["js_required"],
            },
        }
        output = '{"status": "limit", "answer": null, "limit": {"reason": "js_required", "evidence": "<script>"}}'
        html = "<html><script>document.write('dynamic')</script></html>"
        code = 're.search(r"<script>", HTML)'  # No BS4

        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            html=html,
            code_samples=[code],
        )

        # Limitation tasks don't get BS4 penalty
        assert reward == 0.5
        assert metrics["correct"] is True

    def test_bs4_penalty_with_efficiency(self, task_info):
        """BS4 penalty stacks with efficiency penalty."""
        output = '{"status": "ok", "answer": "expected_answer"}'
        code = 're.search(r"pattern", HTML)'

        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            tool_call_count=3,  # 0.8x efficiency multiplier
            code_samples=[code],
        )

        # 1.0 * 0.8 (efficiency) - 0.15 (BS4 penalty) = 0.65
        expected = (REWARD_CORRECT * 0.8) - BS4_USAGE_PENALTY
        assert reward == pytest.approx(expected)


class TestBS4EdgeCases:
    """Edge case tests for BS4 detection."""

    def test_case_insensitive_detection(self):
        """Detection is case-insensitive for class names."""
        code = 'beautifulsoup(html, "html.parser")'  # lowercase
        assert check_bs4_usage([code]) is True

    def test_partial_match_find_all(self):
        """Partial match on .find_all( works (BS4-specific)."""
        code = 'element.find_all("div")'
        assert check_bs4_usage([code]) is True

    def test_comment_containing_bs4(self):
        """Comments containing BS4 keywords count (conservative)."""
        # This is intentional - we're being conservative
        code = '# Use BeautifulSoup here\nHTML.find("div")'
        assert check_bs4_usage([code]) is True

    def test_string_containing_bs4(self):
        """Strings containing BS4 keywords count (conservative)."""
        # This is intentional - we're being conservative
        code = 'print("Use BeautifulSoup for parsing")'
        assert check_bs4_usage([code]) is True
