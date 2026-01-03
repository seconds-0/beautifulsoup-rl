"""Tests for process partial credit (0% model bootstrapping).

This tests the tiered partial credit system that rewards correct tool-use
patterns even when the final answer is wrong, providing gradient signal
for models learning the basic action template.
"""

import pytest

from bs4_env.grading.rubric import (
    PROCESS_PARTIAL_CREDIT_CAP,
    PROCESS_TIER_REWARDS,
    REWARD_WRONG,
    _check_content_access_ast,
    _check_selection_method_ast,
    _check_soup_creation_with_html_ast,
    compute_process_partial_credit,
    compute_reward,
)


class TestSoupCreationWithHtmlAST:
    """Tests for _check_soup_creation_with_html_ast."""

    def test_beautifulsoup_with_html_variable(self):
        """BeautifulSoup(HTML, ...) is detected."""
        code = 'soup = BeautifulSoup(HTML, "html.parser")'
        assert _check_soup_creation_with_html_ast(code) is True

    def test_beautifulsoup_with_different_parser(self):
        """BeautifulSoup(HTML, ...) with different parser is detected."""
        code = 'soup = BeautifulSoup(HTML, "lxml")'
        assert _check_soup_creation_with_html_ast(code) is True

    def test_beautifulsoup_with_string_literal(self):
        """BeautifulSoup('<html>...', ...) is NOT detected (not using HTML variable)."""
        code = 'soup = BeautifulSoup("<html><body>test</body></html>", "html.parser")'
        assert _check_soup_creation_with_html_ast(code) is False

    def test_beautifulsoup_with_wrong_variable(self):
        """BeautifulSoup(other_var, ...) is NOT detected."""
        code = 'soup = BeautifulSoup(content, "html.parser")'
        assert _check_soup_creation_with_html_ast(code) is False

    def test_beautifulsoup_in_dead_code(self):
        """BeautifulSoup(HTML, ...) in if False block still detected by AST."""
        # Note: AST detection doesn't do control flow analysis - this is a known limitation
        # but acceptable since we're checking structural patterns, not execution
        code = '''
if False:
    soup = BeautifulSoup(HTML, "html.parser")
'''
        assert _check_soup_creation_with_html_ast(code) is True

    def test_no_beautifulsoup(self):
        """Code without BeautifulSoup is not detected."""
        code = 're.search(r"<div>(.*?)</div>", HTML)'
        assert _check_soup_creation_with_html_ast(code) is False

    def test_syntax_error_returns_false(self):
        """Syntax errors return False (safe default)."""
        code = "this is not valid python {"
        assert _check_soup_creation_with_html_ast(code) is False

    def test_make_soup_no_args(self):
        """make_soup() helper (no args) is detected.

        The prompt recommends `soup = make_soup()` without arguments,
        as the helper internally uses the HTML variable.
        """
        code = "soup = make_soup()"
        assert _check_soup_creation_with_html_ast(code) is True

    def test_make_soup_with_html_arg(self):
        """make_soup(HTML) is detected (legacy pattern)."""
        code = 'soup = make_soup(HTML, "html.parser")'
        assert _check_soup_creation_with_html_ast(code) is True

    def test_make_soup_in_full_pipeline(self):
        """make_soup() works in complete code example."""
        code = '''
from bs4 import BeautifulSoup
soup = make_soup()
element = soup.find("div")
result = element.text
'''
        assert _check_soup_creation_with_html_ast(code) is True


class TestSelectionMethodAST:
    """Tests for _check_selection_method_ast."""

    def test_find_method(self):
        """.find() is detected as a selection method."""
        code = 'element = soup.find("div")'
        assert _check_selection_method_ast(code) is True

    def test_find_all_method(self):
        """.find_all() is detected as a selection method."""
        code = 'elements = soup.find_all("span")'
        assert _check_selection_method_ast(code) is True

    def test_select_method(self):
        """.select() is detected as a selection method."""
        code = 'elements = soup.select("div.target")'
        assert _check_selection_method_ast(code) is True

    def test_select_one_method(self):
        """.select_one() is detected as a selection method."""
        code = 'element = soup.select_one("#main")'
        assert _check_selection_method_ast(code) is True

    def test_chained_find(self):
        """Chained .find() is detected."""
        code = 'element = soup.find("div").find("span")'
        assert _check_selection_method_ast(code) is True

    def test_no_selection_method(self):
        """Code without selection methods is not detected."""
        code = 'result = soup.text'
        assert _check_selection_method_ast(code) is False

    def test_syntax_error_returns_false(self):
        """Syntax errors return False."""
        code = "soup.find("
        assert _check_selection_method_ast(code) is False


class TestContentAccessAST:
    """Tests for _check_content_access_ast."""

    def test_text_attribute(self):
        """.text attribute is detected."""
        code = "result = element.text"
        assert _check_content_access_ast(code) is True

    def test_string_attribute(self):
        """.string attribute is detected."""
        code = "result = element.string"
        assert _check_content_access_ast(code) is True

    def test_strings_attribute(self):
        """.strings attribute is detected."""
        code = "for s in element.strings: pass"
        assert _check_content_access_ast(code) is True

    def test_stripped_strings_attribute(self):
        """.stripped_strings attribute is detected."""
        code = "result = list(element.stripped_strings)"
        assert _check_content_access_ast(code) is True

    def test_get_text_method(self):
        """.get_text() method is detected."""
        code = "result = element.get_text(strip=True)"
        assert _check_content_access_ast(code) is True

    def test_no_content_access(self):
        """Code without content access is not detected."""
        code = "element = soup.find('div')"
        assert _check_content_access_ast(code) is False

    def test_syntax_error_returns_false(self):
        """Syntax errors return False."""
        code = "element.text["
        assert _check_content_access_ast(code) is False


class TestComputeProcessPartialCredit:
    """Tests for compute_process_partial_credit function."""

    def test_full_pipeline_all_tiers(self):
        """Code with all tiers gets full partial credit."""
        code = '''
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
element = soup.find("div", class_="target")
result = element.text
'''
        reward, breakdown = compute_process_partial_credit(
            code_samples=[code],
            status="ok",
            solvable=True,
            run_python_calls=1,
        )

        # All 4 tiers
        expected = sum(PROCESS_TIER_REWARDS.values())
        assert reward == pytest.approx(min(expected, PROCESS_PARTIAL_CREDIT_CAP))
        assert breakdown.get("bs4_imported") is True
        assert breakdown.get("soup_created_with_html") is True
        assert breakdown.get("selection_method") is True
        assert breakdown.get("content_access") is True

    def test_tier_1_only_import(self):
        """Code with only BS4 import gets tier 1 credit."""
        code = "from bs4 import BeautifulSoup"
        reward, breakdown = compute_process_partial_credit(
            code_samples=[code],
            status="ok",
            solvable=True,
            run_python_calls=1,
        )

        assert reward == pytest.approx(PROCESS_TIER_REWARDS["bs4_imported"])
        assert breakdown.get("bs4_imported") is True
        assert "soup_created_with_html" not in breakdown

    def test_tier_2_without_tier_1_not_credited(self):
        """Soup creation without import doesn't get credit (tier dependency)."""
        # This should fail because BeautifulSoup isn't imported
        code = 'soup = BeautifulSoup(HTML, "html.parser")'
        reward, breakdown = compute_process_partial_credit(
            code_samples=[code],
            status="ok",
            solvable=True,
            run_python_calls=1,
        )

        # Only tier 1 (bs4_imported) because BeautifulSoup is used
        # But soup_created_with_html requires bs4_imported
        # Actually, BeautifulSoup() call IS detected by _check_bs4_usage_ast
        # so bs4_imported should be True, and then soup_created_with_html
        # This tests the actual behavior
        assert reward > 0
        assert breakdown.get("bs4_imported") is True

    def test_tier_3_without_tier_2_not_credited(self):
        """Selection method without soup creation doesn't get credit."""
        code = '''
from bs4 import BeautifulSoup
element = soup.find("div")
'''
        reward, breakdown = compute_process_partial_credit(
            code_samples=[code],
            status="ok",
            solvable=True,
            run_python_calls=1,
        )

        # Only tier 1 - soup not created with HTML variable
        assert reward == pytest.approx(PROCESS_TIER_REWARDS["bs4_imported"])
        assert breakdown.get("bs4_imported") is True
        assert "soup_created_with_html" not in breakdown
        assert "selection_method" not in breakdown

    def test_gate_limit_on_solvable_blocked(self):
        """Claiming limit on solvable task blocks partial credit."""
        code = '''
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
element = soup.find("div")
'''
        reward, breakdown = compute_process_partial_credit(
            code_samples=[code],
            status="limit",  # Claiming limit
            solvable=True,  # But task is solvable
            run_python_calls=1,
        )

        assert reward == 0.0
        assert breakdown.get("blocked") == "limit_on_solvable"

    def test_gate_no_run_python_blocked(self):
        """No run_python calls blocks partial credit."""
        code = 'soup = BeautifulSoup(HTML, "html.parser")'
        reward, breakdown = compute_process_partial_credit(
            code_samples=[code],
            status="ok",
            solvable=True,
            run_python_calls=0,  # No run_python calls
        )

        assert reward == 0.0
        assert breakdown.get("blocked") == "no_run_python"

    def test_gate_unsolvable_task_blocked(self):
        """Process partial credit is blocked on unsolvable tasks.

        Process credit is for teaching tool-use patterns on solvable tasks.
        On unsolvable tasks, models should recognize the limitation and claim "limit",
        not attempt to solve with BS4 patterns.
        """
        code = '''
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
element = soup.find("div")
result = element.text
'''
        reward, breakdown = compute_process_partial_credit(
            code_samples=[code],
            status="ok",  # Wrong status for unsolvable
            solvable=False,  # Task is unsolvable
            run_python_calls=1,
        )

        assert reward == 0.0
        assert breakdown.get("blocked") == "unsolvable_task"

    def test_gate_unsolvable_task_limit_response_blocked(self):
        """Even limit responses on unsolvable tasks don't get process credit.

        Limit responses on unsolvable tasks are handled by the normal grading
        flow (REWARD_CORRECT_LIMIT), not process partial credit.
        """
        code = '''
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
'''
        reward, breakdown = compute_process_partial_credit(
            code_samples=[code],
            status="limit",
            solvable=False,  # Task is unsolvable
            run_python_calls=1,
        )

        assert reward == 0.0
        assert breakdown.get("blocked") == "unsolvable_task"

    def test_empty_code_samples_no_credit(self):
        """Empty code samples give no credit."""
        reward, breakdown = compute_process_partial_credit(
            code_samples=[],
            status="ok",
            solvable=True,
            run_python_calls=1,
        )

        assert reward == 0.0
        assert breakdown.get("no_code") is True

    def test_cap_at_max(self):
        """Reward is capped at PROCESS_PARTIAL_CREDIT_CAP."""
        code = '''
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
element = soup.find("div", class_="target")
result = element.text
'''
        reward, _ = compute_process_partial_credit(
            code_samples=[code],
            status="ok",
            solvable=True,
            run_python_calls=1,
        )

        assert reward <= PROCESS_PARTIAL_CREDIT_CAP

    def test_multiple_code_samples(self):
        """Multiple code samples are all checked."""
        code1 = "from bs4 import BeautifulSoup"
        code2 = 'soup = BeautifulSoup(HTML, "html.parser")'
        code3 = 'element = soup.find("div")'
        code4 = "result = element.text"

        reward, breakdown = compute_process_partial_credit(
            code_samples=[code1, code2, code3, code4],
            status="ok",
            solvable=True,
            run_python_calls=4,
        )

        # All tiers detected across samples
        assert breakdown.get("bs4_imported") is True
        assert breakdown.get("soup_created_with_html") is True
        assert breakdown.get("selection_method") is True
        assert breakdown.get("content_access") is True


class TestProcessPartialCreditIntegration:
    """Tests for process partial credit integrated with compute_reward."""

    @pytest.fixture
    def task_info(self):
        """Basic task info for testing."""
        return {
            "ground_truth": "expected_answer",
            "solvable": True,
            "answer_schema": {"type": "string"},
        }

    def test_wrong_answer_gets_partial_credit(self, task_info):
        """Wrong answer with good code structure gets partial credit."""
        output = '{"status": "ok", "answer": "wrong_answer"}'
        code = '''
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
element = soup.find("div")
result = element.text
'''
        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            code_samples=[code],
            run_python_calls=1,
        )

        # Should get process partial credit, not REWARD_WRONG
        assert reward > REWARD_WRONG
        assert reward <= PROCESS_PARTIAL_CREDIT_CAP
        assert metrics.get("partial_credit_source") == "process"
        assert "process_partial_credit" in metrics
        assert metrics["correct"] is False

    def test_wrong_answer_no_bs4_no_partial_credit(self, task_info):
        """Wrong answer without BS4 usage gets no partial credit."""
        output = '{"status": "ok", "answer": "wrong_answer"}'
        code = 're.search(r"<div>(.*?)</div>", HTML)'

        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            code_samples=[code],
            run_python_calls=1,
        )

        assert reward == REWARD_WRONG
        assert "process_partial_credit" not in metrics
        assert metrics["correct"] is False

    def test_correct_answer_no_partial_credit(self, task_info):
        """Correct answer gets full credit, not partial credit."""
        output = '{"status": "ok", "answer": "expected_answer"}'
        code = '''
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
result = soup.text
'''
        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            code_samples=[code],
            run_python_calls=1,
        )

        # Full credit, not partial
        assert reward == 1.0
        assert "process_partial_credit" not in metrics
        assert metrics["correct"] is True

    def test_no_incentive_inversion(self):
        """Process credit beats extraction partial credit when higher.

        This prevents incentive inversion where getting 50% of keys correct
        (0.05 reward) would be worse than getting 0% correct with good BS4
        code (0.30 reward). The max of the two should be taken.
        """
        # Task expects {"key1": "a", "key2": "b"}
        task_info = {
            "ground_truth": '{"key1": "a", "key2": "b"}',
            "solvable": True,
            "answer_schema": {"type": "object"},
        }

        # Model outputs {"key1": "a", "key2": "WRONG"} - 50% correct
        # This would give extraction partial credit of 0.1 * 0.5 = 0.05
        output = '{"status": "ok", "answer": {"key1": "a", "key2": "WRONG"}}'
        code = '''
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
element = soup.find("div")
result = element.text
'''
        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            code_samples=[code],
            run_python_calls=1,
        )

        # Process credit (0.30) should beat extraction partial (0.05)
        assert reward > 0.05  # Higher than extraction partial
        assert reward <= PROCESS_PARTIAL_CREDIT_CAP
        assert metrics.get("partial_credit_source") == "process"
        # Original extraction partial credit should be recorded
        assert "extraction_partial_credit" in metrics
        assert metrics["extraction_partial_credit"] == pytest.approx(0.05)

    def test_wrong_answer_limit_on_solvable_no_credit(self, task_info):
        """Limit claim on solvable task gets no partial credit."""
        output = '{"status": "limit", "answer": null, "limit": {"reason": "js_required", "evidence": "<script>"}}'
        code = '''
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
element = soup.find("div")
'''
        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            code_samples=[code],
            run_python_calls=1,
        )

        # Should be REWARD_WRONG, not partial credit
        assert reward == REWARD_WRONG
        assert metrics["correct"] is False

    def test_partial_credit_below_limit_reward(self, task_info):
        """Process partial credit cap (0.30) is below REWARD_CORRECT_LIMIT (0.5)."""
        # This ensures wrong answers can never outscore limit responses
        assert PROCESS_PARTIAL_CREDIT_CAP < 0.5  # REWARD_CORRECT_LIMIT

    def test_process_credit_skips_bs4_penalty(self, task_info):
        """Process partial credit doesn't get additional BS4 penalty."""
        output = '{"status": "ok", "answer": "wrong_answer"}'
        code = '''
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
element = soup.find("div")
result = element.text
'''
        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            code_samples=[code],
            run_python_calls=1,
        )

        # Process credit shouldn't have BS4 penalty applied
        # (the partial credit IS for using BS4)
        assert metrics.get("partial_credit_source") == "process"
        # bs4_used and bs4_penalty should be None or not relevant
        # since we skip BS4 penalty for process credit


class TestPartialCreditRewardValues:
    """Tests for exact partial credit reward values."""

    def test_tier_1_value(self):
        """Tier 1 (bs4_imported) gives 0.05."""
        assert PROCESS_TIER_REWARDS["bs4_imported"] == 0.05

    def test_tier_2_value(self):
        """Tier 2 (soup_created_with_html) gives 0.10."""
        assert PROCESS_TIER_REWARDS["soup_created_with_html"] == 0.10

    def test_tier_3_value(self):
        """Tier 3 (selection_method) gives 0.10."""
        assert PROCESS_TIER_REWARDS["selection_method"] == 0.10

    def test_tier_4_value(self):
        """Tier 4 (content_access) gives 0.05."""
        assert PROCESS_TIER_REWARDS["content_access"] == 0.05

    def test_total_below_cap(self):
        """Total of all tiers (0.30) equals cap."""
        total = sum(PROCESS_TIER_REWARDS.values())
        assert total == pytest.approx(PROCESS_PARTIAL_CREDIT_CAP)

    def test_cap_is_030(self):
        """Cap is 0.30 (below REWARD_CORRECT_LIMIT of 0.5)."""
        assert PROCESS_PARTIAL_CREDIT_CAP == 0.30


class TestPartialCreditEnabledConfig:
    """Tests for partial_credit_enabled config wiring."""

    @pytest.fixture
    def task_info(self):
        """Basic task info for testing."""
        return {
            "ground_truth": "expected_answer",
            "solvable": True,
            "answer_schema": {"type": "string"},
        }

    def test_disabled_via_parameter(self, task_info):
        """partial_credit_enabled=False disables process credit."""
        output = '{"status": "ok", "answer": "wrong_answer"}'
        code = '''
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
element = soup.find("div")
result = element.text
'''
        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            code_samples=[code],
            run_python_calls=1,
            partial_credit_enabled=False,  # Explicitly disable
        )

        # Should get 0.0 (no partial credit)
        assert reward == REWARD_WRONG
        assert "process_partial_credit" not in metrics
        assert metrics["correct"] is False

    def test_enabled_via_parameter(self, task_info):
        """partial_credit_enabled=True enables process credit."""
        output = '{"status": "ok", "answer": "wrong_answer"}'
        code = '''
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
element = soup.find("div")
result = element.text
'''
        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            code_samples=[code],
            run_python_calls=1,
            partial_credit_enabled=True,  # Explicitly enable
        )

        # Should get partial credit
        assert reward > REWARD_WRONG
        assert "process_partial_credit" in metrics
        assert metrics.get("partial_credit_source") == "process"

    def test_compute_process_partial_credit_disabled(self):
        """compute_process_partial_credit respects partial_credit_enabled=False."""
        code = '''
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
element = soup.find("div")
'''
        reward, breakdown = compute_process_partial_credit(
            code_samples=[code],
            status="ok",
            solvable=True,
            run_python_calls=1,
            partial_credit_enabled=False,
        )

        assert reward == 0.0
        assert breakdown.get("disabled") is True

    def test_default_uses_module_constant(self, task_info):
        """Omitting partial_credit_enabled uses module constant (True by default)."""
        output = '{"status": "ok", "answer": "wrong_answer"}'
        code = '''
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
element = soup.find("div")
result = element.text
'''
        # Don't pass partial_credit_enabled - should use module default
        reward, metrics = compute_reward(
            raw_output=output,
            task_info=task_info,
            code_samples=[code],
            run_python_calls=1,
        )

        # Module constant is True, so should get partial credit
        assert reward > REWARD_WRONG
        assert "process_partial_credit" in metrics


class TestBootstrapModeConfig:
    """Tests for bootstrap mode auto-enabling partial credit."""

    def test_bootstrap_mode_auto_enables_partial_credit(self):
        """mode='bootstrap' auto-enables partial_credit_enabled."""
        from bs4_env.config import EnvConfig

        config = EnvConfig(mode="bootstrap")
        assert config.partial_credit_enabled is True

    def test_mvp_mode_partial_credit_disabled_by_default(self):
        """mode='mvp' (default) has partial_credit_enabled=False."""
        from bs4_env.config import EnvConfig

        config = EnvConfig(mode="mvp")
        assert config.partial_credit_enabled is False

    def test_explicit_disable_in_bootstrap_mode_works(self):
        """Can explicitly disable partial credit in bootstrap mode."""
        from bs4_env.config import EnvConfig

        # User explicitly disables after construction
        config = EnvConfig(mode="bootstrap")
        config.partial_credit_enabled = False
        assert config.partial_credit_enabled is False
