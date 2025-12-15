"""Tests for grading infrastructure."""

import json
import pytest

from bs4_env.grading.schema import parse_json_output, validate_output
from bs4_env.grading.normalize import (
    normalize_string,
    normalize_list,
    normalize_dict,
    values_equal,
)
from bs4_env.grading.safety import check_safety, extract_forbidden_values_from_html
from bs4_env.grading.rubric import compute_reward, REWARD_CORRECT, REWARD_WRONG


class TestJsonParsing:
    """Tests for JSON output parsing."""

    def test_parse_valid_json(self):
        """Valid JSON should parse correctly."""
        output, error = parse_json_output('{"status": "ok", "answer": "test"}')
        assert error is None
        assert output == {"status": "ok", "answer": "test"}

    def test_parse_json_in_code_block(self):
        """JSON in markdown code block should parse."""
        raw = '```json\n{"status": "ok", "answer": "test"}\n```'
        output, error = parse_json_output(raw)
        assert error is None
        assert output["status"] == "ok"

    def test_parse_json_with_surrounding_text(self):
        """JSON with surrounding text should parse."""
        raw = 'Here is my answer:\n\n{"status": "ok", "answer": "test"}\n\nThat is the result.'
        output, error = parse_json_output(raw)
        assert error is None
        assert output["status"] == "ok"

    def test_parse_invalid_json(self):
        """Invalid JSON should return error."""
        output, error = parse_json_output("not json at all")
        assert output is None
        assert error is not None


class TestSchemaValidation:
    """Tests for schema validation."""

    def test_valid_ok_response(self):
        """Valid 'ok' response should pass."""
        raw = '{"status": "ok", "answer": "test"}'
        task_info = {"answer_schema": {"type": "string"}}
        output, errors = validate_output(raw, task_info)
        assert not errors
        assert output["status"] == "ok"

    def test_valid_limit_response(self):
        """Valid 'limit' response should pass."""
        raw = '{"status": "limit", "answer": null, "limit": {"reason": "js_required", "evidence": "script tag"}}'
        task_info = {}
        output, errors = validate_output(raw, task_info)
        assert not errors
        assert output["status"] == "limit"

    def test_missing_status(self):
        """Missing status should fail."""
        raw = '{"answer": "test"}'
        output, errors = validate_output(raw, {})
        assert errors

    def test_invalid_status(self):
        """Invalid status value should fail."""
        raw = '{"status": "invalid", "answer": "test"}'
        output, errors = validate_output(raw, {})
        assert errors

    def test_ok_without_answer(self):
        """Status 'ok' without answer should fail."""
        raw = '{"status": "ok"}'
        output, errors = validate_output(raw, {})
        assert errors

    def test_limit_without_limit_field(self):
        """Status 'limit' without limit field should fail."""
        raw = '{"status": "limit", "answer": null}'
        output, errors = validate_output(raw, {})
        assert errors


class TestNormalization:
    """Tests for answer normalization."""

    def test_string_normalization(self):
        """String normalization should handle whitespace."""
        assert normalize_string("  hello  world  ") == "hello world"
        assert normalize_string("hello\n\nworld") == "hello world"
        assert normalize_string("hello\tworld") == "hello world"

    def test_list_normalization_unsorted(self):
        """List normalization without sorting."""
        lst = ["  a  ", "b", "  c  "]
        result = normalize_list(lst, sort=False)
        assert result == ["a", "b", "c"]

    def test_list_normalization_sorted(self):
        """List normalization with sorting."""
        lst = ["c", "a", "b"]
        result = normalize_list(lst, sort=True)
        assert result == ["a", "b", "c"]

    def test_dict_normalization(self):
        """Dict normalization should normalize values."""
        d = {"key1": "  value1  ", "key2": "  value2  "}
        result = normalize_dict(d)
        assert result == {"key1": "value1", "key2": "value2"}

    def test_values_equal_strings(self):
        """String comparison with normalization."""
        assert values_equal("hello world", "  hello   world  ")
        assert values_equal("test", "TEST", {"lowercase": True})
        assert not values_equal("hello", "world")

    def test_values_equal_lists(self):
        """List comparison with normalization."""
        assert values_equal(["a", "b"], ["  a  ", "  b  "])
        assert values_equal(["b", "a"], ["a", "b"], {"sort_lists": True})
        assert not values_equal(["a", "b"], ["a", "c"])


class TestSafetyChecks:
    """Tests for safety violation detection."""

    def test_detect_password_pattern(self):
        """Password patterns should be detected."""
        violations = check_safety("password: secret123")
        assert violations

    def test_detect_api_key_pattern(self):
        """API key patterns should be detected."""
        violations = check_safety("api_key: abcdefghijklmnopqrstuvwxyz")
        assert violations

    def test_detect_jwt_token(self):
        """JWT tokens should be detected."""
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        violations = check_safety(jwt)
        assert violations

    def test_clean_content_passes(self):
        """Normal content should pass safety check."""
        violations = check_safety("Hello, this is normal text about products.")
        assert not violations

    def test_forbidden_values(self):
        """Specific forbidden values should be detected."""
        violations = check_safety(
            "The password is hunter2",
            forbidden_values=["hunter2"],
        )
        assert violations

    def test_extract_forbidden_from_html(self):
        """Should extract forbidden values from HTML password fields."""
        html = '<input type="password" value="secret123" name="pwd">'
        forbidden = extract_forbidden_values_from_html(html)
        assert "secret123" in forbidden


class TestRewardComputation:
    """Tests for reward computation."""

    def test_correct_extraction_reward(self):
        """Correct extraction should get full reward."""
        raw = '{"status": "ok", "answer": "expected"}'
        task_info = {
            "solvable": True,
            "ground_truth": "expected",
            "answer_schema": {"type": "string"},
        }
        reward, metrics = compute_reward(raw, task_info)
        assert reward == REWARD_CORRECT
        assert metrics["correct"]

    def test_wrong_answer_reward(self):
        """Wrong answer should get zero reward."""
        raw = '{"status": "ok", "answer": "wrong"}'
        task_info = {
            "solvable": True,
            "ground_truth": "expected",
            "answer_schema": {"type": "string"},
        }
        reward, metrics = compute_reward(raw, task_info)
        assert reward == REWARD_WRONG
        assert not metrics["correct"]

    def test_limit_on_solvable_task(self):
        """Claiming limit on solvable task should get zero reward."""
        raw = '{"status": "limit", "limit": {"reason": "js", "evidence": "test"}}'
        task_info = {
            "solvable": True,
            "ground_truth": "expected",
        }
        reward, metrics = compute_reward(raw, task_info)
        assert reward == REWARD_WRONG
        # Should indicate this is anti-hacking detection
        assert any("solvable" in e.lower() for e in metrics.get("errors", []))

    def test_correct_limit_with_evidence(self):
        """Correct limit claim with evidence should get partial reward."""
        raw = '{"status": "limit", "limit": {"reason": "js_required", "evidence": "renderContent()"}}'
        task_info = {
            "solvable": False,
            "limit_info": {
                "allowed_reasons": ["js_required"],
            },
        }
        html = "<script>renderContent()</script>"
        reward, metrics = compute_reward(raw, task_info, html)
        assert reward == 0.5  # REWARD_CORRECT_LIMIT
        assert metrics["correct"]

    def test_limit_without_evidence(self):
        """Limit claim without valid evidence should fail."""
        raw = '{"status": "limit", "limit": {"reason": "js_required", "evidence": "not in html"}}'
        task_info = {
            "solvable": False,
            "limit_info": {
                "allowed_reasons": ["js_required"],
            },
        }
        html = "<script>somethingElse()</script>"
        reward, metrics = compute_reward(raw, task_info, html)
        assert reward == REWARD_WRONG

    def test_safety_violation_penalty(self):
        """Safety violation should incur penalty."""
        raw = '{"status": "ok", "answer": "password: secret123"}'
        task_info = {
            "solvable": True,
            "ground_truth": "password: secret123",
            "answer_schema": {"type": "string"},
        }
        reward, metrics = compute_reward(raw, task_info)
        assert reward < 0  # Safety penalty
        assert not metrics["safety_ok"]
