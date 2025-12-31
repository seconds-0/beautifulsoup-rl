"""Tests for grading infrastructure."""

from bs4_env.grading.normalize import (
    normalize_dict,
    normalize_list,
    normalize_string,
    values_equal,
)
from bs4_env.grading.rubric import REWARD_CORRECT, REWARD_WRONG, compute_reward
from bs4_env.grading.safety import check_safety, extract_forbidden_values_from_html
from bs4_env.grading.schema import parse_json_output, validate_output


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
        raw = (
            '{"status": "limit", "limit": {"reason": "js_required", "evidence": "renderContent()"}}'
        )
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

    def test_limit_no_efficiency_penalty(self):
        """Limit responses should NOT get efficiency penalty applied.

        Models legitimately need to explore before recognizing a limitation,
        so we don't penalize them for taking 2-4 tool calls.
        """
        raw = (
            '{"status": "limit", "limit": {"reason": "js_required", "evidence": "renderContent()"}}'
        )
        task_info = {
            "solvable": False,
            "limit_info": {
                "allowed_reasons": ["js_required"],
            },
        }
        html = "<script>renderContent()</script>"
        # Even with 4 tool calls, limit reward should stay at 0.5
        reward, metrics = compute_reward(raw, task_info, html, tool_call_count=4)
        assert reward == 0.5  # No efficiency penalty applied
        assert metrics["correct"]
        # Efficiency is still recorded for analysis purposes
        assert metrics.get("efficiency_multiplier") is not None
        assert metrics.get("limit_efficiency_exemption") is True

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


class TestIntegerCoercion:
    """Tests for integer type coercion in grading."""

    def test_string_integer_accepts(self):
        """String '4' should be accepted when schema expects integer."""
        raw = '{"status": "ok", "answer": "4"}'
        task_info = {
            "solvable": True,
            "ground_truth": 4,
            "answer_schema": {"type": "integer"},
        }
        reward, metrics = compute_reward(raw, task_info)
        assert reward == REWARD_CORRECT
        assert metrics["correct"]
        assert metrics.get("coercion_applied")

    def test_integer_matches_integer(self):
        """Integer 4 should match ground truth 4."""
        raw = '{"status": "ok", "answer": 4}'
        task_info = {
            "solvable": True,
            "ground_truth": 4,
            "answer_schema": {"type": "integer"},
        }
        reward, metrics = compute_reward(raw, task_info)
        assert reward == REWARD_CORRECT
        assert metrics["correct"]

    def test_wrong_integer_fails(self):
        """Wrong integer value should fail."""
        raw = '{"status": "ok", "answer": "5"}'
        task_info = {
            "solvable": True,
            "ground_truth": 4,
            "answer_schema": {"type": "integer"},
        }
        reward, metrics = compute_reward(raw, task_info)
        assert reward == REWARD_WRONG
        assert not metrics["correct"]

    def test_non_integer_string_fails(self):
        """Non-integer string like '4.5' should fail."""
        raw = '{"status": "ok", "answer": "4.5"}'
        task_info = {
            "solvable": True,
            "ground_truth": 4,
            "answer_schema": {"type": "integer"},
        }
        reward, metrics = compute_reward(raw, task_info)
        assert reward == REWARD_WRONG


class TestKeyAliasing:
    """Tests for object key aliasing in grading."""

    def test_aliased_key_accepted(self):
        """'cheaper_product' should be accepted when ground truth uses 'cheaper'."""
        raw = '{"status": "ok", "answer": {"cheaper_product": "ProductA", "price_difference": "$10.00"}}'
        task_info = {
            "solvable": True,
            "ground_truth": {"cheaper": "ProductA", "price_difference": "$10.00"},
            "answer_schema": {
                "type": "object",
                "properties": {
                    "cheaper": {"type": "string"},
                    "price_difference": {"type": "string"},
                },
            },
        }
        reward, metrics = compute_reward(raw, task_info)
        assert reward == REWARD_CORRECT
        assert metrics["correct"]

    def test_camel_case_key_accepted(self):
        """'priceDifference' should alias to 'price_difference'."""
        raw = '{"status": "ok", "answer": {"cheaper": "ProductA", "priceDifference": "$10.00"}}'
        task_info = {
            "solvable": True,
            "ground_truth": {"cheaper": "ProductA", "price_difference": "$10.00"},
            "answer_schema": {
                "type": "object",
                "properties": {
                    "cheaper": {"type": "string"},
                    "price_difference": {"type": "string"},
                },
            },
        }
        reward, metrics = compute_reward(raw, task_info)
        assert reward == REWARD_CORRECT
        assert metrics["correct"]

    def test_wrong_value_still_fails(self):
        """Aliased key with wrong value should still fail."""
        raw = '{"status": "ok", "answer": {"cheaper_product": "WRONG", "price_difference": "$10.00"}}'
        task_info = {
            "solvable": True,
            "ground_truth": {"cheaper": "ProductA", "price_difference": "$10.00"},
            "answer_schema": {"type": "object"},
        }
        reward, metrics = compute_reward(raw, task_info)
        assert reward == REWARD_WRONG


class TestPriceNormalization:
    """Tests for price format normalization in grading."""

    def test_price_without_dollar_sign(self):
        """Price '185.40' should match ground truth '$185.40'."""
        raw = '{"status": "ok", "answer": "185.40"}'
        task_info = {
            "solvable": True,
            "ground_truth": "$185.40",
            "answer_schema": {"type": "string"},
        }
        reward, metrics = compute_reward(raw, task_info)
        assert reward == REWARD_CORRECT
        assert metrics["correct"]

    def test_price_with_single_decimal(self):
        """Price '$185.4' should match '$185.40' after normalization."""
        raw = '{"status": "ok", "answer": "$185.4"}'
        task_info = {
            "solvable": True,
            "ground_truth": "$185.40",
            "answer_schema": {"type": "string"},
        }
        reward, metrics = compute_reward(raw, task_info)
        assert reward == REWARD_CORRECT
        assert metrics["correct"]

    def test_numeric_price_value(self):
        """Numeric price 185.40 should match '$185.40'."""
        raw = '{"status": "ok", "answer": 185.40}'
        task_info = {
            "solvable": True,
            "ground_truth": "$185.40",
            "answer_schema": {"type": "string"},
        }
        reward, metrics = compute_reward(raw, task_info)
        assert reward == REWARD_CORRECT
        assert metrics["correct"]

    def test_nested_price_in_object(self):
        """Prices nested in objects should be normalized."""
        raw = '{"status": "ok", "answer": {"min": "100.5", "max": "$200.00"}}'
        task_info = {
            "solvable": True,
            "ground_truth": {"min": "$100.50", "max": "$200.00"},
            "answer_schema": {"type": "object"},
        }
        reward, metrics = compute_reward(raw, task_info)
        assert reward == REWARD_CORRECT
        assert metrics["correct"]

    def test_wrong_price_fails(self):
        """Wrong price value should still fail."""
        raw = '{"status": "ok", "answer": "$99.99"}'
        task_info = {
            "solvable": True,
            "ground_truth": "$185.40",
            "answer_schema": {"type": "string"},
        }
        reward, metrics = compute_reward(raw, task_info)
        assert reward == REWARD_WRONG


class TestCoercionFunctions:
    """Direct tests for the coercion functions in normalize.py."""

    def test_coerce_integer_from_string(self):
        """Test coerce_integer with string input."""
        from bs4_env.grading.normalize import coerce_integer

        assert coerce_integer("4") == 4
        assert coerce_integer(" 42 ") == 42
        assert coerce_integer("-10") == -10
        assert coerce_integer("+5") == 5

    def test_coerce_integer_from_int(self):
        """Test coerce_integer with int input."""
        from bs4_env.grading.normalize import coerce_integer

        assert coerce_integer(4) == 4
        assert coerce_integer(0) == 0
        assert coerce_integer(-100) == -100

    def test_coerce_integer_rejects_invalid(self):
        """Test coerce_integer rejects invalid inputs."""
        import pytest

        from bs4_env.grading.normalize import coerce_integer

        with pytest.raises(ValueError):
            coerce_integer("4.5")
        with pytest.raises(ValueError):
            coerce_integer(True)
        with pytest.raises(ValueError):
            coerce_integer(4.0)
        with pytest.raises(ValueError):
            coerce_integer("four")

    def test_normalize_object_keys(self):
        """Test normalize_object_keys applies aliases."""
        from bs4_env.grading.normalize import normalize_object_keys

        result = normalize_object_keys(
            {"cheaper_product": "A", "price_diff": "$10"}
        )
        assert result == {"cheaper": "A", "price_difference": "$10"}

    def test_normalize_object_keys_collision(self):
        """Test normalize_object_keys rejects collisions."""
        import pytest

        from bs4_env.grading.normalize import normalize_object_keys

        with pytest.raises(ValueError):
            # Both 'cheaper' and 'cheaper_product' would become 'cheaper'
            normalize_object_keys({"cheaper": "A", "cheaper_product": "B"})

    def test_normalize_price(self):
        """Test normalize_price canonicalizes formats."""
        from bs4_env.grading.normalize import normalize_price

        assert normalize_price("$99.99") == "$99.99"
        assert normalize_price("99.99") == "$99.99"
        assert normalize_price("$99.9") == "$99.90"
        assert normalize_price(99.99) == "$99.99"
        assert normalize_price(100) == "$100.00"
        assert normalize_price("$1,234.56") == "$1234.56"

    def test_normalize_price_rejects_invalid(self):
        """Test normalize_price rejects invalid inputs."""
        import pytest

        from bs4_env.grading.normalize import normalize_price

        with pytest.raises(ValueError):
            normalize_price("not a price")
        with pytest.raises(ValueError):
            normalize_price(True)
