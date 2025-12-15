"""Tests for local code execution."""

import pytest

from bs4_env.tools.executor import LocalSubprocessExecutor, ExecResult
from bs4_env.tools.harness import build_runner_script


class TestLocalExecutor:
    """Tests for LocalSubprocessExecutor."""

    @pytest.fixture
    def executor(self):
        """Create executor instance."""
        return LocalSubprocessExecutor(max_output_chars=1000)

    def test_simple_print(self, executor):
        """Simple print should work."""
        code = 'print("hello")'
        globals_dict = {"HTML": "", "QUERY": "", "CONSTRAINTS": {}}

        result = executor.run(code, globals_dict, timeout_s=10)

        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert not result.timed_out

    def test_html_global_available(self, executor):
        """HTML global should be available in executed code."""
        code = 'print(f"HTML length: {len(HTML)}")'
        globals_dict = {"HTML": "<div>test</div>", "QUERY": "", "CONSTRAINTS": {}}

        result = executor.run(code, globals_dict, timeout_s=10)

        assert result.exit_code == 0
        assert "HTML length: 15" in result.stdout  # <div>test</div> is 15 chars

    def test_beautifulsoup_available(self, executor):
        """BeautifulSoup should be available."""
        code = """
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, 'html.parser')
print(soup.find('div').text)
"""
        globals_dict = {"HTML": "<div>hello world</div>", "QUERY": "", "CONSTRAINTS": {}}

        result = executor.run(code, globals_dict, timeout_s=10)

        assert result.exit_code == 0
        assert "hello world" in result.stdout

    def test_make_soup_helper(self, executor):
        """make_soup() helper should work."""
        code = """
soup = make_soup()
print(soup.find('p').text)
"""
        globals_dict = {"HTML": "<p>test paragraph</p>", "QUERY": "", "CONSTRAINTS": {}}

        result = executor.run(code, globals_dict, timeout_s=10)

        assert result.exit_code == 0
        assert "test paragraph" in result.stdout

    def test_syntax_error(self, executor):
        """Syntax errors should be captured in stderr."""
        code = "print('unclosed"
        globals_dict = {"HTML": "", "QUERY": "", "CONSTRAINTS": {}}

        result = executor.run(code, globals_dict, timeout_s=10)

        assert result.exit_code != 0
        assert "SyntaxError" in result.stderr or "syntax" in result.stderr.lower()

    def test_runtime_error(self, executor):
        """Runtime errors should be captured."""
        code = """
x = 1 / 0
"""
        globals_dict = {"HTML": "", "QUERY": "", "CONSTRAINTS": {}}

        result = executor.run(code, globals_dict, timeout_s=10)

        assert result.exit_code != 0
        assert "ZeroDivisionError" in result.stderr

    def test_timeout(self, executor):
        """Long-running code should timeout."""
        code = """
import time
time.sleep(100)
"""
        globals_dict = {"HTML": "", "QUERY": "", "CONSTRAINTS": {}}

        result = executor.run(code, globals_dict, timeout_s=1)

        assert result.timed_out
        assert result.exit_code != 0

    def test_output_truncation(self):
        """Output should be truncated at max_output_chars."""
        executor = LocalSubprocessExecutor(max_output_chars=100)
        code = 'print("x" * 1000)'
        globals_dict = {"HTML": "", "QUERY": "", "CONSTRAINTS": {}}

        result = executor.run(code, globals_dict, timeout_s=10)

        assert len(result.stdout) <= 100

    def test_constraints_available(self, executor):
        """CONSTRAINTS global should be available."""
        code = """
import json
print(json.dumps(CONSTRAINTS))
"""
        globals_dict = {
            "HTML": "",
            "QUERY": "",
            "CONSTRAINTS": {"test": "value"},
        }

        result = executor.run(code, globals_dict, timeout_s=10)

        assert result.exit_code == 0
        assert '"test"' in result.stdout
        assert '"value"' in result.stdout


class TestRunnerScript:
    """Tests for runner script generation."""

    def test_script_has_globals(self):
        """Generated script should define globals."""
        script = build_runner_script(
            "print(HTML)",
            {"HTML": "test", "QUERY": "query", "CONSTRAINTS": {}},
        )

        assert 'HTML = """test"""' in script
        assert 'QUERY = """query"""' in script
        assert "CONSTRAINTS = " in script

    def test_script_has_make_soup(self):
        """Generated script should define make_soup helper."""
        script = build_runner_script(
            "x = 1",
            {"HTML": "", "QUERY": "", "CONSTRAINTS": {}},
        )

        assert "def make_soup(" in script

    def test_script_includes_user_code(self):
        """Generated script should include user code."""
        user_code = "result = soup.find('div').text"
        script = build_runner_script(
            user_code,
            {"HTML": "", "QUERY": "", "CONSTRAINTS": {}},
        )

        assert user_code in script

    def test_html_escaping(self):
        """HTML with quotes should be properly escaped."""
        html = '<div class="test">content</div>'
        script = build_runner_script(
            "print(HTML)",
            {"HTML": html, "QUERY": "", "CONSTRAINTS": {}},
        )

        # Script should be valid Python
        compile(script, "<test>", "exec")
