"""Tests for the unified AST analysis function.

These tests verify that analyze_code_unified correctly detects BS4 usage patterns
with a single AST pass instead of multiple separate passes.
"""

from __future__ import annotations

from bs4_env.grading.rubric import analyze_code_unified


class TestUnifiedASTBasics:
    """Basic functionality tests for analyze_code_unified."""

    def test_empty_code(self):
        """Empty code should return default (all False) result."""
        result = analyze_code_unified("")
        assert result.bs4_imported is False
        assert result.soup_created_with_html is False
        assert result.selection_method_used is False
        assert result.content_accessed is False

    def test_syntax_error_returns_default(self):
        """Code with syntax errors should return default result."""
        result = analyze_code_unified("def foo(")
        assert result.bs4_imported is False
        assert result.soup_created_with_html is False

    def test_basic_bs4_usage(self):
        """Standard BS4 usage pattern should be detected."""
        code = """
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
items = soup.find_all("div")
text = items[0].get_text()
"""
        result = analyze_code_unified(code)
        assert result.bs4_imported is True
        assert result.soup_created_with_html is True
        assert result.selection_method_used is True
        assert result.content_accessed is True

    def test_make_soup_helper(self):
        """Using make_soup() helper should be detected."""
        code = """
soup = make_soup()
text = soup.get_text()
"""
        result = analyze_code_unified(code)
        assert result.bs4_imported is True
        assert result.soup_created_with_html is True
        assert result.content_accessed is True


class TestUnifiedASTAliases:
    """Tests for alias handling in unified AST analysis."""

    def test_import_alias_detected(self):
        """from bs4 import BeautifulSoup as BS should be tracked."""
        code = """
from bs4 import BeautifulSoup as BS
soup = BS(HTML, "html.parser")
"""
        result = analyze_code_unified(code)
        assert result.bs4_imported is True
        assert result.soup_created_with_html is True
        assert "BS" in result.bs4_ctor_aliases

    def test_module_alias_detected(self):
        """import bs4 as x; x.BeautifulSoup should be tracked."""
        code = """
import bs4 as soup_lib
soup = soup_lib.BeautifulSoup(HTML, "html.parser")
"""
        result = analyze_code_unified(code)
        assert result.bs4_imported is True
        assert result.soup_created_with_html is True
        assert "soup_lib" in result.bs4_module_aliases

    def test_shadowed_alias_tracked(self):
        """Shadowed aliases should be in shadowed_names."""
        code = """
from bs4 import BeautifulSoup as BS
BS = lambda *a: None
soup = BS(HTML, "html.parser")
"""
        result = analyze_code_unified(code)
        assert "BS" in result.shadowed_names


class TestUnifiedASTHTMLDerived:
    """Tests for HTML-derived name tracking."""

    def test_html_global_tracked(self):
        """HTML should be in html_derived_names by default."""
        result = analyze_code_unified("")
        assert "HTML" in result.html_derived_names

    def test_assignment_from_html_tracked(self):
        """doc = HTML should add doc to html_derived_names."""
        code = """
doc = HTML
soup = BeautifulSoup(doc, "html.parser")
"""
        result = analyze_code_unified(code)
        assert "doc" in result.html_derived_names
        assert result.soup_created_with_html is True

    def test_html_processing_chain_tracked(self):
        """doc = HTML.replace(...) should track doc as HTML-derived."""
        code = """
cleaned = HTML.replace('&nbsp;', ' ')
soup = BeautifulSoup(cleaned, "html.parser")
"""
        result = analyze_code_unified(code)
        assert "cleaned" in result.html_derived_names
        assert result.soup_created_with_html is True


class TestUnifiedASTSelectionMethods:
    """Tests for selection method detection."""

    def test_find_detected(self):
        """soup.find() should set selection_method_used."""
        code = """
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
item = soup.find("div")
"""
        result = analyze_code_unified(code)
        assert result.selection_method_used is True

    def test_find_all_detected(self):
        """soup.find_all() should set selection_method_used."""
        code = """
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
items = soup.find_all("div")
"""
        result = analyze_code_unified(code)
        assert result.selection_method_used is True

    def test_select_detected(self):
        """soup.select() should set selection_method_used."""
        code = """
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
items = soup.select("div.class")
"""
        result = analyze_code_unified(code)
        assert result.selection_method_used is True


class TestUnifiedASTContentAccess:
    """Tests for content access detection."""

    def test_get_text_detected(self):
        """get_text() should set content_accessed."""
        code = """
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
text = soup.get_text()
"""
        result = analyze_code_unified(code)
        assert result.content_accessed is True

    def test_string_attr_detected(self):
        """element.string should set content_accessed."""
        code = """
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
text = soup.string
"""
        result = analyze_code_unified(code)
        assert result.content_accessed is True

    def test_text_attr_detected(self):
        """element.text should set content_accessed."""
        code = """
from bs4 import BeautifulSoup
soup = BeautifulSoup(HTML, "html.parser")
text = soup.text
"""
        result = analyze_code_unified(code)
        assert result.content_accessed is True


class TestUnifiedASTPerformance:
    """Ensure unified analysis is efficient."""

    def test_single_pass_efficiency(self):
        """Verify unified analysis doesn't require multiple AST traversals.

        This is a documentation test - the function should analyze everything
        in at most 2 passes (metadata collection + pattern detection).
        """
        code = """
from bs4 import BeautifulSoup as BS
html = HTML.replace('&nbsp;', ' ')
soup = BS(html, "html.parser")
items = soup.find_all("div", class_="item")
for item in items:
    text = item.get_text()
    print(text)
"""
        result = analyze_code_unified(code)

        # All flags should be set from a single call
        assert result.bs4_imported is True
        assert result.soup_created_with_html is True
        assert result.selection_method_used is True
        assert result.content_accessed is True

        # Metadata should be populated
        assert "BS" in result.bs4_ctor_aliases
        assert "html" in result.html_derived_names
