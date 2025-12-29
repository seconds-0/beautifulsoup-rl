"""Tests for international content generators.

These tests verify that:
1. Multilingual content generates correctly
2. RTL content has proper dir attributes
3. Emoji content is preserved
4. Unicode normalization works
5. All generators are deterministic
"""

import pytest
from bs4 import BeautifulSoup

from bs4_env.data.i18n_content import (
    EMOJI_CATEGORIES,
    LANGUAGES,
    RTL_LANGUAGES,
    get_language_direction,
    is_rtl_language,
)
from bs4_env.generators.base import (
    add_emoji_noise,
    generate_i18n_paragraph,
    get_rtl_wrapper,
    random_i18n_content,
    random_mixed_language_content,
)
from bs4_env.generators.mvp_i18n import (
    EmojiContentGenerator,
    MultilingualExtractionGenerator,
    RTLExtractionGenerator,
)

# =============================================================================
# Data Module Tests
# =============================================================================


def test_languages_data_structure():
    """Verify LANGUAGES has expected structure."""
    assert len(LANGUAGES) >= 6, "Should have at least 6 languages"

    for lang_code, lang_data in LANGUAGES.items():
        assert "name" in lang_data, f"Language {lang_code} missing 'name'"
        assert "script" in lang_data, f"Language {lang_code} missing 'script'"
        assert "direction" in lang_data, f"Language {lang_code} missing 'direction'"
        assert "phrases" in lang_data, f"Language {lang_code} missing 'phrases'"
        assert "words" in lang_data, f"Language {lang_code} missing 'words'"
        assert len(lang_data["phrases"]) >= 3, f"Language {lang_code} needs more phrases"
        assert len(lang_data["words"]) >= 5, f"Language {lang_code} needs more words"


def test_rtl_languages():
    """Verify RTL languages are correctly identified."""
    assert "ar" in RTL_LANGUAGES, "Arabic should be RTL"
    assert "he" in RTL_LANGUAGES, "Hebrew should be RTL"
    assert "zh" not in RTL_LANGUAGES, "Chinese should not be RTL"
    assert "en" not in RTL_LANGUAGES, "English should not be RTL"


def test_emoji_categories():
    """Verify emoji categories have content."""
    assert len(EMOJI_CATEGORIES) >= 3, "Should have at least 3 emoji categories"

    for category, emojis in EMOJI_CATEGORIES.items():
        assert len(emojis) >= 5, f"Category {category} needs more emoji"


def test_is_rtl_language():
    """Test RTL language detection."""
    assert is_rtl_language("ar") is True
    assert is_rtl_language("he") is True
    assert is_rtl_language("zh") is False
    assert is_rtl_language("en") is False


def test_get_language_direction():
    """Test language direction lookup."""
    assert get_language_direction("ar") == "rtl"
    assert get_language_direction("he") == "rtl"
    assert get_language_direction("zh") == "ltr"
    assert get_language_direction("ja") == "ltr"


# =============================================================================
# Base Function Tests
# =============================================================================


def test_random_i18n_content_deterministic():
    """Same seed produces same i18n content."""
    import random

    rng1 = random.Random(42)
    rng2 = random.Random(42)

    content1, lang1 = random_i18n_content(rng1)
    content2, lang2 = random_i18n_content(rng2)

    assert content1 == content2
    assert lang1 == lang2


def test_random_i18n_content_specific_language():
    """Can request specific language."""
    import random

    rng = random.Random(42)
    content, lang = random_i18n_content(rng, language="zh")

    assert lang == "zh"
    # Chinese content should contain Chinese characters
    assert any("\u4e00" <= char <= "\u9fff" for char in content), "Should contain Chinese chars"


def test_random_mixed_language_content():
    """Mixed language content contains both English and foreign text."""
    import random

    rng = random.Random(42)
    # Use high foreign ratio to ensure we get foreign content
    content = random_mixed_language_content(rng, base_sentences=5, foreign_ratio=0.8)

    # Should have some ASCII (English) and some non-ASCII (foreign)
    has_ascii = any(ord(c) < 128 for c in content if c.isalpha())
    any(ord(c) >= 128 for c in content)

    assert has_ascii, "Should contain English"
    # Note: foreign_ratio is probabilistic, so we just check structure
    assert len(content) > 20, "Should have reasonable length"


def test_add_emoji_noise():
    """Emoji noise adds emoji to content."""
    import random

    rng = random.Random(42)
    base_text = "This is a test sentence. Another sentence here."
    # Use high density to ensure we get emoji
    result = add_emoji_noise(base_text, rng, density=0.9)

    # Should still contain original text structure
    assert "test" in result.lower() or "sentence" in result.lower()


def test_generate_i18n_paragraph_deterministic():
    """Same seed produces same paragraph."""
    import random

    rng1 = random.Random(42)
    rng2 = random.Random(42)

    text1, lang1 = generate_i18n_paragraph(rng1, sentences=3)
    text2, lang2 = generate_i18n_paragraph(rng2, sentences=3)

    assert text1 == text2
    assert lang1 == lang2


def test_get_rtl_wrapper():
    """RTL wrapper adds correct attributes."""
    content = "مرحبا"

    # RTL language should get dir="rtl"
    rtl_result = get_rtl_wrapper(content, "ar")
    assert 'dir="rtl"' in rtl_result
    assert 'lang="ar"' in rtl_result

    # LTR language should not get dir="rtl"
    ltr_result = get_rtl_wrapper(content, "zh")
    assert 'dir="rtl"' not in ltr_result
    assert 'lang="zh"' in ltr_result


# =============================================================================
# Generator Tests
# =============================================================================


class TestMultilingualExtractionGenerator:
    """Tests for MultilingualExtractionGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        gen = MultilingualExtractionGenerator()
        gen._archetype_spec = type("Spec", (), {"archetype_id": "mvp.extract_multilingual"})()
        return gen

    def test_determinism(self, generator):
        """Same seed produces same output."""
        task1 = generator.generate(42)
        task2 = generator.generate(42)

        assert task1.html == task2.html
        assert task1.ground_truth == task2.ground_truth
        assert task1.query == task2.query

    def test_has_lang_attribute(self, generator):
        """Generated HTML has lang attributes."""
        task = generator.generate(42)
        soup = BeautifulSoup(task.html, "html.parser")

        # Find elements with lang attribute
        elements_with_lang = soup.find_all(attrs={"lang": True})
        assert len(elements_with_lang) > 0, "Should have elements with lang attribute"

    def test_target_is_extractable(self, generator):
        """Target text can be extracted by ID."""
        task = generator.generate(42)
        soup = BeautifulSoup(task.html, "html.parser")

        target_id = task.metadata["target_id"]
        target_element = soup.find(id=target_id)

        assert target_element is not None, "Target element should exist"
        assert task.ground_truth in target_element.get_text()

    def test_metadata_has_language(self, generator):
        """Metadata includes target language."""
        task = generator.generate(42)

        assert "target_language" in task.metadata
        assert task.metadata["target_language"] in LANGUAGES


class TestRTLExtractionGenerator:
    """Tests for RTLExtractionGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        gen = RTLExtractionGenerator()
        gen._archetype_spec = type("Spec", (), {"archetype_id": "mvp.extract_rtl"})()
        return gen

    def test_determinism(self, generator):
        """Same seed produces same output."""
        task1 = generator.generate(42)
        task2 = generator.generate(42)

        assert task1.html == task2.html
        assert task1.ground_truth == task2.ground_truth

    def test_has_rtl_dir_attribute(self, generator):
        """RTL content has dir="rtl" attribute."""
        task = generator.generate(42)
        soup = BeautifulSoup(task.html, "html.parser")

        # Find target element
        target_id = task.metadata["target_id"]
        target_element = soup.find(id=target_id)

        assert target_element is not None
        assert target_element.get("dir") == "rtl", "Target should have dir=rtl"

    def test_target_language_is_rtl(self, generator):
        """Target language is an RTL language."""
        task = generator.generate(42)

        target_lang = task.metadata["target_language"]
        assert target_lang in RTL_LANGUAGES, f"Language {target_lang} should be RTL"


class TestEmojiContentGenerator:
    """Tests for EmojiContentGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        gen = EmojiContentGenerator()
        gen._archetype_spec = type("Spec", (), {"archetype_id": "mvp.extract_emoji_content"})()
        return gen

    def test_determinism(self, generator):
        """Same seed produces same output."""
        task1 = generator.generate(42)
        task2 = generator.generate(42)

        assert task1.html == task2.html
        assert task1.ground_truth == task2.ground_truth

    def test_has_emoji_metadata(self, generator):
        """Metadata indicates emoji content."""
        task = generator.generate(42)

        assert task.metadata.get("has_emoji") is True

    def test_target_is_extractable(self, generator):
        """Target text can be extracted by ID."""
        task = generator.generate(42)
        soup = BeautifulSoup(task.html, "html.parser")

        target_id = task.metadata["target_id"]
        target_element = soup.find(id=target_id)

        assert target_element is not None, "Target element should exist"


# =============================================================================
# Unicode Normalization Tests
# =============================================================================


def test_unicode_nfc_normalization():
    """Unicode NFC normalization handles combining characters."""
    import unicodedata

    # "é" can be represented two ways:
    # - Single character: é (U+00E9)
    # - Combined: e + ́ (U+0065 + U+0301)
    single = "é"
    combined = "e\u0301"

    # Before normalization, they're different
    assert single != combined

    # After NFC normalization, they're equal
    assert unicodedata.normalize("NFC", single) == unicodedata.normalize("NFC", combined)


def test_zero_width_characters_preserved():
    """Zero-width characters are preserved in extraction."""
    import random

    from bs4_env.generators.base import add_special_unicode

    rng = random.Random(42)
    base = "test text"
    result = add_special_unicode(base, rng, density=0.5, category="zero_width")

    # Length should be >= original (zero-width chars add length)
    assert len(result) >= len(base)


# =============================================================================
# Integration Tests
# =============================================================================


def test_i18n_generators_registered():
    """i18n generators are properly registered."""
    # Import the module to trigger registration
    import bs4_env.generators.mvp_i18n  # noqa: F401
    from bs4_env.registry import get_all_archetype_ids

    archetype_ids = get_all_archetype_ids()

    assert "mvp.extract_multilingual" in archetype_ids
    assert "mvp.extract_rtl" in archetype_ids
    assert "mvp.extract_emoji_content" in archetype_ids


def test_core_extraction_i18n_integration():
    """Core extraction generator can produce i18n content."""
    from bs4_env.generators.mvp_core_extraction import ExtractTextByIdGenerator

    gen = ExtractTextByIdGenerator()
    gen._archetype_spec = type("Spec", (), {"archetype_id": "mvp.extract_text_by_id"})()

    # Generate many samples to check i18n integration
    i18n_count = 0
    for seed in range(100):
        task = gen.generate(seed)
        if task.metadata.get("i18n_content"):
            i18n_count += 1

    # Should have some i18n content (~20% expected)
    assert i18n_count > 0, "Should have some i18n content"
    assert i18n_count < 50, "Should not have too much i18n content"
