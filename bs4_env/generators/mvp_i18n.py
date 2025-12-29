"""International content archetypes for BeautifulSoup RL environment.

These tasks test parsing of non-English content, RTL text, emoji, and
special Unicode characters - common edge cases in real web scraping.

Skill categories tested:
- UTF-8 encoding handling
- get_text() with various scripts (Han, Cyrillic, Arabic, etc.)
- RTL text direction
- Emoji preservation
- Unicode normalization
"""

from bs4_env.config import STRING_SCHEMA
from bs4_env.generators.base import (
    Generator,
    HtmlStyle,
    TaskInstance,
    add_emoji_noise,
    add_noise_comments,
    add_special_unicode,
    generate_variable_i18n_content,
    make_rng,
    random_class_for_style,
    random_id,
    random_paragraph,
    wrap_with_realistic_chrome,
)
from bs4_env.registry import register

# Tags that can be used for target elements
TARGET_TAGS = ["article", "section", "div", "main", "aside", "p", "span"]


@register(
    archetype_id="mvp.extract_multilingual",
    category="i18n",
    difficulty="medium",
    solvable=True,
    description="Extract text from multilingual HTML content",
    tags=["extraction", "i18n", "unicode", "text"],
    phase=2,
    answer_schema=STRING_SCHEMA,
)
class MultilingualExtractionGenerator(Generator):
    """Generate tasks with non-English content.

    Tests:
    - UTF-8 encoding handling
    - get_text() with various scripts (Chinese, Japanese, Korean, Arabic, etc.)
    - Proper string comparison with Unicode
    - Mixed script content

    This is a fundamental test - if a model can't handle Unicode extraction,
    it will fail on a large portion of real-world websites.
    """

    def generate(
        self,
        seed: int,
        style: HtmlStyle | None = None,
        complexity: str = "medium",
    ) -> TaskInstance:
        """Generate a task instance.

        Args:
            seed: Random seed for deterministic generation.
            style: HTML framework style. If None, randomly selected.
            complexity: "low", "medium", "high", or "realistic".

        Returns:
            TaskInstance with multilingual extraction task.
        """
        rng = make_rng(self.archetype_id, seed)

        # Select style randomly if not specified
        if style is None:
            style = rng.choice(list(HtmlStyle))

        # Generate target content in a non-English language
        target_text, target_lang = generate_variable_i18n_content(
            rng, min_sentences=1, max_sentences=4
        )

        # Generate random IDs and classes (no semantic prefixes)
        target_id = random_id(rng)
        distractor_ids = [random_id(rng) for _ in range(3)]

        # Generate distractor content - mix of languages
        distractor_texts = []
        distractor_langs = []
        for _ in range(3):
            if rng.random() < 0.5:
                # 50% chance of same language as target
                text, lang = generate_variable_i18n_content(
                    rng, min_sentences=1, max_sentences=2, language=target_lang
                )
            else:
                # 50% chance of different language
                text, lang = generate_variable_i18n_content(rng, min_sentences=1, max_sentences=2)
            distractor_texts.append(text)
            distractor_langs.append(lang)

        # Randomize tag types
        target_tag = rng.choice(TARGET_TAGS)
        distractor_tags = [rng.choice(TARGET_TAGS) for _ in range(3)]

        # Build elements for shuffling (randomize position)
        content_class = random_class_for_style(rng, style, count=2)

        elements = [
            (target_id, target_text, target_tag, content_class, target_lang, True),
            (
                distractor_ids[0],
                distractor_texts[0],
                distractor_tags[0],
                "",
                distractor_langs[0],
                False,
            ),
            (
                distractor_ids[1],
                distractor_texts[1],
                distractor_tags[1],
                "",
                distractor_langs[1],
                False,
            ),
            (
                distractor_ids[2],
                distractor_texts[2],
                distractor_tags[2],
                "",
                distractor_langs[2],
                False,
            ),
        ]

        # Shuffle to randomize target position
        rng.shuffle(elements)

        # Build body content with lang attributes
        body_parts = []
        for elem_id, text, tag, cls, lang, _is_target in elements:
            class_attr = f' class="{cls}"' if cls else ""
            lang_attr = f' lang="{lang}"'
            body_parts.append(f'<{tag} id="{elem_id}"{class_attr}{lang_attr}>{text}</{tag}>')

        body_content = "\n".join(body_parts)

        # Build HTML
        if complexity == "realistic":
            html = wrap_with_realistic_chrome(
                body_content,
                style,
                rng,
                title="International Content",
                complexity=complexity,
                include_nav=True,
                include_footer=True,
            )
        else:
            html_parts = [
                "<!DOCTYPE html>",
                '<html lang="mul">',  # mul = multiple languages
                "<head>",
                '<meta charset="UTF-8">',
                "<title>International Page</title>",
                "</head>",
                "<body>",
                body_content,
                "</body>",
                "</html>",
            ]
            html = "\n".join(html_parts)

        html = add_noise_comments(html, rng, count=2)

        # Build query
        query = f'Extract the text content from the element with id="{target_id}".'

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=target_text,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema=STRING_SCHEMA,
            normalization={
                "strip_whitespace": True,
                "collapse_whitespace": True,
                "unicode_nfc": True,  # Critical for Unicode comparison
            },
            metadata={
                "target_id": target_id,
                "target_tag": target_tag,
                "target_language": target_lang,
                "html_style": style.value,
                "complexity": complexity,
            },
        )


@register(
    archetype_id="mvp.extract_rtl",
    category="i18n",
    difficulty="medium",
    solvable=True,
    description="Extract text from RTL (right-to-left) content",
    tags=["extraction", "i18n", "rtl", "arabic", "hebrew"],
    phase=2,
    answer_schema=STRING_SCHEMA,
)
class RTLExtractionGenerator(Generator):
    """Generate tasks with Arabic/Hebrew RTL content.

    Tests:
    - RTL text direction handling
    - Mixed LTR/RTL content
    - dir="rtl" attribute recognition
    - Bidirectional text algorithms

    RTL content is common on Middle Eastern websites and in international
    e-commerce. Proper handling requires understanding Unicode BiDi.
    """

    def generate(
        self,
        seed: int,
        style: HtmlStyle | None = None,
        complexity: str = "medium",
    ) -> TaskInstance:
        """Generate a task instance with RTL content.

        Args:
            seed: Random seed for deterministic generation.
            style: HTML framework style. If None, randomly selected.
            complexity: "low", "medium", "high", or "realistic".

        Returns:
            TaskInstance with RTL extraction task.
        """
        rng = make_rng(self.archetype_id, seed)

        # Select style randomly if not specified
        if style is None:
            style = rng.choice(list(HtmlStyle))

        # Generate target content in an RTL language (Arabic or Hebrew)
        rtl_lang = rng.choice(["ar", "he"])
        target_text, target_lang = generate_variable_i18n_content(
            rng, min_sentences=1, max_sentences=3, language=rtl_lang
        )

        # Generate random IDs
        target_id = random_id(rng)
        distractor_ids = [random_id(rng) for _ in range(3)]

        # Generate distractor content - mix of RTL and LTR
        distractor_texts = []
        distractor_langs = []
        distractor_dirs = []
        for i in range(3):
            if i == 0:
                # First distractor is also RTL (same language)
                text, lang = generate_variable_i18n_content(
                    rng, min_sentences=1, max_sentences=2, language=rtl_lang
                )
                direction = "rtl"
            elif i == 1:
                # Second distractor is LTR (different language)
                text, lang = generate_variable_i18n_content(
                    rng, min_sentences=1, max_sentences=2, language=rng.choice(["zh", "ja", "ko"])
                )
                direction = "ltr"
            else:
                # Third distractor is English
                text = random_paragraph(rng, sentences=rng.randint(1, 2))
                lang = "en"
                direction = "ltr"
            distractor_texts.append(text)
            distractor_langs.append(lang)
            distractor_dirs.append(direction)

        # Randomize tag types
        target_tag = rng.choice(TARGET_TAGS)
        distractor_tags = [rng.choice(TARGET_TAGS) for _ in range(3)]

        # Build elements for shuffling
        content_class = random_class_for_style(rng, style, count=2)

        elements = [
            (target_id, target_text, target_tag, content_class, target_lang, "rtl", True),
            (
                distractor_ids[0],
                distractor_texts[0],
                distractor_tags[0],
                "",
                distractor_langs[0],
                distractor_dirs[0],
                False,
            ),
            (
                distractor_ids[1],
                distractor_texts[1],
                distractor_tags[1],
                "",
                distractor_langs[1],
                distractor_dirs[1],
                False,
            ),
            (
                distractor_ids[2],
                distractor_texts[2],
                distractor_tags[2],
                "",
                distractor_langs[2],
                distractor_dirs[2],
                False,
            ),
        ]

        # Shuffle to randomize target position
        rng.shuffle(elements)

        # Build body content with dir and lang attributes
        body_parts = []
        for elem_id, text, tag, cls, lang, direction, _is_target in elements:
            class_attr = f' class="{cls}"' if cls else ""
            lang_attr = f' lang="{lang}"'
            dir_attr = f' dir="{direction}"'
            body_parts.append(
                f'<{tag} id="{elem_id}"{class_attr}{lang_attr}{dir_attr}>{text}</{tag}>'
            )

        body_content = "\n".join(body_parts)

        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '<meta charset="UTF-8">',
            "<title>RTL Content Page</title>",
            "</head>",
            "<body>",
            body_content,
            "</body>",
            "</html>",
        ]
        html = "\n".join(html_parts)

        html = add_noise_comments(html, rng, count=2)

        # Build query
        query = f'Extract the text content from the element with id="{target_id}".'

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=target_text,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema=STRING_SCHEMA,
            normalization={
                "strip_whitespace": True,
                "collapse_whitespace": True,
                "unicode_nfc": True,
            },
            metadata={
                "target_id": target_id,
                "target_tag": target_tag,
                "target_language": target_lang,
                "text_direction": "rtl",
                "html_style": style.value,
            },
        )


@register(
    archetype_id="mvp.extract_emoji_content",
    category="i18n",
    difficulty="easy",
    solvable=True,
    description="Extract text containing emoji and special characters",
    tags=["extraction", "emoji", "unicode", "special_chars"],
    phase=2,
    answer_schema=STRING_SCHEMA,
)
class EmojiContentGenerator(Generator):
    """Generate tasks with emoji-rich content.

    Tests:
    - Emoji preservation in extraction
    - get_text() with emoji sequences (including ZWJ sequences)
    - Mixed text and emoji
    - Special Unicode symbols (currency, math, etc.)

    Emoji are ubiquitous in modern web content - reviews, social media,
    product descriptions. Models must handle them correctly.
    """

    def generate(
        self,
        seed: int,
        style: HtmlStyle | None = None,
        complexity: str = "medium",
    ) -> TaskInstance:
        """Generate a task instance with emoji content.

        Args:
            seed: Random seed for deterministic generation.
            style: HTML framework style. If None, randomly selected.
            complexity: "low", "medium", "high", or "realistic".

        Returns:
            TaskInstance with emoji extraction task.
        """
        rng = make_rng(self.archetype_id, seed)

        # Select style randomly if not specified
        if style is None:
            style = rng.choice(list(HtmlStyle))

        # Generate base English content
        base_text = random_paragraph(rng, sentences=rng.randint(2, 4))

        # Add emoji noise (higher density than usual for this task)
        target_text = add_emoji_noise(
            base_text,
            rng,
            density=0.3,
            category=rng.choice(["positive", "commerce", "navigation", "status", "social"]),
        )

        # Optionally add special Unicode characters
        if rng.random() < 0.3:
            target_text = add_special_unicode(target_text, rng, density=0.02, category="currency")

        # Generate random IDs
        target_id = random_id(rng)
        distractor_ids = [random_id(rng) for _ in range(3)]

        # Generate distractor content with emoji
        distractor_texts = []
        for _ in range(3):
            base = random_paragraph(rng, sentences=rng.randint(1, 2))
            with_emoji = add_emoji_noise(base, rng, density=0.2)
            distractor_texts.append(with_emoji)

        # Randomize tag types
        target_tag = rng.choice(TARGET_TAGS)
        distractor_tags = [rng.choice(TARGET_TAGS) for _ in range(3)]

        # Build elements for shuffling
        content_class = random_class_for_style(rng, style, count=2)

        elements = [
            (target_id, target_text, target_tag, content_class, True),
            (distractor_ids[0], distractor_texts[0], distractor_tags[0], "", False),
            (distractor_ids[1], distractor_texts[1], distractor_tags[1], "", False),
            (distractor_ids[2], distractor_texts[2], distractor_tags[2], "", False),
        ]

        # Shuffle to randomize target position
        rng.shuffle(elements)

        # Build body content
        body_parts = []
        for elem_id, text, tag, cls, _is_target in elements:
            class_attr = f' class="{cls}"' if cls else ""
            body_parts.append(f'<{tag} id="{elem_id}"{class_attr}>{text}</{tag}>')

        body_content = "\n".join(body_parts)

        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '<meta charset="UTF-8">',
            "<title>Content Page</title>",
            "</head>",
            "<body>",
            body_content,
            "</body>",
            "</html>",
        ]
        html = "\n".join(html_parts)

        html = add_noise_comments(html, rng, count=2)

        # Build query
        query = f'Extract the text content from the element with id="{target_id}".'

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=target_text,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema=STRING_SCHEMA,
            normalization={
                "strip_whitespace": True,
                "collapse_whitespace": True,
                "unicode_nfc": True,
            },
            metadata={
                "target_id": target_id,
                "target_tag": target_tag,
                "has_emoji": True,
                "html_style": style.value,
            },
        )
