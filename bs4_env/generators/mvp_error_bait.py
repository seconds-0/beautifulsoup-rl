"""Error bait archetypes for BeautifulSoup RL environment.

These tasks test common BS4 gotchas and pitfalls that trip up developers.
They are SOLVABLE but require awareness of BS4's quirks.
"""

from bs4_env.config import STRING_SCHEMA
from bs4_env.generators.base import (
    Generator,
    TaskInstance,
    make_rng,
    random_paragraph,
    random_id,
    add_noise_comments,
)
from bs4_env.registry import register


@register(
    archetype_id="mvp.string_returns_none",
    category="error_bait",
    difficulty="medium",
    solvable=True,
    description="Element has multiple children so .string returns None",
    tags=["gotcha", "string", "get_text"],
    phase=1,
    answer_schema=STRING_SCHEMA,
)
class StringReturnsNoneGenerator(Generator):
    """Generate tasks where .string returns None.

    In BeautifulSoup, element.string returns None when the element
    has multiple children. The correct approach is to use get_text().

    This is a common gotcha that causes NoneType errors.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        # Generate content that will be split across multiple child elements
        text_part1 = random_paragraph(rng, sentences=1)
        text_part2 = random_paragraph(rng, sentences=1)
        text_part3 = random_paragraph(rng, sentences=1)

        # Full expected text (what get_text() returns)
        full_text = f"{text_part1} {text_part2} {text_part3}"

        target_id = random_id(rng, prefix="content")

        # Build HTML with nested structure that breaks .string
        html = f"""<!DOCTYPE html>
<html>
<head><title>Article</title></head>
<body>
<article id="{target_id}">
    <span class="intro">{text_part1}</span>
    <em class="highlight">{text_part2}</em>
    <span class="conclusion">{text_part3}</span>
</article>
</body>
</html>"""

        html = add_noise_comments(html, rng, count=1)

        query = f'Extract all the text content from the element with id="{target_id}".'

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=full_text,
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
                "gotcha": ".string returns None due to multiple children",
                "solution": "Use get_text() instead of .string",
                "child_count": 3,
            },
        )


@register(
    archetype_id="mvp.none_attribute_error",
    category="error_bait",
    difficulty="easy",
    solvable=True,
    description="Must check for None before accessing attributes",
    tags=["gotcha", "none", "attribute"],
    phase=1,
    answer_schema=STRING_SCHEMA,
)
class NoneAttributeErrorGenerator(Generator):
    """Generate tasks that require None checking.

    The task asks to find an element that exists, but similar elements
    don't exist. Naive code that doesn't check for None will crash.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        # Target element that DOES exist
        target_text = random_paragraph(rng, sentences=1)
        target_class = "target-content"

        # Element that looks similar but with different class
        decoy_text = random_paragraph(rng, sentences=1)
        decoy_class = "similar-content"

        html = f"""<!DOCTYPE html>
<html>
<body>
<div class="{decoy_class}">
    <p>{decoy_text}</p>
</div>
<div class="{target_class}">
    <p>{target_text}</p>
</div>
<div class="other-content">
    <p>Some other text</p>
</div>
</body>
</html>"""

        query = f'Extract the text from the paragraph inside the div with class="{target_class}".'

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
            },
            metadata={
                "target_class": target_class,
                "gotcha": "find() returns None if not found - must check before accessing",
                "decoy_class": decoy_class,
            },
        )


@register(
    archetype_id="mvp.class_reserved_word",
    category="error_bait",
    difficulty="easy",
    solvable=True,
    description="Must use class_ instead of class (reserved word)",
    tags=["gotcha", "class", "reserved"],
    phase=1,
    answer_schema=STRING_SCHEMA,
)
class ClassReservedWordGenerator(Generator):
    """Generate tasks that require using class_ parameter.

    In Python, 'class' is a reserved word. BeautifulSoup requires
    using 'class_' or attrs={'class': ...} instead.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        target_text = random_paragraph(rng, sentences=1)
        target_class = f"highlight-{rng.randint(1, 100)}"

        # Mix of classes to make it non-trivial
        html = f"""<!DOCTYPE html>
<html>
<body>
<p class="intro">Introduction paragraph.</p>
<p class="{target_class}">{target_text}</p>
<p class="outro">Conclusion paragraph.</p>
</body>
</html>"""

        query = f'Extract the text from the <p> element with class="{target_class}".'

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
            },
            metadata={
                "target_class": target_class,
                "gotcha": "Cannot use class= in find(), must use class_= or attrs={'class':...}",
            },
        )
