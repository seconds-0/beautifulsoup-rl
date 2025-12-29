"""Error bait archetypes for BeautifulSoup RL environment.

These tasks test common BS4 gotchas and pitfalls that trip up developers.
They are SOLVABLE but require awareness of BS4's quirks.

Anti-shortcut measures:
- No semantic prefixes (IDs/classes use random bases)
- Randomized element positions
- Variable content lengths
"""

from bs4_env.config import STRING_SCHEMA
from bs4_env.generators.base import (
    Generator,
    HtmlStyle,
    TaskInstance,
    make_rng,
    random_paragraph,
    generate_variable_content,
    random_id,
    random_class_name,
    add_noise_comments,
    wrap_with_realistic_chrome,
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

        # Select random HTML style for realistic variation
        style = rng.choice(list(HtmlStyle))

        # Generate content that will be split across multiple child elements
        # Use variable-length content
        text_part1 = generate_variable_content(rng, min_sentences=1, max_sentences=2)
        text_part2 = generate_variable_content(rng, min_sentences=1, max_sentences=2)
        text_part3 = generate_variable_content(rng, min_sentences=1, max_sentences=2)

        # Full expected text (what get_text() returns)
        full_text = f"{text_part1} {text_part2} {text_part3}"

        # NO semantic prefixes - use random IDs
        target_id = random_id(rng)  # No prefix!

        # Random classes for child elements
        child_classes = [random_class_name(rng) for _ in range(3)]

        # Build body content with nested structure that breaks .string
        body_content = f"""<article id="{target_id}">
    <span class="{child_classes[0]}">{text_part1}</span>
    <em class="{child_classes[1]}">{text_part2}</em>
    <span class="{child_classes[2]}">{text_part3}</span>
</article>"""

        # Wrap with realistic chrome for real-world difficulty
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Article Page",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

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

        # Select random HTML style for realistic variation
        style = rng.choice(list(HtmlStyle))

        # Target element that DOES exist - variable-length content
        target_text = generate_variable_content(rng, min_sentences=1, max_sentences=3)

        # NO semantic prefixes - use random class names
        target_class = random_class_name(rng)

        # Element that looks similar but with different class
        decoy_text = generate_variable_content(rng, min_sentences=1, max_sentences=2)
        decoy_class = random_class_name(rng)

        # Another distractor
        other_text = generate_variable_content(rng, min_sentences=1, max_sentences=2)
        other_class = random_class_name(rng)

        # Build elements for shuffling (randomize positions)
        elements = [
            (decoy_class, decoy_text),
            (target_class, target_text),
            (other_class, other_text),
        ]
        rng.shuffle(elements)

        body_parts = [f'<div class="{cls}"><p>{text}</p></div>' for cls, text in elements]
        body_content = "\n".join(body_parts)

        # Wrap with realistic chrome for real-world difficulty
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Content Page",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

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

        # Select random HTML style for realistic variation
        style = rng.choice(list(HtmlStyle))

        # Variable-length content
        target_text = generate_variable_content(rng, min_sentences=1, max_sentences=3)

        # NO semantic prefixes - use random class name
        target_class = random_class_name(rng)

        # Generate distractor paragraphs with random classes
        distractor_texts = [
            generate_variable_content(rng, min_sentences=1, max_sentences=2)
            for _ in range(2)
        ]
        distractor_classes = [random_class_name(rng) for _ in range(2)]

        # Build elements for shuffling (randomize positions)
        elements = [
            (distractor_classes[0], distractor_texts[0]),
            (target_class, target_text),
            (distractor_classes[1], distractor_texts[1]),
        ]
        rng.shuffle(elements)

        body_parts = [f'<p class="{cls}">{text}</p>' for cls, text in elements]
        body_content = "\n".join(body_parts)

        # Wrap with realistic chrome for real-world difficulty
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Content Page",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

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
