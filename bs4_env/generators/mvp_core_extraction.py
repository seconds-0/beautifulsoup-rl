"""Core extraction archetypes for BeautifulSoup RL environment.

This module implements the fundamental extraction tasks that form the
foundation of BS4 proficiency.
"""

from bs4_env.config import STRING_SCHEMA
from bs4_env.generators.base import (
    Generator,
    HtmlStyle,
    TaskInstance,
    make_rng,
    random_paragraph,
    random_id,
    random_class_name,
    random_class_for_style,
    add_noise_comments,
    add_decoy_elements,
    wrap_with_realistic_chrome,
)
from bs4_env.registry import register


@register(
    archetype_id="mvp.extract_text_by_id",
    category="core_extraction",
    difficulty="easy",
    solvable=True,
    description="Extract visible text from an element with a specific ID",
    tags=["extraction", "id", "text"],
    phase=1,
    answer_schema=STRING_SCHEMA,
)
class ExtractTextByIdGenerator(Generator):
    """Generate tasks to extract text from an element by ID.

    This is one of the most basic BS4 tasks: find an element by ID
    and extract its text content.

    Difficulty variations:
    - Easy: Clear ID, no distractions
    - Medium: Similar IDs exist, some noise
    - Hard: ID in nested structure, heavy noise

    The generator can optionally use realistic HTML chrome (navigation,
    footer, framework-specific patterns) to train on real-world complexity.
    """

    def generate(
        self,
        seed: int,
        style: HtmlStyle | None = None,
        use_realistic_chrome: bool = True,
        complexity: str = "medium",
    ) -> TaskInstance:
        """Generate a task instance.

        Args:
            seed: Random seed for deterministic generation.
            style: HTML framework style. If None, randomly selected.
            use_realistic_chrome: If True, wrap content with realistic HTML.
            complexity: "low", "medium", or "high" for head content richness.

        Returns:
            TaskInstance with extraction task.
        """
        rng = make_rng(self.archetype_id, seed)

        # Select style randomly if not specified
        if style is None:
            style = rng.choice(list(HtmlStyle))

        # Generate the target content (ground truth comes FIRST)
        target_text = random_paragraph(rng, sentences=rng.randint(1, 3))
        target_id = random_id(rng, prefix="target")

        # Generate distractor content
        distractor_texts = [random_paragraph(rng, sentences=1) for _ in range(3)]
        distractor_ids = [random_id(rng, prefix="other") for _ in range(3)]

        # Build core body content with distractors
        content_class = random_class_for_style(rng, style, count=2)
        sidebar_class = random_class_for_style(rng, style, count=2)

        body_content = f"""<div id="{distractor_ids[0]}" class="{sidebar_class}">{distractor_texts[0]}</div>
<article id="{target_id}" class="{content_class}">
{target_text}
</article>
<div id="{distractor_ids[1]}">{distractor_texts[1]}</div>
<aside id="{distractor_ids[2]}">{distractor_texts[2]}</aside>"""

        if use_realistic_chrome:
            # Wrap with full realistic HTML document
            html = wrap_with_realistic_chrome(
                body_content,
                style,
                rng,
                title="Content Page",
                complexity=complexity,
                include_nav=True,
                include_footer=True,
            )
        else:
            # Simple HTML structure (backward compatible)
            html_parts = [
                "<!DOCTYPE html>",
                "<html>",
                "<head><title>Test Page</title></head>",
                "<body>",
                "<header>",
                f'<nav id="{distractor_ids[0]}">{distractor_texts[0]}</nav>',
                "</header>",
                "<main>",
                f'<div id="{distractor_ids[1]}" class="sidebar">{distractor_texts[1]}</div>',
                f'<article id="{target_id}" class="content">',
                target_text,
                "</article>",
                f'<aside id="{distractor_ids[2]}">{distractor_texts[2]}</aside>',
                "</main>",
                "<footer>Footer content</footer>",
                "</body>",
                "</html>",
            ]
            html = "\n".join(html_parts)

        # Add noise comments
        html = add_noise_comments(html, rng, count=rng.randint(1, 3))

        if not use_realistic_chrome:
            # Only add decoy elements for simple mode
            html = add_decoy_elements(html, rng, count=rng.randint(0, 2))

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
                "distractor_count": 3,
                "html_style": style.value,
                "use_realistic_chrome": use_realistic_chrome,
                "complexity": complexity,
            },
        )


@register(
    archetype_id="mvp.extract_text_by_class",
    category="core_extraction",
    difficulty="easy",
    solvable=True,
    description="Extract text from an element with a specific CSS class",
    tags=["extraction", "class", "text"],
    phase=1,
    answer_schema=STRING_SCHEMA,
)
class ExtractTextByClassGenerator(Generator):
    """Generate tasks to extract text from an element by class.

    This tests the common pattern of finding elements by class,
    including the BS4 gotcha of using `class_` instead of `class`.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        # Generate target content
        target_text = random_paragraph(rng, sentences=rng.randint(1, 2))
        target_class = random_class_name(rng, prefix="highlight")

        # Generate distractors with similar-ish classes
        distractor_texts = [random_paragraph(rng, sentences=1) for _ in range(3)]
        distractor_classes = [
            random_class_name(rng, prefix="content"),
            random_class_name(rng, prefix="section"),
            random_class_name(rng, prefix="block"),
        ]

        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<body>",
            f'<div class="{distractor_classes[0]}">{distractor_texts[0]}</div>',
            f'<div class="{distractor_classes[1]}">{distractor_texts[1]}</div>',
            f'<span class="{target_class}">{target_text}</span>',
            f'<div class="{distractor_classes[2]}">{distractor_texts[2]}</div>',
            "</body>",
            "</html>",
        ]

        html = "\n".join(html_parts)
        html = add_noise_comments(html, rng, count=2)

        query = f'Extract the text content from the element with class="{target_class}".'

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
                "target_class": target_class,
                "gotcha": "Must use class_ not class in BS4",
            },
        )
