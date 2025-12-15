"""Core extraction archetypes for BeautifulSoup RL environment.

This module implements the fundamental extraction tasks that form the
foundation of BS4 proficiency.
"""

from bs4_env.config import STRING_SCHEMA
from bs4_env.generators.base import (
    Generator,
    TaskInstance,
    make_rng,
    random_paragraph,
    random_id,
    random_class_name,
    add_noise_comments,
    add_decoy_elements,
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
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        # Generate the target content (ground truth comes FIRST)
        target_text = random_paragraph(rng, sentences=rng.randint(1, 3))
        target_id = random_id(rng, prefix="target")

        # Generate distractor content
        distractor_texts = [random_paragraph(rng, sentences=1) for _ in range(3)]
        distractor_ids = [random_id(rng, prefix="other") for _ in range(3)]

        # Build HTML structure
        html_parts = ["<!DOCTYPE html>", "<html>", "<head><title>Test Page</title></head>", "<body>"]

        # Add header with navigation (distraction)
        html_parts.append("<header>")
        html_parts.append(f'<nav id="{distractor_ids[0]}">{distractor_texts[0]}</nav>')
        html_parts.append("</header>")

        # Main content area
        html_parts.append("<main>")
        html_parts.append(f'<div id="{distractor_ids[1]}" class="sidebar">{distractor_texts[1]}</div>')

        # Target element
        html_parts.append(f'<article id="{target_id}" class="content">')
        html_parts.append(target_text)
        html_parts.append("</article>")

        html_parts.append(f'<aside id="{distractor_ids[2]}">{distractor_texts[2]}</aside>')
        html_parts.append("</main>")

        # Footer
        html_parts.append("<footer>Footer content</footer>")
        html_parts.append("</body>")
        html_parts.append("</html>")

        html = "\n".join(html_parts)

        # Add noise
        html = add_noise_comments(html, rng, count=rng.randint(1, 3))
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
