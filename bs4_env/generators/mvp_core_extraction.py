"""Core extraction archetypes for BeautifulSoup RL environment.

This module implements the fundamental extraction tasks that form the
foundation of BS4 proficiency.

Anti-shortcut measures:
- No semantic prefixes (target_id uses random base, not "target-")
- Randomized element positions (target not always in position 2)
- Varied tag types for targets (article, section, div, main, etc.)
- Near-duplicate decoy content to prevent text-matching shortcuts
- Attribute order randomization (when realistic chrome is used)
"""

from bs4_env.config import STRING_SCHEMA
from bs4_env.generators.base import (
    Generator,
    HtmlStyle,
    TaskInstance,
    add_decoy_elements,
    add_emoji_noise,
    add_noise_comments,
    generate_near_duplicate,
    # Semantic decoy functions
    generate_semantic_decoy,
    generate_similar_class,
    generate_similar_id,
    generate_variable_content,
    # Malformation function
    introduce_malformation,
    make_rng,
    random_class_for_style,
    random_class_name,
    random_id,
    # i18n functions
    random_mixed_language_content,
    wrap_with_realistic_chrome,
)
from bs4_env.registry import register

# Tags that can be used for target elements (prevents always using <article>)
TARGET_TAGS = ["article", "section", "div", "main", "aside", "p", "span"]

# Tags for distractor elements (varied to prevent pattern matching)
DISTRACTOR_TAGS = ["div", "aside", "section", "span", "p", "article"]


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
            complexity: "low", "medium", "high", or "realistic".

        Returns:
            TaskInstance with extraction task.
        """
        rng = make_rng(self.archetype_id, seed)

        # Select style randomly if not specified
        if style is None:
            style = rng.choice(list(HtmlStyle))

        # Always use realistic complexity for real-world difficulty
        # Real websites always have noise - training without it makes tasks too easy
        complexity = "realistic"

        # Generate the target content (ground truth comes FIRST)
        # Use variable-length content to prevent length-based shortcuts
        target_text = generate_variable_content(rng, min_sentences=1, max_sentences=5)

        # 20% chance of international content (mixed language or emoji)
        use_i18n = rng.random() < 0.2
        if use_i18n:
            i18n_type = rng.choice(["mixed", "emoji"])
            if i18n_type == "mixed":
                # Mix English with another language
                target_text = random_mixed_language_content(rng, base_sentences=3)
            else:
                # Add emoji to content
                target_text = add_emoji_noise(target_text, rng, density=0.15)

        # NO semantic prefixes - just random IDs
        target_id = random_id(rng)  # No prefix!

        # Generate SIMILAR IDs (harder decoys) - mix of similar and random
        distractor_ids = [
            generate_similar_id(rng, target_id),  # Similar to target
            generate_similar_id(rng, target_id),  # Another similar one
            random_id(rng),  # One truly random for variety
        ]

        # Generate SEMANTIC DECOY content (harder than random text)
        distractor_texts = [
            generate_semantic_decoy(rng, target_text, "partial_overlap"),  # Shares content
            generate_semantic_decoy(rng, target_text, "similar_topic"),  # Same length/tone
            generate_variable_content(rng, min_sentences=1, max_sentences=3),  # Random for variety
        ]

        # If using i18n, also add to some distractors for consistency
        if use_i18n and rng.random() < 0.5:
            idx = rng.randint(0, 2)
            if i18n_type == "emoji":
                distractor_texts[idx] = add_emoji_noise(distractor_texts[idx], rng, density=0.1)

        # Add near-duplicate decoy (almost the same as target - hardest decoy)
        decoy_text = generate_near_duplicate(rng, target_text)

        # Randomize tag types (target not always <article>)
        target_tag = rng.choice(TARGET_TAGS)
        distractor_tags = [rng.choice(DISTRACTOR_TAGS) for _ in range(3)]

        # Build elements list for shuffling (randomize position)
        content_class = random_class_for_style(rng, style, count=2)
        sidebar_class = random_class_for_style(rng, style, count=2)

        elements = [
            (target_id, target_text, target_tag, content_class, True),  # is_target=True
            (distractor_ids[0], distractor_texts[0], distractor_tags[0], sidebar_class, False),
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

        # Add near-duplicate decoy after shuffled elements
        decoy_class = random_class_for_style(rng, style, count=1)
        body_parts.append(f'<div class="related-content {decoy_class}">{decoy_text}</div>')

        body_content = "\n".join(body_parts)

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
            # Still shuffle position in simple mode
            html_parts = [
                "<!DOCTYPE html>",
                "<html>",
                "<head><title>Test Page</title></head>",
                "<body>",
                "<header>",
                f'<nav id="nav-{rng.randint(1000, 9999)}">Navigation</nav>',
                "</header>",
                "<main>",
                body_content,
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

        # Introduce malformed HTML 30% of the time (real websites are messy)
        is_malformed = rng.random() < 0.3
        if is_malformed:
            html = introduce_malformation(html, rng)

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
                "distractor_count": 3,
                "html_style": style.value,
                "use_realistic_chrome": use_realistic_chrome,
                "complexity": complexity,
                "is_malformed": is_malformed,
                "i18n_content": use_i18n,
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

    Anti-shortcut measures:
    - No semantic prefixes (class names use random bases)
    - Randomized element positions
    - Variable tag types
    - Variable content lengths
    """

    def generate(
        self,
        seed: int,
        style: HtmlStyle | None = None,
        complexity: str = "medium",
    ) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        # Select style randomly if not specified
        if style is None:
            style = rng.choice(list(HtmlStyle))

        # Always use realistic complexity for real-world difficulty
        complexity = "realistic"

        # Generate target content with variable length
        target_text = generate_variable_content(rng, min_sentences=1, max_sentences=4)

        # NO semantic prefixes - generate random class names
        target_class = random_class_name(rng)  # No prefix!

        # Generate SIMILAR class names (harder decoys) - mix of similar and random
        distractor_classes = [
            generate_similar_class(rng, target_class),  # Similar to target
            generate_similar_class(rng, target_class),  # Another similar one
            random_class_name(rng),  # One truly random for variety
        ]

        # Generate SEMANTIC DECOY content (harder than random text)
        distractor_texts = [
            generate_semantic_decoy(rng, target_text, "partial_overlap"),  # Shares content
            generate_semantic_decoy(rng, target_text, "similar_topic"),  # Same length/tone
            generate_variable_content(rng, min_sentences=1, max_sentences=3),  # Random for variety
        ]

        # Add near-duplicate decoy (almost the same as target)
        decoy_text = generate_near_duplicate(rng, target_text)

        # Randomize tag types
        target_tag = rng.choice(TARGET_TAGS)
        distractor_tags = [rng.choice(DISTRACTOR_TAGS) for _ in range(3)]

        # Build elements for shuffling (randomize position)
        elements = [
            (target_tag, target_class, target_text, True),  # is_target=True
            (distractor_tags[0], distractor_classes[0], distractor_texts[0], False),
            (distractor_tags[1], distractor_classes[1], distractor_texts[1], False),
            (distractor_tags[2], distractor_classes[2], distractor_texts[2], False),
        ]

        # Shuffle to randomize target position
        rng.shuffle(elements)

        # Build body content
        body_parts = []
        for tag, cls, text, _is_target in elements:
            body_parts.append(f'<{tag} class="{cls}">{text}</{tag}>')

        # Add near-duplicate decoy
        decoy_class = random_class_name(rng)
        body_parts.append(f'<div class="excerpt {decoy_class}">{decoy_text}</div>')

        body_content = "\n".join(body_parts)

        # Always wrap with realistic HTML chrome
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Content Page",
            complexity=complexity,
            include_nav=True,
            include_footer=True,
        )

        html = add_noise_comments(html, rng, count=2)

        # Introduce malformed HTML 30% of the time (real websites are messy)
        is_malformed = rng.random() < 0.3
        if is_malformed:
            html = introduce_malformation(html, rng)

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
                "target_tag": target_tag,
                "html_style": style.value,
                "complexity": complexity,
                "is_malformed": is_malformed,
                "gotcha": "Must use class_ not class in BS4",
            },
        )
