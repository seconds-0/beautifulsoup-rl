"""NavigableString attribute error archetype for BeautifulSoup RL environment.

This module tests the BS4 gotcha where NavigableStrings (text nodes) don't
have the same attributes as Tag objects:

    result = soup.find(string="Some text")
    result.name  # AttributeError! NavigableStrings don't have .name

Key gotchas:
1. find(string=...) returns NavigableString, not Tag
2. NavigableStrings have no .name, .attrs, .find(), etc.
3. Comments are also NavigableStrings (and match string searches!)
4. Use .parent to get the containing Tag

Correct patterns:
    # To get the parent tag of text:
    text_node = soup.find(string="Some text")
    parent_tag = text_node.parent
    parent_tag.name  # This works

    # To check if result is a Tag:
    from bs4 import Tag
    if isinstance(result, Tag):
        result.name  # Safe
"""

from bs4_env.config import STRING_SCHEMA
from bs4_env.generators.base import (
    Generator,
    HtmlStyle,
    TaskInstance,
    make_rng,
    generate_variable_content,
    random_id,
    random_class_name,
    wrap_with_realistic_chrome,
    add_noise_comments,
)
from bs4_env.registry import register


@register(
    archetype_id="mvp.navigablestring_parent",
    category="bs4_gotchas",
    difficulty="medium",
    solvable=True,
    description="Find text content and get info about its parent tag (NavigableString has no .name)",
    tags=["navigation", "text", "gotcha", "navigablestring", "parent"],
    phase=1,
    answer_schema=STRING_SCHEMA,
)
class NavigableStringParentGenerator(Generator):
    """Generate tasks testing NavigableString vs Tag distinction.

    This tests the common mistake where developers:
    1. Use find(string=...) expecting a Tag
    2. Try to access .name or .attrs on the result
    3. Get AttributeError because NavigableStrings don't have these

    The task asks to find specific text AND get information about its
    parent element, requiring use of .parent to navigate from text to tag.
    """

    def generate(
        self,
        seed: int,
        style: HtmlStyle | None = None,
    ) -> TaskInstance:
        """Generate a task instance.

        Args:
            seed: Random seed for deterministic generation.
            style: HTML framework style. If None, randomly selected.

        Returns:
            TaskInstance with NavigableString navigation task.
        """
        rng = make_rng(self.archetype_id, seed)

        # Select style randomly if not specified
        if style is None:
            style = rng.choice(list(HtmlStyle))

        # Generate a UNIQUE target text that we'll search for
        # Make it specific enough to be unambiguous
        target_markers = [
            "MARKER-ALPHA",
            "MARKER-BETA",
            "MARKER-GAMMA",
            "UNIQUE-TEXT-001",
            "UNIQUE-TEXT-002",
            "SPECIAL-CONTENT-X",
        ]
        target_text = rng.choice(target_markers) + f"-{rng.randint(1000, 9999)}"

        # Choose the parent tag and its attributes
        parent_tags = ["span", "div", "p", "strong", "em", "a", "li", "td"]
        parent_tag = rng.choice(parent_tags)
        parent_id = random_id(rng)
        parent_class = random_class_name(rng)

        # What to extract - the parent's tag name OR class OR id
        extract_type = rng.choice(["tag_name", "class", "id"])

        if extract_type == "tag_name":
            ground_truth = parent_tag
            query_suffix = "what HTML tag contains it (the parent tag name)"
        elif extract_type == "class":
            ground_truth = parent_class
            query_suffix = "the class attribute of the element containing it"
        else:  # id
            ground_truth = parent_id
            query_suffix = "the id attribute of the element containing it"

        # Build the target element
        target_element = f'<{parent_tag} id="{parent_id}" class="{parent_class}">{target_text}</{parent_tag}>'

        # Add distractors - similar text in different elements
        distractors = []
        for i in range(3):
            d_tag = rng.choice(parent_tags)
            d_id = random_id(rng)
            d_class = random_class_name(rng)
            # Similar but different text
            d_text = f"NOT-{target_text[:10]}-{rng.randint(100, 999)}"
            distractors.append(f'<{d_tag} id="{d_id}" class="{d_class}">{d_text}</{d_tag}>')

        # Add an HTML comment that contains the target text (extra gotcha!)
        # Comments are NavigableStrings too and can match string searches
        comment = f"<!-- {target_text} appears here in a comment -->"

        # Build body content with shuffled elements
        all_elements = [target_element] + distractors
        rng.shuffle(all_elements)

        # Insert comment somewhere
        insert_pos = rng.randint(0, len(all_elements))
        all_elements.insert(insert_pos, comment)

        # Add some surrounding context
        intro = generate_variable_content(rng, min_sentences=1, max_sentences=2)
        outro = generate_variable_content(rng, min_sentences=1, max_sentences=1)

        body_content = f"""
<div class="content-wrapper">
    <p class="intro">{intro}</p>
    <div class="items">
        {chr(10).join(all_elements)}
    </div>
    <p class="outro">{outro}</p>
</div>
"""

        # Wrap with realistic chrome
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Content Page",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

        html = add_noise_comments(html, rng, count=2)

        # Build query
        query = f'Find the text "{target_text}" in the HTML and extract {query_suffix}.'

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=ground_truth,
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
                "target_text": target_text,
                "parent_tag": parent_tag,
                "parent_id": parent_id,
                "parent_class": parent_class,
                "extract_type": extract_type,
                "has_comment_with_text": True,
                "html_style": style.value,
                "gotcha": "find(string=...) returns NavigableString, not Tag",
                "common_bug": "soup.find(string='...').name raises AttributeError",
                "correct_solution": "soup.find(string='...').parent.name",
            },
        )
