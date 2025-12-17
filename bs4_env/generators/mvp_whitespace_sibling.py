"""Whitespace sibling navigation archetype for BeautifulSoup RL environment.

This module tests the BS4 gotcha where .next_sibling returns whitespace
text nodes between elements:

    # HTML:
    # <a>Link1</a>
    # <a>Link2</a>

    tag.next_sibling       # Returns '\\n' - whitespace text node!
    tag.find_next_sibling('a')  # Correct way to get next <a>

This is one of the most common BS4 mistakes. The DOM includes whitespace
text nodes between elements, so .next_sibling often returns unexpected
NavigableString objects instead of the next element tag.
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
    archetype_id="mvp.whitespace_sibling",
    category="bs4_gotchas",
    difficulty="medium",
    solvable=True,
    description="Navigate to sibling element when whitespace text nodes exist between elements",
    tags=["navigation", "sibling", "gotcha", "whitespace"],
    phase=1,
    answer_schema=STRING_SCHEMA,
)
class WhitespaceSiblingGenerator(Generator):
    """Generate tasks testing whitespace sibling navigation gotcha.

    This tests the common mistake where developers use .next_sibling
    expecting the next element, but get whitespace text instead.

    The task asks to extract the text from an element's next sibling,
    requiring use of find_next_sibling() instead of .next_sibling.
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
            TaskInstance with sibling navigation task.
        """
        rng = make_rng(self.archetype_id, seed)

        # Select style randomly if not specified
        if style is None:
            style = rng.choice(list(HtmlStyle))

        # Generate a list of sibling elements with whitespace between them
        # The key is having FORMATTED HTML with newlines/spaces between tags
        num_items = rng.randint(4, 6)
        items = []
        for i in range(num_items):
            text = generate_variable_content(rng, min_sentences=1, max_sentences=1)
            # Keep text short for list items
            text = text.split('.')[0] + '.'
            items.append(text)

        # Pick the "start" element and "target" (next sibling)
        start_idx = rng.randint(0, num_items - 2)  # Not the last one
        target_idx = start_idx + 1

        start_text = items[start_idx]
        target_text = items[target_idx]

        # Generate IDs/classes for the start element
        start_id = random_id(rng)
        start_class = random_class_name(rng)

        # Choose tag type for the list
        list_tag = rng.choice(["ul", "ol", "div"])
        item_tag = "li" if list_tag in ["ul", "ol"] else "div"

        # Build the list with EXPLICIT WHITESPACE between elements
        # This is the key - pretty-printed HTML has whitespace nodes
        list_items = []
        for i, text in enumerate(items):
            if i == start_idx:
                list_items.append(f'    <{item_tag} id="{start_id}" class="{start_class}">{text}</{item_tag}>')
            else:
                item_class = random_class_name(rng)
                list_items.append(f'    <{item_tag} class="{item_class}">{text}</{item_tag}>')

        # Join with newlines - this creates whitespace text nodes
        list_content = "\n".join(list_items)
        list_wrapper = f'<{list_tag} class="item-list">\n{list_content}\n</{list_tag}>'

        # Add some other content to make it more realistic
        header_text = generate_variable_content(rng, min_sentences=1, max_sentences=2)
        footer_text = generate_variable_content(rng, min_sentences=1, max_sentences=1)

        body_content = f"""
<div class="content-section">
    <h2>{rng.choice(['Items', 'List', 'Options', 'Menu', 'Categories'])}</h2>
    <p>{header_text}</p>
    {list_wrapper}
    <p class="footer-note">{footer_text}</p>
</div>
"""

        # Wrap with realistic chrome
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="List Page",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

        html = add_noise_comments(html, rng, count=2)

        # Build query - ask for the NEXT sibling's text
        query = f'Find the element with id="{start_id}" and extract the text content of its next sibling element (the next {item_tag} tag).'

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
                "start_id": start_id,
                "start_text": start_text,
                "target_text": target_text,
                "start_idx": start_idx,
                "target_idx": target_idx,
                "list_tag": list_tag,
                "item_tag": item_tag,
                "num_items": num_items,
                "html_style": style.value,
                "gotcha": ".next_sibling returns whitespace text node, not next element",
                "common_bug": "elem.next_sibling returns '\\n    ' instead of next element",
                "correct_solution": f"elem.find_next_sibling('{item_tag}')",
            },
        )
