"""Multi-valued class attribute archetype for BeautifulSoup RL environment.

This module tests the BS4 gotcha where class attributes return a LIST,
not a string:

    # HTML: <div class="nav primary active"></div>
    soup.find('div')['class']  # Returns ['nav', 'primary', 'active'] - a LIST!

Common bug:
    soup.find('div')['class'] == 'primary'  # False! It's a list

Correct solutions:
    soup.find('div', class_='primary')  # BS4 matches any class in the list
    'primary' in soup.find('div')['class']  # Explicit list check
"""

from bs4_env.config import STRING_SCHEMA
from bs4_env.generators.base import (
    Generator,
    HtmlStyle,
    TaskInstance,
    make_rng,
    generate_variable_content,
    random_id,
    wrap_with_realistic_chrome,
    add_noise_comments,
)
from bs4_env.registry import register


# Class name parts for realistic multi-class elements
CLASS_PREFIXES = ["btn", "nav", "card", "list", "form", "input", "text", "grid", "flex", "col"]
CLASS_MODIFIERS = ["primary", "secondary", "active", "disabled", "hidden", "large", "small", "dark", "light"]
CLASS_STATES = ["hover", "focus", "selected", "expanded", "collapsed", "open", "closed"]
CLASS_SIZES = ["sm", "md", "lg", "xl", "xs", "xxl"]

# Safe modifiers for query classes - these won't collide with chrome templates
# Avoids: "hidden" (used in Tailwind nav), "active" (common in Bootstrap),
# "disabled" (common in forms), "primary"/"secondary" (Bootstrap buttons)
SAFE_QUERY_MODIFIERS = ["featured", "highlighted", "promoted", "pinned", "starred", "verified", "premium", "urgent"]


def generate_multiclass_name(rng, num_classes: int = 3) -> list[str]:
    """Generate a realistic multi-class attribute.

    Args:
        rng: Random instance.
        num_classes: Number of classes to include.

    Returns:
        List of class names.
    """
    classes = []

    # First class is usually a component type
    classes.append(rng.choice(CLASS_PREFIXES) + "-" + rng.choice(["item", "container", "wrapper", "box"]))

    # Add modifiers/states
    if num_classes >= 2:
        classes.append(rng.choice(CLASS_MODIFIERS))
    if num_classes >= 3:
        # Could be size, state, or another modifier
        choice = rng.choice(["size", "state", "modifier"])
        if choice == "size":
            classes.append(rng.choice(CLASS_SIZES))
        elif choice == "state":
            classes.append(rng.choice(CLASS_STATES))
        else:
            classes.append(rng.choice(CLASS_MODIFIERS))
    if num_classes >= 4:
        classes.append(rng.choice(CLASS_STATES))

    return classes[:num_classes]


@register(
    archetype_id="mvp.multivalue_class",
    category="bs4_gotchas",
    difficulty="medium",
    solvable=True,
    description="Extract from element with multi-valued class attribute (class returns list)",
    tags=["extraction", "class", "gotcha", "list"],
    phase=1,
    answer_schema=STRING_SCHEMA,
)
class MultivalueClassGenerator(Generator):
    """Generate tasks testing multi-valued class attribute gotcha.

    This tests the common mistake where developers assume class attributes
    are strings, when BS4 actually returns them as lists.

    The task asks to extract content from an element that has a specific
    class among multiple classes. The model must understand that:
    1. soup.find('div')['class'] returns a LIST
    2. Use class_= parameter or 'in' check for matching
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
            TaskInstance with multi-class extraction task.
        """
        rng = make_rng(self.archetype_id, seed)

        # Select style randomly if not specified
        if style is None:
            style = rng.choice(list(HtmlStyle))

        # Generate target content
        target_text = generate_variable_content(rng, min_sentences=1, max_sentences=3)

        # Generate multi-class names for target
        # Target has 3-4 classes
        target_classes = generate_multiclass_name(rng, num_classes=rng.randint(3, 4))

        # Pick query_class from SAFE_QUERY_MODIFIERS to avoid chrome collisions
        # (e.g., "hidden" appears in Tailwind nav templates)
        query_class = rng.choice(SAFE_QUERY_MODIFIERS)

        # Replace a non-first class with the safe query_class
        replace_idx = rng.randint(1, len(target_classes) - 1)
        target_classes[replace_idx] = query_class

        target_class_str = " ".join(target_classes)

        # Generate distractors - some share SOME classes but not the query class
        distractors = []
        for i in range(3):
            distractor_text = generate_variable_content(rng, min_sentences=1, max_sentences=2)
            distractor_classes = generate_multiclass_name(rng, num_classes=rng.randint(2, 4))

            # Make sure query_class is NOT in distractor (otherwise ground truth would be wrong)
            distractor_classes = [c for c in distractor_classes if c != query_class]
            if not distractor_classes:
                distractor_classes = [rng.choice(CLASS_PREFIXES) + "-default"]

            # Occasionally add a class that's similar to query_class (harder decoy)
            if rng.random() < 0.3 and i == 0:
                similar = query_class + "-alt"
                distractor_classes.append(similar)

            distractor_class_str = " ".join(distractor_classes)
            distractors.append((distractor_text, distractor_class_str))

        # Build elements - randomize order
        target_id = random_id(rng)
        elements = [
            ("div", target_class_str, target_text, target_id, True),
        ]
        for i, (text, cls) in enumerate(distractors):
            elements.append(("div", cls, text, random_id(rng), False))

        rng.shuffle(elements)

        # Build body content
        body_parts = []
        for tag, cls, text, elem_id, is_target in elements:
            body_parts.append(f'<{tag} id="{elem_id}" class="{cls}">{text}</{tag}>')

        body_content = "\n".join(body_parts)

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

        # Build query - emphasize the class
        query = f'Extract the text content from the element that has class="{query_class}".'

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
                "target_classes": target_classes,
                "query_class": query_class,
                "target_class_str": target_class_str,
                "target_id": target_id,
                "html_style": style.value,
                "gotcha": "class attribute returns LIST, not string",
                "common_bug": f"soup.find('div')['class'] == '{query_class}' returns False",
                "correct_solution": f"soup.find('div', class_='{query_class}')",
            },
        )
