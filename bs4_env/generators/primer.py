"""Primer archetypes for bootstrapping 0% models.

These ultra-simple tasks teach the basic action template:
1. Import BeautifulSoup
2. Parse HTML with BeautifulSoup(HTML, 'html.parser')
3. Select element with .find() or similar
4. Extract and return content

The HTML is intentionally minimal to remove all ambiguity.
"""

from bs4_env.config import STRING_SCHEMA
from bs4_env.generators.base import Generator, TaskInstance, make_rng
from bs4_env.registry import register


@register(
    archetype_id="primer.extract_by_id",
    category="primer",
    difficulty="primer",
    description="Extract text from element with id='target'",
    tags=["bootstrap", "single-element"],
    phase=1,
    answer_schema=STRING_SCHEMA,
)
class PrimerExtractByIdGenerator(Generator):
    """Ultra-simple: extract text from <span id="target">text</span>."""

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        # Simple text content - no special characters, no formatting
        words = ["Hello", "World", "Test", "Example", "Data", "Value", "Item", "Result"]
        text = rng.choice(words)

        html = f'<span id="target">{text}</span>'

        return TaskInstance(
            html=html,
            query="Extract the text from the element with id='target'.",
            ground_truth=text,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            difficulty=self.difficulty,
            answer_schema=STRING_SCHEMA,
            metadata={"primer_type": "id_selector"},
        )


@register(
    archetype_id="primer.extract_by_class",
    category="primer",
    difficulty="primer",
    description="Extract text from element with class='target'",
    tags=["bootstrap", "single-element"],
    phase=1,
    answer_schema=STRING_SCHEMA,
)
class PrimerExtractByClassGenerator(Generator):
    """Ultra-simple: extract text from <div class="target">text</div>."""

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        words = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
        text = rng.choice(words)

        html = f'<div class="target">{text}</div>'

        return TaskInstance(
            html=html,
            query="Extract the text from the element with class='target'.",
            ground_truth=text,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            difficulty=self.difficulty,
            answer_schema=STRING_SCHEMA,
            metadata={"primer_type": "class_selector"},
        )


@register(
    archetype_id="primer.extract_by_tag",
    category="primer",
    difficulty="primer",
    description="Extract text from the only <h1> element",
    tags=["bootstrap", "single-element"],
    phase=1,
    answer_schema=STRING_SCHEMA,
)
class PrimerExtractByTagGenerator(Generator):
    """Ultra-simple: extract text from the only <h1>text</h1>."""

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        titles = ["Welcome", "Introduction", "Overview", "Summary", "Details", "About"]
        text = rng.choice(titles)

        html = f"<h1>{text}</h1>"

        return TaskInstance(
            html=html,
            query="Extract the text from the <h1> heading.",
            ground_truth=text,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            difficulty=self.difficulty,
            answer_schema=STRING_SCHEMA,
            metadata={"primer_type": "tag_selector"},
        )


@register(
    archetype_id="primer.extract_attribute",
    category="primer",
    difficulty="primer",
    description="Extract href attribute from a link",
    tags=["bootstrap", "attribute"],
    phase=1,
    answer_schema=STRING_SCHEMA,
)
class PrimerExtractAttributeGenerator(Generator):
    """Ultra-simple: extract href from <a href="url" id="link">text</a>."""

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        urls = ["/home", "/about", "/contact", "/products", "/services", "/help"]
        url = rng.choice(urls)
        text = rng.choice(["Click here", "Link", "Go", "Visit", "See more"])

        html = f'<a href="{url}" id="link">{text}</a>'

        return TaskInstance(
            html=html,
            query="Extract the href attribute from the link with id='link'.",
            ground_truth=url,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            difficulty=self.difficulty,
            answer_schema=STRING_SCHEMA,
            metadata={"primer_type": "attribute_extraction"},
        )


@register(
    archetype_id="primer.count_elements",
    category="primer",
    difficulty="primer",
    description="Count the number of <li> elements",
    tags=["bootstrap", "counting"],
    phase=1,
    answer_schema={"type": "integer"},
)
class PrimerCountElementsGenerator(Generator):
    """Ultra-simple: count <li> elements in a short list."""

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        # Generate 2-5 list items
        count = rng.randint(2, 5)
        items = ["Apple", "Banana", "Cherry", "Date", "Elderberry"][:count]

        li_elements = "\n".join(f"  <li>{item}</li>" for item in items)
        html = f"<ul>\n{li_elements}\n</ul>"

        return TaskInstance(
            html=html,
            query="Count the number of <li> elements in the list.",
            ground_truth=count,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            difficulty=self.difficulty,
            answer_schema={"type": "integer"},
            metadata={"primer_type": "element_counting", "expected_count": count},
        )
