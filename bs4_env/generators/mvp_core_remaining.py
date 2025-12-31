"""Remaining Core Extraction archetypes for BeautifulSoup RL environment.

This module implements additional core extraction tasks from the PRD:
- Extract single attribute value
- Extract all links as {text, href}
- Extract all images as {src, alt}
- Direct children enumeration
- Descendants with filter
- Table column by header
- Custom predicate matching

These complete the PRD's 50 archetype target.
"""

from bs4_env.config import STRING_SCHEMA
from bs4_env.generators.base import (
    Generator,
    HtmlStyle,
    TaskInstance,
    add_noise_comments,
    generate_variable_content,
    make_rng,
    random_class_name,
    random_id,
    wrap_with_realistic_chrome,
)
from bs4_env.registry import register


@register(
    archetype_id="mvp.extract_attribute",
    category="core_extraction",
    difficulty="easy",
    solvable=True,
    description="Extract a single attribute value (href, src, data-*) from an element",
    tags=["extraction", "attributes", "data"],
    phase=1,
    answer_schema=STRING_SCHEMA,
)
class ExtractAttributeGenerator(Generator):
    """Generate tasks to extract a single attribute value.

    Tests the ability to:
    - Find an element by selector
    - Extract a specific attribute (not text)
    - Handle data-* attributes correctly
    """

    ATTRIBUTE_TYPES = [
        ("href", "a", "/page/{}"),
        ("src", "img", "/images/{}.jpg"),
        ("data-id", "div", "item-{}"),
        ("data-value", "span", "val-{}"),
        ("title", "a", "Link to {}"),
        ("alt", "img", "Image of {}"),
    ]

    def generate(
        self,
        seed: int,
        style: HtmlStyle | None = None,
    ) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        if style is None:
            style = rng.choice(list(HtmlStyle))

        # Pick attribute type
        attr_name, tag, value_template = rng.choice(self.ATTRIBUTE_TYPES)

        # Generate target value
        target_value = value_template.format(rng.randint(1000, 9999))
        target_id = random_id(rng)

        # Build target element
        if tag == "a":
            target_html = f'<a id="{target_id}" {attr_name}="{target_value}">Click here</a>'
        elif tag == "img":
            if attr_name == "src":
                target_html = f'<img id="{target_id}" {attr_name}="{target_value}" alt="placeholder">'
            else:
                target_html = f'<img id="{target_id}" src="/img/x.jpg" {attr_name}="{target_value}">'
        else:
            target_html = f'<{tag} id="{target_id}" {attr_name}="{target_value}">Content</{tag}>'

        # Add distractor elements with different attribute values
        distractors = []
        for _ in range(3):
            dist_id = random_id(rng)
            dist_value = value_template.format(rng.randint(1000, 9999))
            if tag == "a":
                distractors.append(f'<a id="{dist_id}" {attr_name}="{dist_value}">Other link</a>')
            elif tag == "img":
                distractors.append(f'<img id="{dist_id}" src="{dist_value}" alt="other">')
            else:
                distractors.append(f'<{tag} id="{dist_id}" {attr_name}="{dist_value}">Other</{tag}>')

        # Shuffle elements
        elements = [target_html] + distractors
        rng.shuffle(elements)

        body_content = f"""
<div class="content">
  {"".join(elements)}
</div>
"""

        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Attribute Test",
            complexity="low",
        )

        query = f'Extract the value of the "{attr_name}" attribute from the element with id="{target_id}".'

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=target_value,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema=STRING_SCHEMA,
            normalization={"strip_whitespace": True},
            metadata={
                "target_id": target_id,
                "attr_name": attr_name,
                "tag": tag,
                "html_style": style.value,
            },
        )


@register(
    archetype_id="mvp.extract_links",
    category="core_extraction",
    difficulty="medium",
    solvable=True,
    description="Extract all links as a list of {text, href} objects",
    tags=["extraction", "links", "structured"],
    phase=1,
    answer_schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "href": {"type": "string"},
            },
            "required": ["text", "href"],
        },
    },
)
class ExtractLinksGenerator(Generator):
    """Generate tasks to extract all links from a container.

    Tests the ability to:
    - Find all <a> elements in a container
    - Extract both text and href
    - Handle whitespace in link text
    - Maintain document order
    """

    LINK_THEMES = {
        "navigation": [
            ("Home", "/"),
            ("About Us", "/about"),
            ("Products", "/products"),
            ("Contact", "/contact"),
            ("Blog", "/blog"),
        ],
        "articles": [
            ("Breaking News", "/news/breaking"),
            ("Sports Update", "/sports/latest"),
            ("Tech Review", "/tech/review"),
            ("Opinion Piece", "/opinion/today"),
        ],
        "products": [
            ("Laptop Pro", "/products/laptop-pro"),
            ("Wireless Mouse", "/products/mouse"),
            ("USB Hub", "/products/usb-hub"),
            ("Monitor Stand", "/products/stand"),
        ],
    }

    def generate(
        self,
        seed: int,
        style: HtmlStyle | None = None,
    ) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        if style is None:
            style = rng.choice(list(HtmlStyle))

        # Pick theme and links
        theme = rng.choice(list(self.LINK_THEMES.keys()))
        all_links = self.LINK_THEMES[theme].copy()
        rng.shuffle(all_links)
        num_links = rng.randint(3, min(5, len(all_links)))
        selected_links = all_links[:num_links]

        # Build ground truth
        ground_truth = [
            {"text": text, "href": href}
            for text, href in selected_links
        ]

        container_id = random_id(rng)
        container_class = random_class_name(rng)

        # Build link HTML
        link_parts = []
        for text, href in selected_links:
            link_parts.append(f'<a href="{href}">{text}</a>')

        links_html = f"""
<nav id="{container_id}" class="{container_class}">
  <ul>
    {"".join(f'<li>{link}</li>' for link in link_parts)}
  </ul>
</nav>
"""

        # Add distractor links outside the container
        distractor_links = """
<div class="sidebar">
  <a href="/login">Login</a>
  <a href="/register">Register</a>
</div>
"""

        body_content = f"""
{links_html}
<main>
  <p>Main content here.</p>
</main>
{distractor_links}
"""

        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Links Page",
            complexity="medium",
        )

        query = f'Extract all links from the nav element with id="{container_id}". Return a list of objects with text and href fields.'

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=ground_truth,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "href": {"type": "string"},
                    },
                },
            },
            normalization={"strip_whitespace": True},
            metadata={
                "container_id": container_id,
                "theme": theme,
                "num_links": num_links,
                "html_style": style.value,
            },
        )


@register(
    archetype_id="mvp.extract_images",
    category="core_extraction",
    difficulty="medium",
    solvable=True,
    description="Extract all images as a list of {src, alt} objects, handling missing alt",
    tags=["extraction", "images", "structured"],
    phase=1,
    answer_schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "src": {"type": "string"},
                "alt": {"type": "string"},
            },
            "required": ["src"],
        },
    },
)
class ExtractImagesGenerator(Generator):
    """Generate tasks to extract all images from a container.

    Tests the ability to:
    - Find all <img> elements
    - Extract src and alt attributes
    - Handle missing alt attribute (return empty string or null)
    - Maintain document order
    """

    IMAGE_THEMES = {
        "gallery": [
            ("/images/photo1.jpg", "Mountain landscape"),
            ("/images/photo2.jpg", "Ocean sunset"),
            ("/images/photo3.jpg", ""),  # Missing alt
            ("/images/photo4.jpg", "Forest path"),
        ],
        "products": [
            ("/products/item1.png", "Product A"),
            ("/products/item2.png", "Product B"),
            ("/products/item3.png", "Product C"),
        ],
        "avatars": [
            ("/avatars/user1.jpg", "John's profile"),
            ("/avatars/user2.jpg", ""),  # Missing alt
            ("/avatars/user3.jpg", "Sarah's profile"),
        ],
    }

    def generate(
        self,
        seed: int,
        style: HtmlStyle | None = None,
    ) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        if style is None:
            style = rng.choice(list(HtmlStyle))

        # Pick theme and images
        theme = rng.choice(list(self.IMAGE_THEMES.keys()))
        all_images = self.IMAGE_THEMES[theme].copy()
        rng.shuffle(all_images)
        num_images = rng.randint(2, min(4, len(all_images)))
        selected_images = all_images[:num_images]

        # Build ground truth
        ground_truth = [
            {"src": src, "alt": alt}
            for src, alt in selected_images
        ]

        container_id = random_id(rng)
        container_class = random_class_name(rng)

        # Build image HTML
        image_parts = []
        for src, alt in selected_images:
            if alt:
                image_parts.append(f'<img src="{src}" alt="{alt}">')
            else:
                image_parts.append(f'<img src="{src}">')  # No alt attribute

        images_html = f"""
<div id="{container_id}" class="{container_class} gallery">
  {"".join(image_parts)}
</div>
"""

        # Add distractor images outside container
        distractor_images = """
<aside class="ads">
  <img src="/ads/banner.jpg" alt="Advertisement">
</aside>
"""

        body_content = f"""
<h2>Image Gallery</h2>
{images_html}
{distractor_images}
"""

        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Image Gallery",
            complexity="low",
        )

        query = f'Extract all images from the container with id="{container_id}". Return a list of objects with src and alt fields. If alt is missing, use an empty string.'

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=ground_truth,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "src": {"type": "string"},
                        "alt": {"type": "string"},
                    },
                },
            },
            normalization={"strip_whitespace": True},
            metadata={
                "container_id": container_id,
                "theme": theme,
                "num_images": num_images,
                "html_style": style.value,
            },
        )


@register(
    archetype_id="mvp.direct_children",
    category="traversal",
    difficulty="medium",
    solvable=True,
    description="Enumerate only direct children of an element, not descendants",
    tags=["traversal", "children", "bs4_gotcha"],
    phase=1,
    answer_schema={
        "type": "array",
        "items": {"type": "string"},
    },
)
class DirectChildrenGenerator(Generator):
    """Generate tasks to enumerate direct children only.

    Tests the BS4 distinction between:
    - .children (direct children only, including NavigableString)
    - .descendants (all descendants recursively)
    - .find_all(recursive=False) (direct child tags only)

    This is a common gotcha where models use find_all() which
    returns all descendants, not just direct children.
    """

    def generate(
        self,
        seed: int,
        style: HtmlStyle | None = None,
    ) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        if style is None:
            style = rng.choice(list(HtmlStyle))

        container_id = random_id(rng)

        # Generate direct children (what we want)
        num_children = rng.randint(3, 5)
        child_texts = [f"Child {i+1}" for i in range(num_children)]

        # Generate nested content that looks like children but isn't direct
        nested_item_texts = ["Nested A", "Nested B", "Deep Item"]

        # Build HTML with nested structure
        # Direct children have simple text, but there's a nested wrapper
        # with its own items that should NOT be included
        children_html = []
        for text in child_texts:
            children_html.append(f'<div class="item">{text}</div>')

        # Add a nested container with items that look similar but are NOT direct
        nested_wrapper = f"""
<div class="nested-wrapper">
  <div class="item">{nested_item_texts[0]}</div>
  <div class="item">{nested_item_texts[1]}</div>
  <div class="deep">
    <div class="item">{nested_item_texts[2]}</div>
  </div>
</div>
"""

        container_html = f"""
<div id="{container_id}" class="parent">
  {"".join(children_html)}
  {nested_wrapper}
</div>
"""

        body_content = f"""
<section>
  <h2>Container</h2>
  {container_html}
</section>
"""

        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Children Test",
            complexity="low",
        )

        # Ground truth is ONLY the direct child texts
        ground_truth = child_texts

        query = f'Extract the text content of only the DIRECT children (div.item elements) of the container with id="{container_id}". Do not include items from nested wrappers. Return as a list of strings.'

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=ground_truth,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema={
                "type": "array",
                "items": {"type": "string"},
            },
            normalization={
                "strip_whitespace": True,
                "collapse_whitespace": True,
            },
            metadata={
                "container_id": container_id,
                "num_children": num_children,
                "has_nested": True,
                "html_style": style.value,
                "gotcha": "Must use recursive=False or .children, not find_all()",
            },
        )


@register(
    archetype_id="mvp.descendants_filter",
    category="traversal",
    difficulty="medium",
    solvable=True,
    description="Search descendants with a conditional filter (tags containing substring)",
    tags=["traversal", "filter", "predicate"],
    phase=1,
    answer_schema={
        "type": "array",
        "items": {"type": "string"},
    },
)
class DescendantsFilterGenerator(Generator):
    """Generate tasks to find descendants matching a filter.

    Tests the ability to:
    - Use find_all with a custom filter function
    - Filter by text content containing a substring
    - Search through nested structures
    """

    def generate(
        self,
        seed: int,
        style: HtmlStyle | None = None,
    ) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        if style is None:
            style = rng.choice(list(HtmlStyle))

        container_id = random_id(rng)

        # Pick a filter keyword
        keywords = ["important", "featured", "special", "highlight"]
        keyword = rng.choice(keywords)

        # Generate items - some match, some don't
        items = [
            (f"Regular item {rng.randint(1, 99)}", False),
            (f"This is {keyword} content", True),
            (f"Another regular item", False),
            (f"A {keyword} announcement", True),
            (f"Normal text here", False),
            (f"Something {keyword} to note", True),
        ]
        rng.shuffle(items)

        # Build HTML
        item_parts = []
        for text, _ in items:
            item_parts.append(f'<div class="item"><p>{text}</p></div>')

        container_html = f"""
<div id="{container_id}" class="container">
  {"".join(item_parts)}
</div>
"""

        body_content = f"""
<section>
  {container_html}
</section>
"""

        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Filter Test",
            complexity="low",
        )

        # Ground truth is only items containing the keyword
        ground_truth = [text for text, matches in items if matches]

        query = f'Find all text content in the container with id="{container_id}" that contains the word "{keyword}". Return as a list of strings.'

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=ground_truth,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema={
                "type": "array",
                "items": {"type": "string"},
            },
            normalization={
                "strip_whitespace": True,
                "collapse_whitespace": True,
            },
            metadata={
                "container_id": container_id,
                "keyword": keyword,
                "num_matches": len(ground_truth),
                "html_style": style.value,
            },
        )


@register(
    archetype_id="mvp.table_column_by_header",
    category="table_parsing",
    difficulty="medium",
    solvable=True,
    description="Extract a specific column from a table by header name",
    tags=["tables", "extraction", "columns"],
    phase=1,
    answer_schema={
        "type": "array",
        "items": {"type": "string"},
    },
)
class TableColumnByHeaderGenerator(Generator):
    """Generate tasks to extract a table column by header name.

    Tests the ability to:
    - Parse table headers to find column index
    - Extract all cells in that column
    - Handle dynamic/randomized data (prevents reward hacking)
    """

    # Templates for dynamic data generation
    TABLE_TEMPLATES = {
        "employees": {
            "headers": ["Name", "Department", "Salary", "Start Date"],
            "name_pool": ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry"],
            "dept_pool": ["Engineering", "Marketing", "Sales", "HR", "Finance", "Legal"],
        },
        "products": {
            "headers": ["Product", "Category", "Price", "Stock"],
            "product_pool": ["Widget", "Gadget", "Tool", "Device", "Module", "Unit"],
            "category_pool": ["Electronics", "Hardware", "Software", "Accessories"],
        },
        "inventory": {
            "headers": ["Item", "Location", "Quantity", "Last Updated"],
            "item_pool": ["Part A", "Part B", "Component X", "Supply Y", "Material Z"],
            "location_pool": ["Warehouse A", "Warehouse B", "Store 1", "Store 2", "Factory"],
        },
    }

    def _generate_dynamic_rows(self, rng, table_type: str, num_rows: int) -> list[list[str]]:
        """Generate dynamic table rows using rng to prevent memorization."""
        template = self.TABLE_TEMPLATES[table_type]
        rows = []

        if table_type == "employees":
            names = rng.sample(template["name_pool"], min(num_rows, len(template["name_pool"])))
            for i in range(num_rows):
                name = names[i % len(names)] if i < len(names) else f"Person{i}"
                dept = rng.choice(template["dept_pool"])
                salary = f"${rng.randint(45, 120) * 1000:,}"
                year = rng.randint(2015, 2023)
                month = rng.randint(1, 12)
                day = rng.randint(1, 28)
                date = f"{year}-{month:02d}-{day:02d}"
                rows.append([name, dept, salary, date])

        elif table_type == "products":
            products = rng.sample(template["product_pool"], min(num_rows, len(template["product_pool"])))
            for i in range(num_rows):
                prod = products[i % len(products)] if i < len(products) else f"Item{i}"
                prod_name = f"{prod} {rng.choice(['Pro', 'Plus', 'Lite', 'Max'])}"
                cat = rng.choice(template["category_pool"])
                price = f"${rng.randint(10, 200)}.{rng.randint(0, 99):02d}"
                stock = str(rng.randint(10, 500))
                rows.append([prod_name, cat, price, stock])

        elif table_type == "inventory":
            items = rng.sample(template["item_pool"], min(num_rows, len(template["item_pool"])))
            for i in range(num_rows):
                item = items[i % len(items)] if i < len(items) else f"Item{i}"
                loc = rng.choice(template["location_pool"])
                qty = str(rng.randint(1, 1000))
                year = rng.randint(2023, 2024)
                month = rng.randint(1, 12)
                day = rng.randint(1, 28)
                date = f"{year}-{month:02d}-{day:02d}"
                rows.append([item, loc, qty, date])

        return rows

    def generate(
        self,
        seed: int,
        style: HtmlStyle | None = None,
    ) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        if style is None:
            style = rng.choice(list(HtmlStyle))

        # Pick table type and generate dynamic data
        table_type = rng.choice(list(self.TABLE_TEMPLATES.keys()))
        template = self.TABLE_TEMPLATES[table_type]
        headers = template["headers"]
        num_rows = rng.randint(3, 5)
        rows = self._generate_dynamic_rows(rng, table_type, num_rows)

        # Pick target column (not the first one, as that's too easy)
        target_col_idx = rng.randint(1, len(headers) - 1)
        target_header = headers[target_col_idx]

        table_id = random_id(rng)

        # Build table HTML
        header_html = "".join(f"<th>{h}</th>" for h in headers)
        rows_html = ""
        for row in rows:
            cells = "".join(f"<td>{cell}</td>" for cell in row)
            rows_html += f"<tr>{cells}</tr>"

        table_html = f"""
<table id="{table_id}">
  <thead>
    <tr>{header_html}</tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
"""

        body_content = f"""
<div class="data-section">
  <h2>{table_type.title()} Data</h2>
  {table_html}
</div>
"""

        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title=f"{table_type.title()} Table",
            complexity="low",
        )

        # Ground truth is all values in the target column
        ground_truth = [row[target_col_idx] for row in rows]

        query = f'Extract all values from the "{target_header}" column of the table with id="{table_id}". Return as a list of strings.'

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=ground_truth,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema={
                "type": "array",
                "items": {"type": "string"},
            },
            normalization={"strip_whitespace": True},
            metadata={
                "table_id": table_id,
                "target_header": target_header,
                "target_col_idx": target_col_idx,
                "num_rows": len(rows),
                "html_style": style.value,
            },
        )


@register(
    archetype_id="mvp.remove_scripts_styles",
    category="output_normalization",
    difficulty="medium",
    solvable=True,
    description="Remove script and style nodes before extracting text content",
    tags=["extraction", "cleanup", "scripts"],
    phase=1,
    answer_schema=STRING_SCHEMA,
)
class RemoveScriptsStylesGenerator(Generator):
    """Generate tasks to extract text after removing scripts/styles.

    Tests the ability to:
    - Use decompose() or extract() to remove unwanted nodes
    - Extract clean text without script/style content
    - Handle inline scripts and styles

    This is critical for real-world scraping where pages have
    embedded JavaScript and CSS that contaminates get_text().
    """

    def generate(
        self,
        seed: int,
        style: HtmlStyle | None = None,
    ) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        if style is None:
            style = rng.choice(list(HtmlStyle))

        container_id = random_id(rng)

        # Generate the actual content (ground truth)
        target_text = generate_variable_content(rng, min_sentences=2, max_sentences=3)

        # Generate script content (should NOT appear in result)
        script_content = f"""
function init() {{
  console.log('Initialized');
  var data = {{'key': 'value_{rng.randint(1000, 9999)}'}};
}}
"""

        # Generate style content (should NOT appear in result)
        style_content = f"""
.container {{ padding: 20px; margin: 10px; }}
.content-{rng.randint(100, 999)} {{ color: blue; }}
"""

        # Build HTML with inline scripts and styles mixed with content
        container_html = f"""
<div id="{container_id}" class="article">
  <style>{style_content}</style>
  <p>{target_text}</p>
  <script>{script_content}</script>
</div>
"""

        body_content = f"""
<article>
  {container_html}
</article>
"""

        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Article Page",
            complexity="low",
        )

        query = f'Extract the text content from the container with id="{container_id}". Remove all <script> and <style> elements before extraction to get clean text only.'

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
                "container_id": container_id,
                "has_script": True,
                "has_style": True,
                "html_style": style.value,
                "gotcha": "Must remove script/style before get_text() or use decompose()",
            },
        )
