"""Advanced archetypes for BeautifulSoup RL environment.

This module implements harder tasks that test real-world scraping skills:
- Deep nesting extraction
- Semantic ambiguity (which element?)
- Attribute selectors (beyond ID/class)
- CSS combinators (direct child vs descendant)
- Partial data extraction (handling missing fields)
- List extraction (multiple items)
- Sibling navigation
- Parser differences

These archetypes go beyond API trivia to test genuine parsing ability.
"""

from bs4_env.config import STRING_SCHEMA, LIST_SCHEMA, DICT_SCHEMA, DICT_LIST_SCHEMA
from bs4_env.generators.base import (
    Generator,
    HtmlStyle,
    TaskInstance,
    make_rng,
    generate_variable_content,
    random_id,
    random_class_name,
    random_price,
    random_product_name,
    random_person_name,
    add_noise_comments,
    wrap_with_realistic_chrome,
    introduce_malformation,
    generate_deep_nested_wrapper,
)
from bs4_env.registry import register


# =============================================================================
# Deep Nesting Archetype
# =============================================================================


@register(
    archetype_id="mvp.deep_nesting_extraction",
    category="advanced",
    difficulty="medium",
    solvable=True,
    description="Extract content from deeply nested element (5-8 levels)",
    tags=["extraction", "nesting", "traversal"],
    phase=2,
    answer_schema=STRING_SCHEMA,
)
class DeepNestingExtractionGenerator(Generator):
    """Generate tasks where the target is buried deep in the DOM.

    Real websites often have deeply nested structures (Bootstrap grids,
    React component hierarchies, etc.). This tests whether the model can
    traverse deep DOM structures to find the target.

    Difficulty comes from:
    - Target is 5-8 levels deep
    - Many sibling elements at each level
    - Similar class names at different levels
    - Decoys at various depths
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate target content
        target_text = generate_variable_content(rng, min_sentences=1, max_sentences=3)
        target_id = random_id(rng)

        # Nesting depth: 5-8 levels
        depth = rng.randint(5, 8)

        # Generate wrapper classes that look like real framework patterns
        wrapper_classes = [
            ["container", "mx-auto"],
            ["row", "flex-wrap"],
            ["col", "col-md-8"],
            ["card", "shadow"],
            ["card-body", "p-4"],
            ["content-wrapper", "mt-3"],
            ["text-block", "mb-2"],
            ["inner-content", "leading-relaxed"],
        ]

        # Build nested structure
        html_parts = []
        indent = 0

        for i in range(depth):
            classes = wrapper_classes[i % len(wrapper_classes)]
            class_str = " ".join(classes)
            html_parts.append("  " * indent + f'<div class="{class_str}">')

            # Add decoy content at some levels
            if rng.random() < 0.4:
                decoy_text = generate_variable_content(rng, min_sentences=1, max_sentences=2)
                decoy_class = random_class_name(rng)
                html_parts.append("  " * (indent + 1) + f'<div class="{decoy_class}">{decoy_text}</div>')

            indent += 1

        # Target element at deepest level
        html_parts.append("  " * indent + f'<span id="{target_id}" class="target-content">{target_text}</span>')

        # Close all divs
        for i in range(depth - 1, -1, -1):
            html_parts.append("  " * i + "</div>")

        body_content = "\n".join(html_parts)

        # Wrap with realistic chrome
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Nested Content",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

        # Optional malformation
        if rng.random() < 0.3:
            html = introduce_malformation(html, rng)

        query = f'Extract the text from the element with id="{target_id}". It is nested several levels deep.'

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
                "target_id": target_id,
                "nesting_depth": depth,
                "html_style": style.value,
            },
        )


# =============================================================================
# Semantic Ambiguity Archetype
# =============================================================================


@register(
    archetype_id="mvp.semantic_ambiguity",
    category="advanced",
    difficulty="hard",
    solvable=True,
    description="Extract specific element when multiple similar elements exist",
    tags=["extraction", "ambiguity", "precision"],
    phase=2,
    answer_schema=STRING_SCHEMA,
)
class SemanticAmbiguityGenerator(Generator):
    """Generate tasks where multiple similar elements exist.

    Like Amazon product pages with original price, sale price, shipping cost,
    tax, and total. The model must extract the SPECIFIC price mentioned
    in the query, not just any price.

    Difficulty comes from:
    - Multiple elements with same tag/similar classes
    - All look like valid answers
    - Query specifies which one is needed
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate multiple price-like values
        original_price = random_price(rng, min_val=50, max_val=500)
        sale_price = random_price(rng, min_val=20, max_val=49.99)
        shipping_price = random_price(rng, min_val=5, max_val=15)
        tax_amount = random_price(rng, min_val=2, max_val=30)

        # Randomly select which price to ask for
        price_types = [
            ("original", original_price, "original-price", "Original Price"),
            ("sale", sale_price, "sale-price", "Sale Price"),
            ("shipping", shipping_price, "shipping-cost", "Shipping"),
            ("tax", tax_amount, "tax-amount", "Tax"),
        ]

        target_type, target_value, target_class, target_label = rng.choice(price_types)

        # Build HTML with all prices - order is randomized
        rng.shuffle(price_types)

        price_html_parts = []
        for ptype, value, cls, label in price_types:
            price_html_parts.append(f'''
<div class="price-row">
    <span class="price-label">{label}:</span>
    <span class="price-value {cls}">{value}</span>
</div>''')

        product_name = random_product_name(rng)

        body_content = f'''
<article class="product-card">
    <h1 class="product-name">{product_name}</h1>
    <div class="pricing-section">
        {"".join(price_html_parts)}
    </div>
    <button class="add-to-cart">Add to Cart</button>
</article>
'''

        # Wrap with realistic chrome
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title=f"{product_name} - Shop",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

        if rng.random() < 0.3:
            html = introduce_malformation(html, rng)

        # Query asks for specific price type
        query = f'Extract the {target_label.lower()} for this product. There are multiple prices shown - return only the {target_label.lower()}.'

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=target_value,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema=STRING_SCHEMA,
            normalization={
                "strip_whitespace": True,
            },
            metadata={
                "target_type": target_type,
                "target_class": target_class,
                "all_prices": {pt[0]: pt[1] for pt in price_types},
                "html_style": style.value,
            },
        )


# =============================================================================
# Attribute Selector Archetype
# =============================================================================


@register(
    archetype_id="mvp.attribute_selector",
    category="advanced",
    difficulty="medium",
    solvable=True,
    description="Extract using attribute selectors beyond ID and class",
    tags=["extraction", "attribute", "selector"],
    phase=2,
    answer_schema=STRING_SCHEMA,
)
class AttributeSelectorGenerator(Generator):
    """Generate tasks requiring attribute selectors beyond ID/class.

    Many real-world extractions need to select by data attributes,
    name attributes, href patterns, etc. This tests soup.find() with
    attrs parameter or CSS selector syntax.

    Examples:
    - soup.find('input', {'name': 'email'})
    - soup.select('[data-product-id="123"]')
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Choose attribute type to test
        attr_types = ["name", "data-id", "data-type", "type", "href_pattern"]
        attr_type = rng.choice(attr_types)

        if attr_type == "name":
            # Form field extraction
            target_value = f"user{rng.randint(100, 999)}@example.com"
            target_attr_name = "email"
            decoy_values = [
                f"phone{rng.randint(100, 999)}",
                f"username{rng.randint(100, 999)}",
                f"address{rng.randint(100, 999)}",
            ]
            decoy_attr_names = ["phone", "username", "address"]

            body_content = '<form class="user-form">\n'
            fields = [(target_attr_name, target_value)] + list(zip(decoy_attr_names, decoy_values))
            rng.shuffle(fields)

            for name, value in fields:
                body_content += f'<input type="text" name="{name}" value="{value}" class="form-input">\n'
            body_content += '</form>'

            query = f'Extract the value from the input field with name="{target_attr_name}".'

        elif attr_type in ["data-id", "data-type"]:
            # Data attribute extraction
            target_id = f"prod-{rng.randint(1000, 9999)}"
            target_text = random_product_name(rng)

            decoys = [
                (f"prod-{rng.randint(1000, 9999)}", random_product_name(rng)),
                (f"prod-{rng.randint(1000, 9999)}", random_product_name(rng)),
                (f"prod-{rng.randint(1000, 9999)}", random_product_name(rng)),
            ]

            items = [(target_id, target_text)] + decoys
            rng.shuffle(items)

            body_content = '<div class="product-list">\n'
            for data_id, name in items:
                body_content += f'<div class="product-item" data-product-id="{data_id}">{name}</div>\n'
            body_content += '</div>'

            query = f'Extract the product name from the element with data-product-id="{target_id}".'
            target_value = target_text

        elif attr_type == "type":
            # Input type extraction
            target_type = "submit"
            target_value = f"Submit Form {rng.randint(1, 99)}"

            body_content = f'''
<form>
    <input type="text" value="Enter text here">
    <input type="password" value="secret">
    <input type="hidden" value="hidden-data">
    <input type="{target_type}" value="{target_value}">
    <input type="reset" value="Clear Form">
</form>
'''
            query = f'Extract the value attribute from the input with type="{target_type}".'

        else:  # href_pattern
            # Link pattern extraction
            target_href = f"/products/{rng.randint(100, 999)}"
            target_text = random_product_name(rng)

            links = [
                ("/about", "About Us"),
                ("/contact", "Contact"),
                (target_href, target_text),
                ("/faq", "FAQ"),
            ]
            rng.shuffle(links)

            body_content = '<nav class="site-nav">\n'
            for href, text in links:
                body_content += f'<a href="{href}" class="nav-link">{text}</a>\n'
            body_content += '</nav>'

            query = f'Extract the text from the link with href="{target_href}".'
            target_value = target_text

        # Wrap with realistic chrome
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Form Page",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

        if rng.random() < 0.3:
            html = introduce_malformation(html, rng)

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=target_value,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema=STRING_SCHEMA,
            normalization={
                "strip_whitespace": True,
            },
            metadata={
                "attr_type": attr_type,
                "html_style": style.value,
            },
        )


# =============================================================================
# CSS Combinator Archetype
# =============================================================================


@register(
    archetype_id="mvp.css_combinator",
    category="advanced",
    difficulty="hard",
    solvable=True,
    description="Extract using CSS combinators (direct child vs descendant)",
    tags=["extraction", "css", "combinator"],
    phase=2,
    answer_schema=LIST_SCHEMA,
)
class CSSCombinatorGenerator(Generator):
    """Generate tasks requiring CSS combinators.

    Tests understanding of:
    - Direct child (>): .menu > a
    - Descendant ( ): .menu a
    - Adjacent sibling (+): h1 + p
    - General sibling (~): h1 ~ p

    Example: Extract only top-level menu links, not submenu links.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate menu items
        top_level_items = [
            f"Menu Item {i}" for i in range(1, rng.randint(3, 5) + 1)
        ]
        submenu_items = [
            f"Submenu Item {i}" for i in range(1, rng.randint(4, 7) + 1)
        ]

        # Build menu with nested structure
        body_content = '<nav class="main-menu">\n'
        for i, item in enumerate(top_level_items):
            body_content += f'<a href="/item{i}" class="menu-link">{item}</a>\n'
            # Add submenu to some items
            if i == 1:  # Add submenu to second item
                body_content += '<div class="submenu">\n'
                for j, subitem in enumerate(submenu_items):
                    body_content += f'<a href="/sub{j}" class="menu-link">{subitem}</a>\n'
                body_content += '</div>\n'
        body_content += '</nav>'

        # Wrap with realistic chrome
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Navigation Menu",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

        if rng.random() < 0.3:
            html = introduce_malformation(html, rng)

        query = (
            'Extract only the TOP-LEVEL menu link texts from the .main-menu element. '
            'Do NOT include submenu items. Return as a list of strings.'
        )

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=top_level_items,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema=LIST_SCHEMA,
            normalization={
                "strip_whitespace": True,
                "sort_lists": True,  # Order doesn't matter
            },
            metadata={
                "top_level_count": len(top_level_items),
                "submenu_count": len(submenu_items),
                "html_style": style.value,
                "solution_hint": "Use .main-menu > a or soup.select('.main-menu > a')",
            },
        )


# =============================================================================
# Partial Data Extraction Archetype
# =============================================================================


@register(
    archetype_id="mvp.partial_data_extraction",
    category="advanced",
    difficulty="medium",
    solvable=True,
    description="Extract data when some fields are missing",
    tags=["extraction", "partial", "none-handling"],
    phase=2,
    answer_schema=DICT_LIST_SCHEMA,
)
class PartialDataExtractionGenerator(Generator):
    """Generate tasks where some data is missing.

    Real websites often have inconsistent data - some products have prices,
    some don't. The model must handle missing fields gracefully, returning
    null/None for missing values instead of crashing.

    Tests:
    - Checking for None before accessing
    - Building partial result objects
    - Not failing on missing data
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate products with some missing fields
        num_products = rng.randint(3, 5)
        products = []
        ground_truth = []

        body_content = '<div class="product-list">\n'

        for i in range(num_products):
            name = random_product_name(rng)
            # Randomly omit price or description
            has_price = rng.random() > 0.3  # 70% have price
            has_description = rng.random() > 0.4  # 60% have description

            price = random_price(rng) if has_price else None
            description = generate_variable_content(rng, 1, 2) if has_description else None

            body_content += f'<div class="product-card" data-id="{i}">\n'
            body_content += f'  <h3 class="product-name">{name}</h3>\n'
            if has_price:
                body_content += f'  <span class="product-price">{price}</span>\n'
            if has_description:
                body_content += f'  <p class="product-description">{description}</p>\n'
            body_content += '</div>\n'

            ground_truth.append({
                "name": name,
                "price": price,
                "description": description,
            })

        body_content += '</div>'

        # Wrap with realistic chrome
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Products",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

        if rng.random() < 0.3:
            html = introduce_malformation(html, rng)

        query = (
            'Extract all products as a list of dictionaries. Each dictionary should have: '
            '"name", "price", and "description" keys. If a field is missing, use null.'
        )

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=ground_truth,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema=DICT_LIST_SCHEMA,
            normalization={
                "strip_whitespace": True,
                "collapse_whitespace": True,
                "sort_lists": False,  # Order matters
                "sort_dict_keys": True,
            },
            metadata={
                "num_products": num_products,
                "html_style": style.value,
            },
        )


# =============================================================================
# List Extraction Archetype
# =============================================================================


@register(
    archetype_id="mvp.list_extraction",
    category="advanced",
    difficulty="easy",
    solvable=True,
    description="Extract multiple items from a list structure",
    tags=["extraction", "list", "multiple"],
    phase=2,
    answer_schema=LIST_SCHEMA,
)
class ListExtractionGenerator(Generator):
    """Generate tasks to extract all items from a list.

    Basic but important skill: finding all elements matching a pattern.
    Uses find_all() or select() to get multiple results.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate list items
        num_items = rng.randint(4, 8)
        items = [random_product_name(rng) for _ in range(num_items)]

        # Choose list type
        list_type = rng.choice(["ul", "ol", "div"])

        if list_type in ["ul", "ol"]:
            body_content = f'<{list_type} class="item-list">\n'
            for item in items:
                body_content += f'  <li class="list-item">{item}</li>\n'
            body_content += f'</{list_type}>'
        else:
            body_content = '<div class="item-list">\n'
            for item in items:
                body_content += f'  <div class="list-item">{item}</div>\n'
            body_content += '</div>'

        # Wrap with realistic chrome
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Item List",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

        if rng.random() < 0.3:
            html = introduce_malformation(html, rng)

        query = 'Extract all item texts from the list. Return as a list of strings.'

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=items,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema=LIST_SCHEMA,
            normalization={
                "strip_whitespace": True,
                "sort_lists": True,  # Order doesn't matter for this test
            },
            metadata={
                "num_items": num_items,
                "list_type": list_type,
                "html_style": style.value,
            },
        )


# =============================================================================
# Sibling Navigation Archetype
# =============================================================================


@register(
    archetype_id="mvp.sibling_navigation",
    category="advanced",
    difficulty="medium",
    solvable=True,
    description="Extract data using sibling navigation",
    tags=["extraction", "sibling", "navigation"],
    phase=2,
    answer_schema=STRING_SCHEMA,
)
class SiblingNavigationGenerator(Generator):
    """Generate tasks requiring sibling navigation.

    Common pattern: label followed by value as sibling elements.
    <dt>Name:</dt><dd>John Doe</dd>

    Tests:
    - find_next_sibling()
    - find_previous_sibling()
    - Navigating between adjacent elements
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate label-value pairs
        pairs = [
            ("Name", random_person_name(rng)),
            ("Email", f"user{rng.randint(100, 999)}@example.com"),
            ("Phone", f"+1-{rng.randint(200, 999)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"),
            ("Location", rng.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"])),
        ]

        # Select random target
        target_label, target_value = rng.choice(pairs)

        # Shuffle pairs
        rng.shuffle(pairs)

        # Build definition list
        body_content = '<dl class="info-list">\n'
        for label, value in pairs:
            body_content += f'  <dt class="info-label">{label}:</dt>\n'
            body_content += f'  <dd class="info-value">{value}</dd>\n'
        body_content += '</dl>'

        # Wrap with realistic chrome
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Contact Info",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

        if rng.random() < 0.3:
            html = introduce_malformation(html, rng)

        query = f'Find the value that comes after the "{target_label}:" label. Use sibling navigation.'

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=target_value,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema=STRING_SCHEMA,
            normalization={
                "strip_whitespace": True,
            },
            metadata={
                "target_label": target_label,
                "html_style": style.value,
                "solution_hint": "dt.find_next_sibling('dd') or use CSS: dt:contains('Label') + dd",
            },
        )


# =============================================================================
# Parser Differences Archetype
# =============================================================================


@register(
    archetype_id="mvp.parser_differences",
    category="advanced",
    difficulty="hard",
    solvable=True,
    description="Handle malformed HTML that parses differently across parsers",
    tags=["parser", "malformed", "html.parser", "lxml"],
    phase=2,
    answer_schema=STRING_SCHEMA,
)
class ParserDifferencesGenerator(Generator):
    """Generate tasks where parser choice matters.

    Different BS4 parsers (html.parser, lxml, html5lib) handle malformed
    HTML differently. This tests awareness of parser behavior and the
    importance of specifying the right parser.

    Examples:
    - Unclosed tags: html.parser vs lxml handle differently
    - Attribute quoting: some parsers are more lenient
    - Nesting violations: parsers auto-correct differently
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate target content
        target_text = generate_variable_content(rng, min_sentences=1, max_sentences=2)
        decoy_text = generate_variable_content(rng, min_sentences=1, max_sentences=2)

        # Choose a malformation type that causes parser differences
        malform_types = [
            "unclosed_tag",
            "mismatched_nesting",
            "unquoted_attribute",
        ]
        malform_type = rng.choice(malform_types)

        target_class = random_class_name(rng)

        if malform_type == "unclosed_tag":
            # Unclosed <p> tag - parsers handle the boundary differently
            body_content = f'''<div class="content">
<p class="decoy">{decoy_text}
<p class="{target_class}">{target_text}</p>
<p class="other">Other content here.</p>
</div>'''
            parser_note = "Note: The first <p> is not closed. Use html.parser for consistent results."

        elif malform_type == "mismatched_nesting":
            # Tags closed in wrong order
            body_content = f'''<div class="wrapper">
<b><i class="{target_class}">{target_text}</b></i>
<p>{decoy_text}</p>
</div>'''
            parser_note = "Note: Tags are closed in wrong order (<b><i>...</b></i>). Parsers may restructure this differently."

        else:  # unquoted_attribute
            # Attribute without quotes (space in value causes issues)
            target_class_simple = f"target{rng.randint(100, 999)}"
            body_content = f'''<div class="container">
<span class={target_class_simple}>{target_text}</span>
<span class="decoy">{decoy_text}</span>
</div>'''
            target_class = target_class_simple
            parser_note = "Note: Class attribute is unquoted. Some parsers handle this differently."

        # Wrap with realistic chrome (but simpler to not obscure the malformation)
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Malformed HTML Test",
            complexity="medium",  # Simpler chrome to keep focus on malformation
            include_nav=True,
            include_footer=True,
        )

        query = f'Extract the text from the element with class="{target_class}". {parser_note}'

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
                "malform_type": malform_type,
                "target_class": target_class,
                "html_style": style.value,
                "parser_hint": "Use html.parser for most consistent results on malformed HTML",
            },
        )
