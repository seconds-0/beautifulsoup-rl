"""Hard archetypes for BeautifulSoup RL environment.

This module implements challenging tasks that require semantic reasoning:
- Relational queries (find value by label relationship)
- Multi-hop extraction (filter → navigate → extract)
- Aggregation tasks (collect → compute)
- Structured multi-field output

These archetypes are designed to:
1. Not be solvable with simple regex/string matching
2. Require understanding of DOM structure and relationships
3. Provide good RL training signal for small models
4. Challenge but not saturate frontier models
"""

from bs4_env.config import (
    INT_SCHEMA,
    STRING_SCHEMA,
)
from bs4_env.generators.base import (
    Generator,
    HtmlStyle,
    TaskInstance,
    introduce_malformation,
    make_rng,
    random_class_name,
    random_price,
    random_product_name,
    wrap_with_realistic_chrome,
)
from bs4_env.registry import register

# =============================================================================
# Relational Query Archetype
# =============================================================================


@register(
    archetype_id="mvp.relational_query",
    category="hard",
    difficulty="hard",
    solvable=True,
    description="Extract value by navigating from a label reference",
    tags=["extraction", "relational", "label", "semantic"],
    phase=2,
    answer_schema=STRING_SCHEMA,
)
class RelationalQueryGenerator(Generator):
    """Generate tasks requiring relational reasoning.

    Instead of "find element with id='x'", asks "find the value after the
    label 'Shipping'". This requires:
    1. Finding the label element
    2. Navigating to the related value element
    3. Extracting the correct value

    This prevents simple ID/class lookup and requires understanding
    HTML structure and relationships.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate label-value pairs for a product/order summary
        pairs = [
            ("Subtotal", random_price(rng, min_val=50, max_val=200)),
            ("Shipping", random_price(rng, min_val=5, max_val=25)),
            ("Tax", random_price(rng, min_val=3, max_val=20)),
            ("Discount", f"-{random_price(rng, min_val=5, max_val=30)}"),
            ("Total", random_price(rng, min_val=60, max_val=250)),
        ]

        # Select random target
        target_label, target_value = rng.choice(pairs)

        # Shuffle to prevent position-based shortcuts
        rng.shuffle(pairs)

        # Choose layout pattern
        layout = rng.choice(["table", "dl", "div_pairs"])

        if layout == "table":
            body_content = """<table class="order-summary">
  <thead><tr><th>Item</th><th>Amount</th></tr></thead>
  <tbody>
"""
            for label, value in pairs:
                row_class = random_class_name(rng)
                body_content += f'    <tr class="{row_class}"><td class="label">{label}</td><td class="value">{value}</td></tr>\n'
            body_content += "  </tbody>\n</table>"

        elif layout == "dl":
            body_content = '<dl class="order-summary">\n'
            for label, value in pairs:
                body_content += f"  <dt>{label}</dt>\n  <dd>{value}</dd>\n"
            body_content += "</dl>"

        else:  # div_pairs
            body_content = '<div class="order-summary">\n'
            for label, value in pairs:
                row_class = random_class_name(rng)
                body_content += f"""  <div class="row {row_class}">
    <span class="label">{label}</span>
    <span class="amount">{value}</span>
  </div>\n"""
            body_content += "</div>"

        # Wrap with realistic chrome
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Order Summary",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

        if rng.random() < 0.3:
            html = introduce_malformation(html, rng)

        query = f'Extract the value from the row labeled "{target_label}". Navigate from the label to find its associated value.'

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
                "layout": layout,
                "html_style": style.value,
            },
        )


# =============================================================================
# Multi-Hop Filter Archetype
# =============================================================================


@register(
    archetype_id="mvp.multi_hop_filter",
    category="hard",
    difficulty="hard",
    solvable=True,
    description="Filter by condition, then extract related data",
    tags=["extraction", "filter", "multi-hop", "semantic"],
    phase=2,
    answer_schema=STRING_SCHEMA,
)
class MultiHopFilterGenerator(Generator):
    """Generate tasks requiring multi-step reasoning.

    Example: "Find the product with rating > 4.5, extract its SKU"

    This requires:
    1. Finding all product elements
    2. Filtering by the condition (rating > 4.5)
    3. Extracting the requested field from matching element

    Can't be solved by simple pattern matching.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate products
        num_products = rng.randint(4, 7)
        products = []

        for _i in range(num_products):
            products.append(
                {
                    "name": random_product_name(rng),
                    "price": random_price(rng, min_val=20, max_val=200),
                    "rating": round(rng.uniform(2.5, 5.0), 1),
                    "sku": f"SKU-{rng.randint(10000, 99999)}",
                    "in_stock": rng.random() > 0.3,
                }
            )

        # Choose filter condition and target field
        filter_types = [
            ("rating", "> 4.0", lambda p: p["rating"] > 4.0),
            ("rating", ">= 4.5", lambda p: p["rating"] >= 4.5),
            ("in_stock", "is true", lambda p: p["in_stock"]),
        ]

        filter_field, filter_desc, filter_fn = rng.choice(filter_types)

        # Find products matching filter
        matching = [p for p in products if filter_fn(p)]

        # Ensure at least one match (adjust if needed)
        if not matching:
            # Force one product to match
            products[0]["rating"] = 4.8
            products[0]["in_stock"] = True
            matching = [p for p in products if filter_fn(p)]

        # Choose extraction target (different from filter field)
        if filter_field == "rating":
            extract_fields = [("sku", "SKU"), ("name", "product name")]
        else:
            extract_fields = [("sku", "SKU"), ("price", "price")]

        extract_field, extract_desc = rng.choice(extract_fields)

        # Shuffle products BEFORE selecting target
        # This ensures "first listed" in HTML matches ground truth
        rng.shuffle(products)

        # Find first matching product in shuffled (display) order
        target_product = None
        for p in products:
            if filter_fn(p):
                target_product = p
                break

        ground_truth = str(target_product[extract_field])

        # Build HTML
        body_content = '<div class="product-grid">\n'
        for p in products:
            stock_class = "in-stock" if p["in_stock"] else "out-of-stock"
            body_content += f"""  <div class="product-card {stock_class}">
    <h3 class="product-name">{p["name"]}</h3>
    <span class="price">{p["price"]}</span>
    <span class="rating" data-rating="{p["rating"]}">{p["rating"]} stars</span>
    <span class="sku">{p["sku"]}</span>
    <span class="stock-status">{"In Stock" if p["in_stock"] else "Out of Stock"}</span>
  </div>\n"""
        body_content += "</div>"

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

        query = f"Find the product where {filter_field} {filter_desc}. If multiple products match, use the first one listed. Extract its {extract_desc}."

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
            },
            metadata={
                "filter_field": filter_field,
                "filter_desc": filter_desc,
                "extract_field": extract_field,
                "num_products": num_products,
                "num_matching": len(matching),
                "html_style": style.value,
            },
        )


# =============================================================================
# Aggregation Archetype
# =============================================================================


@register(
    archetype_id="mvp.aggregation_min_max",
    category="hard",
    difficulty="hard",
    solvable=True,
    description="Extract all values, compute min/max/sum",
    tags=["extraction", "aggregation", "computation", "semantic"],
    phase=2,
    answer_schema=STRING_SCHEMA,
)
class AggregationMinMaxGenerator(Generator):
    """Generate tasks requiring aggregation over extracted values.

    Example: "Find the lowest price among all products"

    This requires:
    1. Extracting all price values
    2. Parsing them as numbers
    3. Computing the aggregate (min/max)
    4. Returning the result

    Tests actual data processing, not just extraction.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate products with prices
        num_products = rng.randint(5, 10)
        prices = []
        products = []

        for _i in range(num_products):
            price_val = round(rng.uniform(10, 200), 2)
            prices.append(price_val)
            products.append(
                {
                    "name": random_product_name(rng),
                    "price": price_val,
                    "price_formatted": f"${price_val:.2f}",
                }
            )

        # Choose aggregation type
        agg_types = [
            ("lowest", "minimum", min(prices)),
            ("highest", "maximum", max(prices)),
        ]

        agg_name, agg_desc, agg_value = rng.choice(agg_types)

        # Format ground truth
        ground_truth = f"${agg_value:.2f}"

        # Shuffle products
        rng.shuffle(products)

        # Build HTML
        body_content = '<div class="product-list">\n'
        for p in products:
            body_content += f"""  <div class="product">
    <span class="name">{p["name"]}</span>
    <span class="price">{p["price_formatted"]}</span>
  </div>\n"""
        body_content += "</div>"

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

        query = f'Extract all product prices and return the {agg_name} price. Return the full price string (e.g., "$XX.XX").'

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
            },
            metadata={
                "agg_type": agg_name,
                "num_products": num_products,
                "all_prices": [p["price_formatted"] for p in products],
                "html_style": style.value,
            },
        )


# =============================================================================
# Structured Output Archetype
# =============================================================================


PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "string"},
        "sku": {"type": "string"},
        "url": {"type": "string"},
    },
    "required": ["name", "price", "sku", "url"],
}


@register(
    archetype_id="mvp.structured_output",
    category="hard",
    difficulty="hard",
    solvable=True,
    description="Extract multiple consistent fields into structured object",
    tags=["extraction", "structured", "multi-field", "semantic"],
    phase=2,
    answer_schema=PRODUCT_SCHEMA,
)
class StructuredOutputGenerator(Generator):
    """Generate tasks requiring consistent multi-field extraction.

    Extract {name, price, sku, url} for a product. All fields must come
    from the SAME product, not mixed from different ones.

    This tests:
    1. Identifying the correct element boundary
    2. Extracting multiple related fields
    3. Keeping them consistent (all from same source)
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate products
        num_products = rng.randint(3, 6)
        products = []

        for _ in range(num_products):
            products.append(
                {
                    "name": random_product_name(rng),
                    "price": random_price(rng, min_val=20, max_val=200),
                    "sku": f"SKU-{rng.randint(10000, 99999)}",
                    "url": f"/products/{rng.randint(1000, 9999)}",
                }
            )

        # Select target using identifying hints (class-based, allows shuffling)
        identifying_hints = [
            ("featured", 0),  # Featured product (first with special class)
            ("recommended", rng.randint(0, num_products - 1)),  # Random recommended
        ]

        hint_type, target_idx = rng.choice(identifying_hints)
        target = products[target_idx]
        ground_truth = {
            "name": target["name"],
            "price": target["price"],
            "sku": target["sku"],
            "url": target["url"],
        }

        # Build HTML with identifier on target
        body_content = '<div class="product-list">\n'
        for i, p in enumerate(products):
            extra_class = ""
            if i == target_idx:
                extra_class = f" {hint_type}"

            body_content += f"""  <div class="product-card{extra_class}">
    <h3 class="product-name">{p["name"]}</h3>
    <a href="{p["url"]}" class="product-link">View Details</a>
    <span class="price">{p["price"]}</span>
    <span class="sku">{p["sku"]}</span>
  </div>\n"""
        body_content += "</div>"

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

        query = f'Extract the {hint_type} product\'s details. Return an object with "name", "price", "sku", and "url" fields.'

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=ground_truth,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema=PRODUCT_SCHEMA,
            normalization={
                "strip_whitespace": True,
                "sort_dict_keys": True,
            },
            metadata={
                "hint_type": hint_type,
                "target_idx": target_idx,
                "num_products": num_products,
                "html_style": style.value,
            },
        )


# =============================================================================
# Count/Existence Archetype
# =============================================================================


@register(
    archetype_id="mvp.count_elements",
    category="hard",
    difficulty="hard",
    solvable=True,
    description="Count elements matching a condition",
    tags=["extraction", "count", "aggregation", "semantic"],
    phase=2,
    answer_schema=INT_SCHEMA,
)
class CountElementsGenerator(Generator):
    """Generate tasks requiring element counting.

    Example: "How many products are in stock?"

    This requires:
    1. Finding all elements
    2. Filtering by condition
    3. Counting matches

    Tests iteration and conditional logic.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate products
        num_products = rng.randint(6, 12)
        products = []
        in_stock_count = 0
        high_rated_count = 0

        for _i in range(num_products):
            in_stock = rng.random() > 0.4
            rating = round(rng.uniform(2.5, 5.0), 1)

            if in_stock:
                in_stock_count += 1
            if rating >= 4.0:
                high_rated_count += 1

            products.append(
                {
                    "name": random_product_name(rng),
                    "in_stock": in_stock,
                    "rating": rating,
                }
            )

        # Choose count condition
        count_types = [
            ("in stock", "in-stock", in_stock_count),
            ("with rating 4.0 or higher", "high-rated", high_rated_count),
        ]

        count_desc, count_class, ground_truth = rng.choice(count_types)

        # Build HTML
        body_content = '<div class="product-list">\n'
        for p in products:
            stock_class = "in-stock" if p["in_stock"] else "out-of-stock"
            rating_class = "high-rated" if p["rating"] >= 4.0 else "low-rated"
            body_content += f"""  <div class="product {stock_class} {rating_class}">
    <span class="name">{p["name"]}</span>
    <span class="rating">{p["rating"]} stars</span>
    <span class="stock">{"In Stock" if p["in_stock"] else "Out of Stock"}</span>
  </div>\n"""
        body_content += "</div>"

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

        query = f"Count how many products are {count_desc}. Return just the number."

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=ground_truth,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema=INT_SCHEMA,
            normalization={},
            metadata={
                "count_type": count_class,
                "num_products": num_products,
                "html_style": style.value,
            },
        )


# =============================================================================
# Semantic Decoy Extreme Archetype
# =============================================================================


@register(
    archetype_id="mvp.semantic_decoy_extreme",
    category="hard",
    difficulty="hard",
    solvable=True,
    description="Extract from 5+ near-identical elements with subtle differentiators",
    tags=["extraction", "ambiguity", "decoy", "semantic"],
    phase=2,
    answer_schema=STRING_SCHEMA,
)
class SemanticDecoyExtremeGenerator(Generator):
    """Generate tasks with many near-identical decoy elements.

    Like the semantic_ambiguity archetype but with 5+ similar elements,
    each with subtle differences. The query specifies exactly which one
    is needed based on a small differentiator.

    This tests:
    1. Careful reading of query
    2. Precise element selection
    3. Not taking shortcuts
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate many similar items
        num_items = rng.randint(5, 8)
        base_name = rng.choice(
            [
                "Product",
                "Item",
                "Option",
                "Plan",
                "Package",
            ]
        )

        # Differentiators
        colors = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Black", "White"]
        sizes = ["Small", "Medium", "Large", "XL", "XXL"]

        items = []
        used_combos = set()

        for _i in range(num_items):
            # Ensure unique combinations
            while True:
                color = rng.choice(colors)
                size = rng.choice(sizes)
                combo = (color, size)
                if combo not in used_combos:
                    used_combos.add(combo)
                    break

            items.append(
                {
                    "name": f"{color} {size} {base_name}",
                    "color": color,
                    "size": size,
                    "price": random_price(rng, min_val=20, max_val=100),
                    "sku": f"SKU-{rng.randint(10000, 99999)}",
                }
            )

        # Select target
        target = rng.choice(items)

        # Shuffle
        rng.shuffle(items)

        # Build HTML with all items looking similar
        body_content = '<div class="item-list">\n'
        for item in items:
            body_content += f"""  <div class="item-card" data-color="{item["color"]}" data-size="{item["size"]}">
    <h3 class="item-name">{item["name"]}</h3>
    <span class="item-price">{item["price"]}</span>
    <span class="item-sku">{item["sku"]}</span>
  </div>\n"""
        body_content += "</div>"

        # Wrap with realistic chrome
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title=f"{base_name}s",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

        if rng.random() < 0.3:
            html = introduce_malformation(html, rng)

        # Query by specific attribute
        query_type = rng.choice(["color_and_size", "color", "size"])

        if query_type == "color_and_size":
            query = f'Extract the price of the {target["color"]} {target["size"]} {base_name}. There are many similar items - be precise.'
        elif query_type == "color":
            # Find items with same color
            same_color = [i for i in items if i["color"] == target["color"]]
            if len(same_color) > 1:
                query = f'Extract the price of the {target["color"]} {target["size"]} {base_name}. Multiple {target["color"]} items exist.'
            else:
                query = f'Extract the price of the {target["color"]} item.'
        else:
            query = f'Extract the price of the {target["color"]} {target["size"]} {base_name}.'

        ground_truth = target["price"]

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
            },
            metadata={
                "target_name": target["name"],
                "target_color": target["color"],
                "target_size": target["size"],
                "num_items": num_items,
                "html_style": style.value,
            },
        )
