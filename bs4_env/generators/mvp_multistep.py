"""Multi-step archetypes for BeautifulSoup RL environment.

This module implements tasks requiring navigation between multiple pages:
- Search → Detail: Find item in list, navigate to detail page, extract data
- Pagination: Navigate through paginated results, aggregate data
- Link Chain: Follow breadcrumb/link chain to destination

These archetypes test planning, navigation, and multi-step reasoning.
They use the navigate tool to switch between pre-generated HTML pages.
"""

from bs4_env.config import (
    STRING_SCHEMA,
    LIST_SCHEMA,
    FLOAT_SCHEMA,
    INT_SCHEMA,
)
from bs4_env.generators.base import (
    Generator,
    HtmlStyle,
    TaskInstance,
    make_rng,
    random_id,
    random_class_name,
    random_price,
    random_product_name,
    random_person_name,
    wrap_with_realistic_chrome,
)
from bs4_env.registry import register


# =============================================================================
# Search Then Detail Archetype
# =============================================================================


@register(
    archetype_id="mvp.search_then_detail",
    category="hard",
    difficulty="hard",
    solvable=True,
    description="Find item in search results, navigate to detail page, extract data",
    tags=["extraction", "multi-step", "navigation", "search"],
    phase=2,
    answer_schema=STRING_SCHEMA,
)
class SearchThenDetailGenerator(Generator):
    """Generate multi-step search→detail tasks.

    The model must:
    1. Parse the search results page
    2. Find the correct item based on the query
    3. Use the navigate tool to go to the detail page
    4. Extract the requested information from the detail page

    This tests planning and multi-step execution.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate products for search results
        num_products = rng.randint(5, 10)
        products = []

        for i in range(num_products):
            product_id = rng.randint(1000, 9999)
            products.append({
                "id": product_id,
                "name": random_product_name(rng),
                "price": random_price(rng, min_val=20, max_val=200),
                "sku": f"SKU-{rng.randint(10000, 99999)}",
                "description": f"High-quality {random_product_name(rng).lower()} with premium features.",
                "manufacturer": rng.choice([
                    "TechCorp", "GlobalGoods", "PremiumBrands", "QualityFirst",
                    "InnovateCo", "ModernMakers", "EliteProducts"
                ]),
                "warranty": f"{rng.randint(1, 5)} years",
                "href": f"/products/{product_id}",
            })

        # Select target product
        target = rng.choice(products)

        # Choose what to extract from detail page
        detail_fields = [
            ("manufacturer", "manufacturer name"),
            ("warranty", "warranty period"),
            ("sku", "SKU"),
        ]
        extract_field, extract_desc = rng.choice(detail_fields)
        ground_truth = target[extract_field]

        # Shuffle products for search results
        rng.shuffle(products)

        # Build search results page
        search_html = self._build_search_page(products, style, rng)

        # Build detail pages for all products
        pages = {}
        for p in products:
            pages[p["href"]] = self._build_detail_page(p, style, rng)

        # Wrap search page with chrome
        html = wrap_with_realistic_chrome(
            search_html,
            style,
            rng,
            title="Search Results",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

        query = (
            f'Find the product named "{target["name"]}" in the search results. '
            f'Navigate to its detail page and extract the {extract_desc}.'
        )

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
                "target_href": target["href"],
                "extract_field": extract_field,
                "num_products": num_products,
                "html_style": style.value,
            },
            pages=pages,
        )

    def _build_search_page(self, products: list, style: HtmlStyle, rng) -> str:
        """Build the search results listing page."""
        content = '<div class="search-results">\n'
        content += f'  <h1>Search Results ({len(products)} items)</h1>\n'

        for p in products:
            content += f'''  <div class="product-result" data-id="{p["id"]}">
    <h3 class="product-title"><a href="{p["href"]}">{p["name"]}</a></h3>
    <span class="price">{p["price"]}</span>
    <p class="snippet">{p["description"][:50]}...</p>
    <a href="{p["href"]}" class="view-details">View Details</a>
  </div>\n'''

        content += '</div>'
        return content

    def _build_detail_page(self, product: dict, style: HtmlStyle, rng) -> str:
        """Build a product detail page."""
        content = f'''<div class="product-detail">
  <h1 class="product-name">{product["name"]}</h1>
  <div class="product-info">
    <span class="price">{product["price"]}</span>
    <span class="sku">SKU: {product["sku"]}</span>
  </div>
  <div class="product-description">
    <p>{product["description"]}</p>
  </div>
  <div class="product-specs">
    <dl>
      <dt>Manufacturer</dt>
      <dd class="manufacturer">{product["manufacturer"]}</dd>
      <dt>Warranty</dt>
      <dd class="warranty">{product["warranty"]}</dd>
    </dl>
  </div>
  <div class="actions">
    <button class="add-to-cart">Add to Cart</button>
  </div>
</div>'''

        return wrap_with_realistic_chrome(
            content,
            style,
            rng,
            title=product["name"],
            complexity="minimal",
            include_nav=True,
            include_footer=False,
        )


# =============================================================================
# Pagination Aggregate Archetype
# =============================================================================


@register(
    archetype_id="mvp.pagination_aggregate",
    category="hard",
    difficulty="hard",
    solvable=True,
    description="Navigate through paginated results and aggregate data",
    tags=["extraction", "multi-step", "navigation", "pagination", "aggregation"],
    phase=2,
    answer_schema=STRING_SCHEMA,
)
class PaginationAggregateGenerator(Generator):
    """Generate multi-page pagination tasks.

    The model must:
    1. Parse the first page of results
    2. Navigate through all pages
    3. Collect data from each page
    4. Aggregate to produce final answer

    This tests multi-step planning and data aggregation.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate products across multiple pages
        num_pages = rng.randint(2, 4)
        items_per_page = rng.randint(3, 5)
        total_items = num_pages * items_per_page

        all_products = []
        for i in range(total_items):
            price_val = round(rng.uniform(10, 200), 2)
            all_products.append({
                "name": random_product_name(rng),
                "price_val": price_val,
                "price": f"${price_val:.2f}",
                "in_stock": rng.random() > 0.3,
            })

        # Choose aggregation task
        agg_types = [
            ("lowest", min, lambda p: p["price_val"]),
            ("highest", max, lambda p: p["price_val"]),
        ]
        agg_name, agg_fn, key_fn = rng.choice(agg_types)

        # Compute ground truth across ALL pages
        agg_value = agg_fn(all_products, key=key_fn)["price_val"]
        ground_truth = f"${agg_value:.2f}"

        # Build pages
        pages = {}
        for page_num in range(1, num_pages + 1):
            start_idx = (page_num - 1) * items_per_page
            end_idx = start_idx + items_per_page
            page_products = all_products[start_idx:end_idx]

            page_html = self._build_results_page(
                page_products, page_num, num_pages, items_per_page, style, rng
            )

            if page_num == 1:
                # First page is the initial HTML
                initial_html = page_html
            else:
                pages[f"/results?page={page_num}"] = page_html

        # Wrap initial page with chrome
        html = wrap_with_realistic_chrome(
            initial_html,
            style,
            rng,
            title="Products - Page 1",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

        query = (
            f"Find the {agg_name} priced product across ALL pages of results. "
            f"There are {num_pages} pages total. Navigate through each page, "
            f"collect all prices, and return the {agg_name} price (e.g., \"$XX.XX\")."
        )

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
                "num_pages": num_pages,
                "items_per_page": items_per_page,
                "total_items": total_items,
                "html_style": style.value,
            },
            pages=pages,
        )

    def _build_results_page(
        self, products: list, page_num: int, total_pages: int,
        items_per_page: int, style: HtmlStyle, rng
    ) -> str:
        """Build a single page of results."""
        content = f'<div class="results-page" data-page="{page_num}">\n'
        content += f'  <h1>Products (Page {page_num} of {total_pages})</h1>\n'
        content += '  <div class="product-list">\n'

        for p in products:
            stock_class = "in-stock" if p["in_stock"] else "out-of-stock"
            content += f'''    <div class="product {stock_class}">
      <span class="name">{p["name"]}</span>
      <span class="price">{p["price"]}</span>
    </div>\n'''

        content += '  </div>\n'

        # Add pagination links
        content += '  <div class="pagination">\n'
        for i in range(1, total_pages + 1):
            if i == page_num:
                content += f'    <span class="current-page">{i}</span>\n'
            else:
                content += f'    <a href="/results?page={i}" class="page-link">{i}</a>\n'
        content += '  </div>\n'
        content += '</div>'

        return content


# =============================================================================
# Link Chain Archetype
# =============================================================================


@register(
    archetype_id="mvp.link_chain",
    category="hard",
    difficulty="hard",
    solvable=True,
    description="Follow a chain of links to reach destination and extract data",
    tags=["extraction", "multi-step", "navigation", "chain"],
    phase=2,
    answer_schema=STRING_SCHEMA,
)
class LinkChainGenerator(Generator):
    """Generate link-following tasks.

    The model must:
    1. Parse the initial page
    2. Find the correct link to follow
    3. Repeat on each intermediate page
    4. Extract data from the final destination

    This tests multi-step navigation and link following.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate chain structure
        chain_length = rng.randint(2, 3)

        # Generate category hierarchy
        categories = [
            ("Electronics", ["Computers", "Phones", "Audio", "Cameras"]),
            ("Home & Garden", ["Furniture", "Kitchen", "Outdoor", "Decor"]),
            ("Sports", ["Fitness", "Outdoor", "Team Sports", "Water Sports"]),
            ("Fashion", ["Men", "Women", "Kids", "Accessories"]),
        ]

        top_cat, subcats = rng.choice(categories)
        subcat = rng.choice(subcats)

        # Generate final product
        final_product = {
            "name": random_product_name(rng),
            "price": random_price(rng, min_val=50, max_val=500),
            "sku": f"SKU-{rng.randint(10000, 99999)}",
            "rating": round(rng.uniform(3.5, 5.0), 1),
        }

        # Choose what to extract
        extract_options = [
            ("price", final_product["price"]),
            ("sku", final_product["sku"]),
            ("rating", str(final_product["rating"])),
        ]
        extract_field, ground_truth = rng.choice(extract_options)

        # Build chain of pages
        pages = {}

        # Page 1: Category listing (links to subcategory)
        subcat_href = f"/{top_cat.lower().replace(' ', '-')}/{subcat.lower().replace(' ', '-')}"

        # Page 2: Subcategory (links to product)
        product_href = f"{subcat_href}/{rng.randint(1000, 9999)}"

        # Build intermediate pages
        pages[subcat_href] = self._build_subcategory_page(
            subcat, [final_product], product_href, style, rng
        )
        pages[product_href] = self._build_product_page(final_product, style, rng)

        # Build initial page (category listing)
        initial_html = self._build_category_page(
            top_cat, subcats, subcat, subcat_href, style, rng
        )

        html = wrap_with_realistic_chrome(
            initial_html,
            style,
            rng,
            title=f"{top_cat} - Categories",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

        query = (
            f'Navigate to the "{subcat}" subcategory under "{top_cat}", '
            f'then go to the product "{final_product["name"]}" and extract its {extract_field}.'
        )

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
                "top_category": top_cat,
                "subcategory": subcat,
                "product_name": final_product["name"],
                "extract_field": extract_field,
                "chain_length": chain_length,
                "html_style": style.value,
            },
            pages=pages,
        )

    def _build_category_page(
        self, top_cat: str, subcats: list, target_subcat: str,
        target_href: str, style: HtmlStyle, rng
    ) -> str:
        """Build the top-level category page."""
        content = f'<div class="category-page">\n'
        content += f'  <h1>{top_cat}</h1>\n'
        content += '  <nav class="subcategories">\n'
        content += '    <h2>Browse Categories</h2>\n'
        content += '    <ul>\n'

        for subcat in subcats:
            if subcat == target_subcat:
                href = target_href
            else:
                href = f"/{top_cat.lower().replace(' ', '-')}/{subcat.lower().replace(' ', '-')}"

            content += f'      <li><a href="{href}">{subcat}</a></li>\n'

        content += '    </ul>\n'
        content += '  </nav>\n'
        content += '</div>'

        return content

    def _build_subcategory_page(
        self, subcat: str, products: list, product_href: str,
        style: HtmlStyle, rng
    ) -> str:
        """Build a subcategory page with product listings."""
        content = f'<div class="subcategory-page">\n'
        content += f'  <h1>{subcat}</h1>\n'
        content += '  <div class="product-listings">\n'

        # Add some decoy products
        decoy_count = rng.randint(2, 4)
        all_products = []
        for _ in range(decoy_count):
            all_products.append({
                "name": random_product_name(rng),
                "price": random_price(rng),
                "href": f"/products/{rng.randint(1000, 9999)}",
                "is_target": False,
            })

        # Add target product
        target = products[0]
        all_products.append({
            "name": target["name"],
            "price": target["price"],
            "href": product_href,
            "is_target": True,
        })

        rng.shuffle(all_products)

        for p in all_products:
            content += f'''    <div class="product-card">
      <h3><a href="{p["href"]}">{p["name"]}</a></h3>
      <span class="price">{p["price"]}</span>
    </div>\n'''

        content += '  </div>\n'
        content += '</div>'

        return wrap_with_realistic_chrome(
            content,
            style,
            rng,
            title=subcat,
            complexity="minimal",
            include_nav=True,
            include_footer=False,
        )

    def _build_product_page(self, product: dict, style: HtmlStyle, rng) -> str:
        """Build the final product page."""
        content = f'''<div class="product-detail">
  <h1 class="product-name">{product["name"]}</h1>
  <div class="product-info">
    <span class="price">{product["price"]}</span>
    <span class="sku">SKU: {product["sku"]}</span>
    <span class="rating">{product["rating"]} out of 5 stars</span>
  </div>
  <div class="actions">
    <button class="add-to-cart">Add to Cart</button>
    <button class="wishlist">Add to Wishlist</button>
  </div>
</div>'''

        return wrap_with_realistic_chrome(
            content,
            style,
            rng,
            title=product["name"],
            complexity="minimal",
            include_nav=True,
            include_footer=False,
        )


# =============================================================================
# Compare Products Archetype (Multi-page comparison)
# =============================================================================


COMPARISON_SCHEMA = {
    "type": "object",
    "properties": {
        "cheaper": {"type": "string"},
        "price_difference": {"type": "string"},
    },
    "required": ["cheaper", "price_difference"],
}


@register(
    archetype_id="mvp.compare_products",
    category="hard",
    difficulty="hard",
    solvable=True,
    description="Navigate to multiple product pages and compare their attributes",
    tags=["extraction", "multi-step", "navigation", "comparison"],
    phase=2,
    answer_schema=COMPARISON_SCHEMA,
)
class CompareProductsGenerator(Generator):
    """Generate product comparison tasks.

    The model must:
    1. Navigate to multiple product pages
    2. Extract specific attributes from each
    3. Compare and compute the answer

    This tests multi-page navigation and comparison logic.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate two products to compare
        products = []
        for i in range(2):
            price_val = round(rng.uniform(50, 300), 2)
            products.append({
                "name": random_product_name(rng),
                "price_val": price_val,
                "price": f"${price_val:.2f}",
                "href": f"/products/{rng.randint(1000, 9999)}",
            })

        # Determine which is cheaper
        if products[0]["price_val"] < products[1]["price_val"]:
            cheaper = products[0]
            more_expensive = products[1]
        else:
            cheaper = products[1]
            more_expensive = products[0]

        diff = abs(products[0]["price_val"] - products[1]["price_val"])
        ground_truth = {
            "cheaper": cheaper["name"],
            "price_difference": f"${diff:.2f}",
        }

        # Build comparison page listing both products
        comparison_html = '<div class="comparison">\n'
        comparison_html += '  <h1>Compare Products</h1>\n'
        comparison_html += '  <div class="products-to-compare">\n'
        for p in products:
            comparison_html += f'''    <div class="compare-item">
      <h3><a href="{p["href"]}">{p["name"]}</a></h3>
      <p class="hint">Click to see full details and price</p>
    </div>\n'''
        comparison_html += '  </div>\n'
        comparison_html += '</div>'

        # Build detail pages
        pages = {}
        for p in products:
            pages[p["href"]] = self._build_detail_page(p, style, rng)

        html = wrap_with_realistic_chrome(
            comparison_html,
            style,
            rng,
            title="Product Comparison",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

        query = (
            f'Compare the prices of "{products[0]["name"]}" and "{products[1]["name"]}". '
            f'Navigate to each product\'s detail page to find the price. '
            f'Return which product is cheaper and the price difference.'
        )

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=ground_truth,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema=COMPARISON_SCHEMA,
            normalization={
                "strip_whitespace": True,
                "sort_dict_keys": True,
            },
            metadata={
                "product1": products[0]["name"],
                "product2": products[1]["name"],
                "cheaper": cheaper["name"],
                "price_diff": diff,
                "html_style": style.value,
            },
            pages=pages,
        )

    def _build_detail_page(self, product: dict, style: HtmlStyle, rng) -> str:
        """Build a product detail page."""
        content = f'''<div class="product-detail">
  <h1 class="product-name">{product["name"]}</h1>
  <div class="product-info">
    <span class="price">{product["price"]}</span>
  </div>
  <div class="actions">
    <button class="add-to-cart">Add to Cart</button>
  </div>
</div>'''

        return wrap_with_realistic_chrome(
            content,
            style,
            rng,
            title=product["name"],
            complexity="minimal",
            include_nav=True,
            include_footer=False,
        )
