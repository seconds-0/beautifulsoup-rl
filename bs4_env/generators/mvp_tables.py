"""Table parsing archetypes for BeautifulSoup RL environment.

This module implements tasks for extracting data from HTML tables,
a common and important scraping scenario.
"""

from bs4_env.config import DICT_LIST_SCHEMA, STRING_SCHEMA, TABLE_SCHEMA
from bs4_env.generators.base import (
    Generator,
    HtmlStyle,
    TaskInstance,
    add_noise_comments,
    make_rng,
    random_company_name,
    random_email,
    random_person_name,
    random_price,
    wrap_with_realistic_chrome,
)
from bs4_env.registry import register


@register(
    archetype_id="mvp.table_list_of_dicts",
    category="table_parsing",
    difficulty="medium",
    solvable=True,
    description="Convert a headered table into a list of dictionaries",
    tags=["table", "structured", "dict"],
    phase=1,
    answer_schema=DICT_LIST_SCHEMA,
)
class TableListOfDictsGenerator(Generator):
    """Generate tasks to convert tables to list of dicts.

    The model must:
    1. Identify the header row
    2. Use headers as dictionary keys
    3. Extract each data row as a dict
    4. Handle whitespace normalization
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        # Select random HTML style for realistic variation
        style = rng.choice(list(HtmlStyle))

        # Generate table data (ground truth FIRST)
        num_rows = rng.randint(3, 6)

        # Define columns
        columns = ["Name", "Company", "Email", "Price"]

        # Generate data
        data = []
        for _ in range(num_rows):
            row = {
                "Name": random_person_name(rng),
                "Company": random_company_name(rng),
                "Email": random_email(rng),
                "Price": random_price(rng),
            }
            data.append(row)

        # Build HTML table body content
        table_parts = [
            "<h1>Customer Data</h1>",
            '<table id="data-table" class="customers">',
            "<thead>",
            "<tr>",
        ]

        # Header row
        for col in columns:
            table_parts.append(f"<th>{col}</th>")
        table_parts.append("</tr>")
        table_parts.append("</thead>")

        # Data rows
        table_parts.append("<tbody>")
        for row in data:
            table_parts.append("<tr>")
            for col in columns:
                # Add some whitespace variation
                padding = " " * rng.randint(0, 2)
                table_parts.append(f"<td>{padding}{row[col]}{padding}</td>")
            table_parts.append("</tr>")
        table_parts.append("</tbody>")
        table_parts.append("</table>")

        body_content = "\n".join(table_parts)

        # Wrap with realistic chrome for real-world difficulty
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Customer Data",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )
        html = add_noise_comments(html, rng, count=2)

        query = (
            "Extract all rows from the table as a list of dictionaries. "
            "Use the table headers as keys."
        )

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=data,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema=DICT_LIST_SCHEMA,
            normalization={
                "strip_whitespace": True,
                "collapse_whitespace": True,
                "unicode_nfc": True,
                "sort_lists": False,  # Order matters for tables
                "sort_dict_keys": True,
            },
            metadata={
                "columns": columns,
                "row_count": num_rows,
            },
        )


@register(
    archetype_id="mvp.table_list_of_lists",
    category="table_parsing",
    difficulty="easy",
    solvable=True,
    description="Convert a table into a list of lists (rows)",
    tags=["table", "structured", "list"],
    phase=1,
    answer_schema=TABLE_SCHEMA,
)
class TableListOfListsGenerator(Generator):
    """Generate tasks to convert tables to list of lists.

    Simpler than dict conversion - just extract rows as arrays.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        # Select random HTML style for realistic variation
        style = rng.choice(list(HtmlStyle))

        # Generate simple numeric/text data
        num_rows = rng.randint(3, 5)
        num_cols = rng.randint(2, 4)

        # Generate data (list of lists)
        data = []
        for i in range(num_rows):
            row = []
            for j in range(num_cols):
                if j == 0:
                    row.append(f"Item {i + 1}")
                else:
                    row.append(str(rng.randint(10, 999)))
            data.append(row)

        # Build table content
        table_parts = ["<table>"]
        for row in data:
            table_parts.append("<tr>")
            for cell in row:
                table_parts.append(f"<td>{cell}</td>")
            table_parts.append("</tr>")
        table_parts.append("</table>")

        body_content = "\n".join(table_parts)

        # Wrap with realistic chrome for real-world difficulty
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Data Table",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )

        query = "Extract all table rows as a list of lists."

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=data,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=True,
            answer_schema=TABLE_SCHEMA,
            normalization={
                "strip_whitespace": True,
                "collapse_whitespace": True,
                "sort_lists": False,
            },
            metadata={
                "row_count": num_rows,
                "col_count": num_cols,
            },
        )


@register(
    archetype_id="mvp.table_rowspan",
    category="table_parsing",
    difficulty="hard",
    solvable=True,
    description="Extract data from complex tables with rowspan and colspan attributes",
    tags=["table", "rowspan", "colspan", "complex"],
    phase=2,
    answer_schema=STRING_SCHEMA,
)
class TableRowspanGenerator(Generator):
    """Generate tasks with complex tables using rowspan/colspan.

    This tests the model's ability to understand cell spanning in tables:
    1. Cells with rowspan span multiple rows vertically
    2. Cells with colspan span multiple columns horizontally
    3. The model must track which cell belongs to which logical row/column

    Example difficulty: Given a table where "Category A" spans 3 rows,
    extract the value in the "Price" column for the row labeled "Item 2".
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)
        style = rng.choice(list(HtmlStyle))

        # Generate categories and items
        categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
        category = rng.choice(categories)

        # Generate 3-5 items in this category
        num_items = rng.randint(3, 5)
        items = []
        for i in range(num_items):
            items.append(
                {
                    "name": f"Item {i + 1}",
                    "price": f"${rng.randint(10, 200)}.{rng.randint(0, 99):02d}",
                    "stock": str(rng.randint(0, 500)),
                }
            )

        # Pick which item's price to ask for
        target_idx = rng.randint(0, num_items - 1)
        target_item = items[target_idx]
        ground_truth = target_item["price"]

        # Build table with rowspan for category
        table_parts = [
            '<table class="product-table" border="1">',
            "<thead>",
            "<tr><th>Category</th><th>Product</th><th>Price</th><th>Stock</th></tr>",
            "</thead>",
            "<tbody>",
        ]

        # First row has rowspan for category
        table_parts.append("<tr>")
        table_parts.append(f'<td rowspan="{num_items}">{category}</td>')
        table_parts.append(f"<td>{items[0]['name']}</td>")
        table_parts.append(f"<td>{items[0]['price']}</td>")
        table_parts.append(f"<td>{items[0]['stock']}</td>")
        table_parts.append("</tr>")

        # Remaining rows don't have category column (spanned)
        for item in items[1:]:
            table_parts.append("<tr>")
            table_parts.append(f"<td>{item['name']}</td>")
            table_parts.append(f"<td>{item['price']}</td>")
            table_parts.append(f"<td>{item['stock']}</td>")
            table_parts.append("</tr>")

        table_parts.append("</tbody>")
        table_parts.append("</table>")

        body_content = "\n".join(table_parts)

        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Product Inventory",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )
        html = add_noise_comments(html, rng, count=2)

        query = (
            f'In the table, find the row for "{target_item["name"]}" '
            f'in the "{category}" category and extract its Price value.'
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
                "category": category,
                "target_item": target_item["name"],
                "target_idx": target_idx,
                "num_items": num_items,
                "has_rowspan": True,
            },
        )
