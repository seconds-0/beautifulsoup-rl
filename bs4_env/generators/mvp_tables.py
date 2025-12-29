"""Table parsing archetypes for BeautifulSoup RL environment.

This module implements tasks for extracting data from HTML tables,
a common and important scraping scenario.
"""

from bs4_env.config import DICT_LIST_SCHEMA, TABLE_SCHEMA
from bs4_env.generators.base import (
    Generator,
    HtmlStyle,
    TaskInstance,
    make_rng,
    random_person_name,
    random_company_name,
    random_price,
    random_email,
    add_noise_comments,
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
