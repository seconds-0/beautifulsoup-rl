"""JSON-LD structured data extraction archetype for BeautifulSoup RL environment.

This module tests extraction from JSON-LD structured data, which is one of
the most common patterns on real websites (Walmart, Newegg, Redfin, AllRecipes).

JSON-LD (JavaScript Object Notation for Linked Data) is embedded in HTML as:
    <script type="application/ld+json">
    {
        "@type": "Product",
        "name": "Example Product",
        "price": "29.99"
    }
    </script>

Extraction pattern:
    scripts = soup.find_all("script", type="application/ld+json")
    data = json.loads(scripts[0].string)
    name = data["name"]

This is often the CLEANEST way to extract structured data from e-commerce,
recipe, and article pages.
"""

import json

from bs4_env.config import STRING_SCHEMA
from bs4_env.generators.base import (
    Generator,
    HtmlStyle,
    TaskInstance,
    add_noise_comments,
    generate_variable_content,
    make_rng,
    wrap_with_realistic_chrome,
)
from bs4_env.registry import register

# Schema.org types commonly found in JSON-LD
SCHEMA_TYPES = {
    "Product": {
        "required": ["name", "description"],
        "optional": ["brand", "sku", "price", "priceCurrency", "availability", "image"],
    },
    "Article": {
        "required": ["headline", "author", "datePublished"],
        "optional": ["description", "image", "publisher", "dateModified"],
    },
    "Recipe": {
        "required": ["name", "recipeIngredient", "recipeInstructions"],
        "optional": ["prepTime", "cookTime", "totalTime", "recipeYield", "author"],
    },
    "LocalBusiness": {
        "required": ["name", "address"],
        "optional": ["telephone", "openingHours", "priceRange", "image"],
    },
    "Organization": {
        "required": ["name", "url"],
        "optional": ["logo", "description", "contactPoint", "sameAs"],
    },
}


def generate_json_ld_data(rng, schema_type: str) -> dict:
    """Generate realistic JSON-LD data for a schema type.

    Args:
        rng: Random instance.
        schema_type: The Schema.org type to generate.

    Returns:
        Dictionary with JSON-LD data.
    """
    SCHEMA_TYPES[schema_type]
    data = {
        "@context": "https://schema.org",
        "@type": schema_type,
    }

    if schema_type == "Product":
        data["name"] = rng.choice(
            [
                "Premium Wireless Headphones",
                "Ergonomic Office Chair",
                "Stainless Steel Water Bottle",
                "Organic Cotton T-Shirt",
                "Smart Home Hub Controller",
            ]
        )
        data["description"] = generate_variable_content(rng, min_sentences=1, max_sentences=2)
        data["brand"] = {
            "@type": "Brand",
            "name": rng.choice(["TechPro", "HomeLife", "EcoWear", "SmartGear"]),
        }
        data["sku"] = f"SKU-{rng.randint(10000, 99999)}"
        data["offers"] = {
            "@type": "Offer",
            "price": f"{rng.randint(10, 500)}.{rng.randint(0, 99):02d}",
            "priceCurrency": "USD",
            "availability": rng.choice(
                ["https://schema.org/InStock", "https://schema.org/OutOfStock"]
            ),
        }

    elif schema_type == "Article":
        data["headline"] = rng.choice(
            [
                "Breaking News: Major Discovery Announced",
                "How to Improve Your Productivity in 5 Steps",
                "The Future of Technology: What Experts Say",
                "Local Community Celebrates Annual Festival",
            ]
        )
        data["author"] = {
            "@type": "Person",
            "name": rng.choice(["Jane Smith", "John Doe", "Alex Johnson", "Sam Brown"]),
        }
        data["datePublished"] = f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        data["description"] = generate_variable_content(rng, min_sentences=1, max_sentences=2)
        data["publisher"] = {
            "@type": "Organization",
            "name": rng.choice(["Daily News", "Tech Weekly", "Local Times"]),
        }

    elif schema_type == "Recipe":
        data["name"] = rng.choice(
            [
                "Classic Chocolate Chip Cookies",
                "Homemade Margherita Pizza",
                "Easy Chicken Stir-Fry",
                "Creamy Mushroom Risotto",
            ]
        )
        data["author"] = {
            "@type": "Person",
            "name": rng.choice(["Chef Maria", "Cook Bob", "Baker Lisa"]),
        }
        data["recipeIngredient"] = [
            f"{rng.randint(1, 3)} cups flour",
            f"{rng.randint(1, 2)} tsp baking soda",
            f"{rng.randint(100, 300)}g butter",
            f"{rng.randint(1, 3)} eggs",
        ]
        data["recipeInstructions"] = [
            {"@type": "HowToStep", "text": "Preheat oven to 350°F."},
            {"@type": "HowToStep", "text": "Mix dry ingredients."},
            {"@type": "HowToStep", "text": "Add wet ingredients."},
            {"@type": "HowToStep", "text": "Bake for 12 minutes."},
        ]
        data["prepTime"] = f"PT{rng.randint(10, 30)}M"
        data["cookTime"] = f"PT{rng.randint(15, 45)}M"
        data["recipeYield"] = f"{rng.randint(4, 12)} servings"

    elif schema_type == "LocalBusiness":
        data["name"] = rng.choice(
            ["Joe's Coffee Shop", "Main Street Bookstore", "Green Leaf Restaurant"]
        )
        data["address"] = {
            "@type": "PostalAddress",
            "streetAddress": f"{rng.randint(100, 999)} Main Street",
            "addressLocality": rng.choice(["Springfield", "Oakland", "Portland"]),
            "addressRegion": rng.choice(["CA", "OR", "WA"]),
            "postalCode": f"{rng.randint(10000, 99999)}",
        }
        data["telephone"] = (
            f"+1-{rng.randint(200, 999)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
        )

    elif schema_type == "Organization":
        data["name"] = rng.choice(["TechCorp Inc.", "Global Solutions Ltd.", "Innovation Labs"])
        data["url"] = f"https://www.{data['name'].lower().replace(' ', '').replace('.', '')}.com"
        data["description"] = generate_variable_content(rng, min_sentences=1, max_sentences=1)

    return data


def get_nested_value(data: dict, path: str) -> str:
    """Get a nested value from JSON-LD data by path.

    Args:
        data: The JSON-LD dictionary.
        path: Dot-separated path like "offers.price" or "author.name".

    Returns:
        String value at the path.
    """
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        else:
            return str(current) if current else ""
    return str(current) if current else ""


# Extraction paths per schema type (what we'll ask to extract)
EXTRACTION_PATHS = {
    "Product": [
        ("name", "Extract the product name"),
        ("offers.price", "Extract the product price"),
        ("brand.name", "Extract the brand name"),
        ("sku", "Extract the SKU"),
    ],
    "Article": [
        ("headline", "Extract the article headline"),
        ("author.name", "Extract the author's name"),
        ("datePublished", "Extract the publication date"),
        ("publisher.name", "Extract the publisher name"),
    ],
    "Recipe": [
        ("name", "Extract the recipe name"),
        ("author.name", "Extract the recipe author"),
        ("prepTime", "Extract the prep time"),
        ("recipeYield", "Extract the serving size"),
    ],
    "LocalBusiness": [
        ("name", "Extract the business name"),
        ("telephone", "Extract the phone number"),
        ("address.streetAddress", "Extract the street address"),
        ("address.addressLocality", "Extract the city"),
    ],
    "Organization": [
        ("name", "Extract the organization name"),
        ("url", "Extract the website URL"),
    ],
}


@register(
    archetype_id="mvp.json_ld_extraction",
    category="structured_data",
    difficulty="medium",
    solvable=True,
    description="Extract data from JSON-LD structured data in script tags",
    tags=["extraction", "json-ld", "structured-data", "script"],
    phase=1,
    answer_schema=STRING_SCHEMA,
)
class JsonLdExtractionGenerator(Generator):
    """Generate tasks testing JSON-LD structured data extraction.

    JSON-LD is one of the most common ways real websites provide structured
    data. This archetype tests the model's ability to:
    1. Find script tags with type="application/ld+json"
    2. Parse the JSON content
    3. Navigate nested structures to extract specific fields

    This pattern is used by Walmart, Newegg, Redfin, AllRecipes, and many
    other major websites.
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
            TaskInstance with JSON-LD extraction task.
        """
        rng = make_rng(self.archetype_id, seed)

        # Select style randomly if not specified
        if style is None:
            style = rng.choice(list(HtmlStyle))

        # Choose a schema type
        schema_type = rng.choice(list(SCHEMA_TYPES.keys()))

        # Generate the JSON-LD data
        json_ld_data = generate_json_ld_data(rng, schema_type)

        # Choose what to extract
        path, query_template = rng.choice(EXTRACTION_PATHS[schema_type])
        ground_truth = get_nested_value(json_ld_data, path)

        # Generate JSON-LD script tag
        json_ld_script = (
            f'<script type="application/ld+json">\n{json.dumps(json_ld_data, indent=2)}\n</script>'
        )

        # Sometimes add multiple JSON-LD scripts (like Redfin with 60!)
        num_extra_scripts = rng.randint(0, 3)
        extra_scripts = []
        for _ in range(num_extra_scripts):
            extra_type = rng.choice([t for t in SCHEMA_TYPES if t != schema_type])
            extra_data = generate_json_ld_data(rng, extra_type)
            extra_scripts.append(
                f'<script type="application/ld+json">\n{json.dumps(extra_data, indent=2)}\n</script>'
            )

        # Generate visible page content (the JSON-LD might have DIFFERENT data!)
        # This tests that the model extracts from JSON-LD, not visible HTML
        visible_content = generate_variable_content(rng, min_sentences=2, max_sentences=4)

        body_content = f"""
<div class="page-content">
    <h1>Page Title</h1>
    <p>{visible_content}</p>
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

        # Insert JSON-LD scripts in the head
        # Find </head> and insert before it
        head_end = html.find("</head>")
        if head_end != -1:
            all_scripts = json_ld_script + "\n" + "\n".join(extra_scripts)
            html = html[:head_end] + all_scripts + "\n" + html[head_end:]

        html = add_noise_comments(html, rng, count=2)

        # Build query
        query = f'{query_template} from the JSON-LD structured data (found in script type="application/ld+json").'

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
                "schema_type": schema_type,
                "extraction_path": path,
                "num_json_ld_scripts": 1 + num_extra_scripts,
                "html_style": style.value,
                "pattern": 'soup.find("script", type="application/ld+json")',
                "json_ld_keys": list(json_ld_data.keys()),
            },
        )
