"""Limitation detection archetypes for BeautifulSoup RL environment.

These tasks are intentionally UNSOLVABLE with static BeautifulSoup parsing.
The correct behavior is to recognize the limitation and abstain with evidence.
"""

from bs4_env.config import STRING_SCHEMA
from bs4_env.generators.base import (
    Generator,
    HtmlStyle,
    TaskInstance,
    make_rng,
    random_paragraph,
    random_product_name,
    random_price,
    add_noise_comments,
    wrap_with_realistic_chrome,
)
from bs4_env.registry import register


@register(
    archetype_id="mvp.limit_js_required",
    category="limitations",
    difficulty="medium",
    solvable=False,  # This is a limitation task!
    description="Content requires JavaScript execution to render",
    tags=["limitation", "javascript", "dynamic"],
    phase=1,
    answer_schema=STRING_SCHEMA,
    allowed_limit_reasons=["js_required", "javascript_required", "dynamic_content"],
    evidence_patterns=[
        r"<script[^>]*>",
        r"document\.(getElementById|querySelector|write)",
        r"innerHTML",
        r"render\w*\(",
        r"hydrate",
        r"ReactDOM",
        r"Vue\.",
        r"ng-app",
    ],
)
class JSRequiredGenerator(Generator):
    """Generate tasks where content is JavaScript-rendered.

    The HTML contains a placeholder that would be filled by JS.
    The actual content exists only in a script tag or data attribute.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        # The "real" content that would be rendered by JS
        product_name = random_product_name(rng)
        product_price = random_price(rng)

        # Choose a JS framework/pattern
        patterns = [
            self._react_pattern,
            self._vanilla_js_pattern,
            self._vue_pattern,
            self._data_fetch_pattern,
        ]
        pattern_fn = rng.choice(patterns)
        html, evidence_hint = pattern_fn(rng, product_name, product_price)

        query = (
            "Extract the product name and price from this page. "
            "The product information should be visible on the page."
        )

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=None,  # No correct extraction answer
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=False,
            answer_schema=STRING_SCHEMA,
            limit_info={
                "allowed_reasons": ["js_required", "javascript_required", "dynamic_content"],
                "evidence_patterns": [
                    r"<script",
                    r"getElementById",
                    r"innerHTML",
                    r"ReactDOM",
                    r"createApp",
                ],
            },
            metadata={
                "actual_product": product_name,
                "actual_price": product_price,
                "pattern": pattern_fn.__name__,
                "evidence_hint": evidence_hint,
            },
        )

    def _react_pattern(self, rng, product_name: str, price: str) -> tuple[str, str]:
        """Generate React-style JS rendering."""
        html = f"""<!DOCTYPE html>
<html>
<head><title>Product Page</title></head>
<body>
<div id="root">
    <div class="loading">Loading product...</div>
</div>
<script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script>
const productData = {{
    name: "{product_name}",
    price: "{price}"
}};
ReactDOM.render(
    React.createElement('div', null,
        React.createElement('h1', null, productData.name),
        React.createElement('span', {{className: 'price'}}, productData.price)
    ),
    document.getElementById('root')
);
</script>
</body>
</html>"""
        return html, "ReactDOM.render"

    def _vanilla_js_pattern(self, rng, product_name: str, price: str) -> tuple[str, str]:
        """Generate vanilla JS DOM manipulation."""
        html = f"""<!DOCTYPE html>
<html>
<head><title>Product Page</title></head>
<body>
<div id="product-container">
    <p class="placeholder">Product information loading...</p>
</div>
<script>
window.onload = function() {{
    var container = document.getElementById('product-container');
    container.innerHTML = '<h1>{product_name}</h1><span class="price">{price}</span>';
}};
</script>
</body>
</html>"""
        return html, "document.getElementById"

    def _vue_pattern(self, rng, product_name: str, price: str) -> tuple[str, str]:
        """Generate Vue.js style rendering."""
        html = f"""<!DOCTYPE html>
<html>
<head><title>Product Page</title></head>
<body>
<div id="app">
    <div v-if="loading">Loading...</div>
    <div v-else>
        <h1>{{{{ product.name }}}}</h1>
        <span class="price">{{{{ product.price }}}}</span>
    </div>
</div>
<script src="https://unpkg.com/vue@3"></script>
<script>
Vue.createApp({{
    data() {{
        return {{
            loading: false,
            product: {{ name: '{product_name}', price: '{price}' }}
        }}
    }}
}}).mount('#app');
</script>
</body>
</html>"""
        return html, "Vue.createApp"

    def _data_fetch_pattern(self, rng, product_name: str, price: str) -> tuple[str, str]:
        """Generate async data fetch pattern."""
        html = f"""<!DOCTYPE html>
<html>
<head><title>Product Page</title></head>
<body>
<main id="content">
    <div class="skeleton">Loading product data...</div>
</main>
<script>
async function loadProduct() {{
    // Simulated API response
    const data = {{ name: "{product_name}", price: "{price}" }};
    document.getElementById('content').innerHTML =
        `<h1>${{data.name}}</h1><p class="price">${{data.price}}</p>`;
}}
loadProduct();
</script>
</body>
</html>"""
        return html, "async function loadProduct"


@register(
    archetype_id="mvp.limit_image_text",
    category="limitations",
    difficulty="medium",
    solvable=False,
    description="Target text is embedded in an image",
    tags=["limitation", "image", "ocr"],
    phase=1,
    answer_schema=STRING_SCHEMA,
    allowed_limit_reasons=["image_text", "ocr_required", "text_in_image"],
    evidence_patterns=[
        r"<img[^>]*>",
        r"\.png",
        r"\.jpg",
        r"\.svg",
        r"background-image",
    ],
)
class ImageTextGenerator(Generator):
    """Generate tasks where target text is in an image.

    BeautifulSoup cannot extract text from images - OCR would be required.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        # Select random HTML style for realistic variation
        style = rng.choice(list(HtmlStyle))

        # The text that's "in" the image
        hidden_text = random_paragraph(rng, sentences=1)
        image_filename = f"content-{rng.randint(1000, 9999)}.png"

        body_content = f"""<article>
    <h1>Important Information</h1>
    <p>Please see the details below:</p>
    <div class="content-image">
        <img src="/images/{image_filename}"
             alt="Document content"
             title="Content image"
             data-content="{hidden_text}">
    </div>
    <p>For more information, contact support.</p>
</article>"""

        # Wrap with realistic chrome for real-world difficulty
        html = wrap_with_realistic_chrome(
            body_content,
            style,
            rng,
            title="Document",
            complexity="realistic",
            include_nav=True,
            include_footer=True,
        )
        html = add_noise_comments(html, rng, count=1)

        query = (
            "Extract the main content text from this page. "
            "The important information should be in the content section."
        )

        return TaskInstance(
            html=html,
            query=query,
            ground_truth=None,
            archetype_id=self.archetype_id,
            seed=seed,
            solvable=False,
            answer_schema=STRING_SCHEMA,
            limit_info={
                "allowed_reasons": ["image_text", "ocr_required", "text_in_image"],
                "evidence_patterns": [r"<img", r"\.png", r"content-image"],
            },
            metadata={
                "actual_text": hidden_text,
                "image_filename": image_filename,
            },
        )
