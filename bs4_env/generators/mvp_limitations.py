"""Limitation detection archetypes for BeautifulSoup RL environment.

These tasks are intentionally UNSOLVABLE with static BeautifulSoup parsing.
The correct behavior is to recognize the limitation and abstain with evidence.
"""

from bs4_env.config import STRING_SCHEMA
from bs4_env.generators.base import (
    Generator,
    HtmlStyle,
    TaskInstance,
    add_noise_comments,
    make_rng,
    random_paragraph,
    random_price,
    random_product_name,
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


@register(
    archetype_id="mvp.limit_canvas_text",
    category="limitations",
    difficulty="hard",
    solvable=False,
    description="Target text is rendered via HTML5 canvas drawing API",
    tags=["limitation", "canvas", "graphics"],
    phase=1,
    answer_schema=STRING_SCHEMA,
    allowed_limit_reasons=["canvas_text", "canvas_rendering", "graphics_only"],
    evidence_patterns=[
        r"<canvas",
        r"getContext\(['\"]2d['\"]\)",
        r"fillText",
        r"strokeText",
        r"ctx\.font",
    ],
)
class CanvasTextGenerator(Generator):
    """Generate tasks where target text is drawn on an HTML5 canvas.

    Canvas elements render graphics programmatically - the text exists only
    in JavaScript drawing commands, not as DOM text nodes.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        # The text that will be "drawn" on canvas
        product_name = random_product_name(rng)
        product_price = random_price(rng)

        # Choose a canvas pattern
        patterns = [
            self._simple_text_pattern,
            self._styled_text_pattern,
            self._rotated_text_pattern,
        ]
        pattern_fn = rng.choice(patterns)
        html, evidence_hint = pattern_fn(rng, product_name, product_price)

        query = (
            "Extract the product name and price displayed on this page. "
            "The information should be visible in the main content area."
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
                "allowed_reasons": ["canvas_text", "canvas_rendering", "graphics_only"],
                "evidence_patterns": [r"<canvas", r"fillText", r"getContext"],
            },
            metadata={
                "actual_product": product_name,
                "actual_price": product_price,
                "pattern": pattern_fn.__name__,
                "evidence_hint": evidence_hint,
            },
        )

    def _simple_text_pattern(self, rng, product_name: str, price: str) -> tuple[str, str]:
        """Generate simple canvas text drawing."""
        html = f"""<!DOCTYPE html>
<html>
<head><title>Product Display</title></head>
<body>
<h1>Featured Product</h1>
<canvas id="productCanvas" width="400" height="200"></canvas>
<p>See our featured product above!</p>
<script>
const canvas = document.getElementById('productCanvas');
const ctx = canvas.getContext('2d');
ctx.fillStyle = '#333';
ctx.font = '24px Arial';
ctx.fillText('{product_name}', 20, 50);
ctx.font = '32px Arial';
ctx.fillStyle = '#e63946';
ctx.fillText('{price}', 20, 100);
</script>
</body>
</html>"""
        return html, "ctx.fillText"

    def _styled_text_pattern(self, rng, product_name: str, price: str) -> tuple[str, str]:
        """Generate styled canvas text with gradients."""
        html = f"""<!DOCTYPE html>
<html>
<head><title>Product Showcase</title></head>
<body>
<div class="product-display">
    <canvas id="priceTag" width="300" height="150"></canvas>
</div>
<script>
var c = document.getElementById('priceTag');
var ctx = c.getContext('2d');
var gradient = ctx.createLinearGradient(0, 0, 300, 0);
gradient.addColorStop(0, '#1a1a2e');
gradient.addColorStop(1, '#16213e');
ctx.fillStyle = gradient;
ctx.fillRect(0, 0, 300, 150);
ctx.fillStyle = '#fff';
ctx.font = 'bold 20px Helvetica';
ctx.fillText('{product_name}', 15, 45);
ctx.font = '28px Helvetica';
ctx.fillStyle = '#ffd700';
ctx.fillText('{price}', 15, 90);
</script>
</body>
</html>"""
        return html, "getContext('2d')"

    def _rotated_text_pattern(self, rng, product_name: str, price: str) -> tuple[str, str]:
        """Generate rotated/transformed canvas text."""
        html = f"""<!DOCTYPE html>
<html>
<head><title>Special Offer</title></head>
<body>
<section class="offer">
    <canvas id="offerCanvas" width="400" height="250"></canvas>
    <p>Limited time offer - see details above</p>
</section>
<script>
(function() {{
    var canvas = document.getElementById('offerCanvas');
    var ctx = canvas.getContext('2d');
    ctx.save();
    ctx.translate(200, 125);
    ctx.rotate(-0.1);
    ctx.font = 'bold 22px sans-serif';
    ctx.fillStyle = '#2d3436';
    ctx.textAlign = 'center';
    ctx.fillText('{product_name}', 0, -20);
    ctx.font = '36px sans-serif';
    ctx.fillStyle = '#d63031';
    ctx.fillText('{price}', 0, 30);
    ctx.restore();
}})();
</script>
</body>
</html>"""
        return html, "ctx.fillText"


@register(
    archetype_id="mvp.limit_svg_path_data",
    category="limitations",
    difficulty="hard",
    solvable=False,
    description="Target data is encoded in SVG path coordinates or shape attributes",
    tags=["limitation", "svg", "graphics", "vector"],
    phase=1,
    answer_schema=STRING_SCHEMA,
    allowed_limit_reasons=["svg_path_data", "vector_graphics", "svg_only"],
    evidence_patterns=[
        r"<svg",
        r"<path[^>]+d=",
        r"<polygon",
        r"<polyline",
        r"viewBox",
    ],
)
class SvgPathDataGenerator(Generator):
    """Generate tasks where data is encoded in SVG path/shape attributes.

    The target information is represented visually through SVG paths,
    not as extractable text content.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        # Generate chart data that would be "visible" in the SVG
        data_points = [rng.randint(10, 100) for _ in range(5)]
        total_value = sum(data_points)
        max_value = max(data_points)

        # Choose a pattern
        patterns = [
            self._bar_chart_pattern,
            self._line_chart_pattern,
            self._pie_chart_pattern,
        ]
        pattern_fn = rng.choice(patterns)
        html, evidence_hint = pattern_fn(rng, data_points)

        query = (
            "Extract the data values shown in the chart on this page. "
            "The chart displays numerical information that should be readable."
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
                "allowed_reasons": ["svg_path_data", "vector_graphics", "svg_only"],
                "evidence_patterns": [r"<svg", r"<path", r"<rect", r"<polygon"],
            },
            metadata={
                "actual_data": data_points,
                "total": total_value,
                "max": max_value,
                "pattern": pattern_fn.__name__,
                "evidence_hint": evidence_hint,
            },
        )

    def _bar_chart_pattern(self, rng, data_points: list[int]) -> tuple[str, str]:
        """Generate SVG bar chart where values are in rect heights."""
        bars = []
        for i, value in enumerate(data_points):
            x = 50 + i * 60
            height = value * 2
            y = 200 - height
            bars.append(f'<rect x="{x}" y="{y}" width="40" height="{height}" fill="#4a90d9"/>')

        html = f"""<!DOCTYPE html>
<html>
<head><title>Sales Report</title></head>
<body>
<h2>Quarterly Sales Data</h2>
<svg width="400" height="250" viewBox="0 0 400 250">
    <line x1="40" y1="200" x2="360" y2="200" stroke="#333" stroke-width="2"/>
    <line x1="40" y1="0" x2="40" y2="200" stroke="#333" stroke-width="2"/>
    {"".join(bars)}
</svg>
<p>Data visualization - hover for details</p>
</body>
</html>"""
        return html, "<rect"

    def _line_chart_pattern(self, rng, data_points: list[int]) -> tuple[str, str]:
        """Generate SVG line chart where values are in path coordinates."""
        points = []
        for i, value in enumerate(data_points):
            x = 50 + i * 70
            y = 180 - value * 1.5
            points.append(f"{x},{y}")

        path_d = "M " + " L ".join(points)

        html = f"""<!DOCTYPE html>
<html>
<head><title>Performance Metrics</title></head>
<body>
<div class="chart-container">
    <h3>Performance Over Time</h3>
    <svg width="400" height="220" viewBox="0 0 400 220">
        <path d="{path_d}" fill="none" stroke="#e74c3c" stroke-width="3"/>
        <path d="{path_d} L {50 + (len(data_points) - 1) * 70},180 L 50,180 Z"
              fill="rgba(231,76,60,0.2)" stroke="none"/>
    </svg>
</div>
</body>
</html>"""
        return html, '<path d="'

    def _pie_chart_pattern(self, rng, data_points: list[int]) -> tuple[str, str]:
        """Generate SVG pie chart where values are in arc paths."""
        import math

        total = sum(data_points)
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
        paths = []
        start_angle = 0

        for i, value in enumerate(data_points):
            angle = (value / total) * 2 * math.pi
            end_angle = start_angle + angle

            x1 = 150 + 80 * math.cos(start_angle)
            y1 = 100 + 80 * math.sin(start_angle)
            x2 = 150 + 80 * math.cos(end_angle)
            y2 = 100 + 80 * math.sin(end_angle)

            large_arc = 1 if angle > math.pi else 0
            path = f'<path d="M 150 100 L {x1:.1f} {y1:.1f} A 80 80 0 {large_arc} 1 {x2:.1f} {y2:.1f} Z" fill="{colors[i]}"/>'
            paths.append(path)
            start_angle = end_angle

        html = f"""<!DOCTYPE html>
<html>
<head><title>Market Share</title></head>
<body>
<h2>Market Distribution</h2>
<svg width="300" height="220" viewBox="0 0 300 220">
    {"".join(paths)}
</svg>
<p>Interactive chart - click segments for details</p>
</body>
</html>"""
        return html, '<path d="M 150 100'


@register(
    archetype_id="mvp.limit_pdf_embed",
    category="limitations",
    difficulty="hard",
    solvable=False,
    description="Target content is in an embedded PDF document",
    tags=["limitation", "pdf", "embed", "object"],
    phase=1,
    answer_schema=STRING_SCHEMA,
    allowed_limit_reasons=["pdf_embed", "embedded_document", "pdf_required"],
    evidence_patterns=[
        r"<embed[^>]+application/pdf",
        r"<object[^>]+application/pdf",
        r"<iframe[^>]+\.pdf",
        r"\.pdf['\"]",
    ],
)
class PdfEmbedGenerator(Generator):
    """Generate tasks where target content is in an embedded PDF.

    The PDF is referenced but not included - BeautifulSoup cannot
    parse PDF content embedded in HTML pages.
    """

    def generate(self, seed: int) -> TaskInstance:
        rng = make_rng(self.archetype_id, seed)

        # The content that's "in" the PDF
        document_title = f"Report {rng.randint(2020, 2024)}-{rng.randint(1, 12):02d}"
        key_value = f"${rng.randint(100, 999)},{rng.randint(100, 999):03d}"

        # Choose an embed pattern
        patterns = [
            self._embed_tag_pattern,
            self._object_tag_pattern,
            self._iframe_pattern,
        ]
        pattern_fn = rng.choice(patterns)
        html, evidence_hint = pattern_fn(rng, document_title, key_value)

        query = (
            "Extract the key financial figure from the embedded document. "
            "The document contains important numerical data."
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
                "allowed_reasons": ["pdf_embed", "embedded_document", "pdf_required"],
                "evidence_patterns": [r"\.pdf", r"application/pdf", r"<embed", r"<object"],
            },
            metadata={
                "document_title": document_title,
                "actual_value": key_value,
                "pattern": pattern_fn.__name__,
                "evidence_hint": evidence_hint,
            },
        )

    def _embed_tag_pattern(self, rng, doc_title: str, value: str) -> tuple[str, str]:
        """Generate PDF embedded with <embed> tag."""
        pdf_filename = f"report-{rng.randint(1000, 9999)}.pdf"
        html = f"""<!DOCTYPE html>
<html>
<head><title>{doc_title}</title></head>
<body>
<header>
    <h1>Financial Documents</h1>
    <p>Please review the document below for Q4 results.</p>
</header>
<main>
    <div class="document-viewer">
        <embed src="/documents/{pdf_filename}"
               type="application/pdf"
               width="100%"
               height="600px"
               title="{doc_title}">
        <p>Your browser does not support PDF viewing.
           <a href="/documents/{pdf_filename}">Download PDF</a></p>
    </div>
</main>
<footer>
    <p>For questions, contact investor relations.</p>
</footer>
</body>
</html>"""
        return html, 'type="application/pdf"'

    def _object_tag_pattern(self, rng, doc_title: str, value: str) -> tuple[str, str]:
        """Generate PDF embedded with <object> tag."""
        pdf_filename = f"annual-{rng.randint(2020, 2024)}.pdf"
        html = f"""<!DOCTYPE html>
<html>
<head><title>Annual Report</title></head>
<body>
<div class="report-container">
    <h2>{doc_title}</h2>
    <object data="/reports/{pdf_filename}"
            type="application/pdf"
            width="800"
            height="700">
        <p>Unable to display PDF.
           <a href="/reports/{pdf_filename}">Click here to download</a>.</p>
    </object>
</div>
</body>
</html>"""
        return html, "application/pdf"

    def _iframe_pattern(self, rng, doc_title: str, value: str) -> tuple[str, str]:
        """Generate PDF embedded with <iframe>."""
        pdf_filename = f"statement-{rng.randint(1, 12):02d}-{rng.randint(2020, 2024)}.pdf"
        html = f"""<!DOCTYPE html>
<html>
<head><title>Account Statement</title></head>
<body>
<section class="statement-viewer">
    <h1>Account Statement</h1>
    <p>Your statement for the period is shown below:</p>
    <iframe src="/statements/{pdf_filename}"
            width="100%"
            height="800"
            title="{doc_title}"
            frameborder="0">
        <p>Your browser doesn't support iframes.
           <a href="/statements/{pdf_filename}">View PDF</a></p>
    </iframe>
</section>
</body>
</html>"""
        return html, f"/statements/{pdf_filename}"
