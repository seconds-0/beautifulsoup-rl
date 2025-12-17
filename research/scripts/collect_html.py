"""Helper script for collecting HTML from websites.

Usage:
    python collect_html.py <url> <output_file>
    python collect_html.py "https://amazon.com/dp/B0..." "corpus/e-commerce/amazon/product_001.html"
"""

import sys
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import json
from datetime import datetime


# Common headers to look more like a browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def fetch_html(url: str, timeout: int = 30) -> tuple[str, dict]:
    """Fetch HTML from a URL and return content + metadata.

    Returns:
        Tuple of (html_content, metadata_dict)
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()

        metadata = {
            "url": url,
            "status_code": response.status_code,
            "content_length": len(response.text),
            "encoding": response.encoding,
            "fetched_at": datetime.now().isoformat(),
            "headers": dict(response.headers),
        }

        return response.text, metadata

    except requests.exceptions.RequestException as e:
        return None, {
            "url": url,
            "error": str(e),
            "error_type": type(e).__name__,
            "fetched_at": datetime.now().isoformat(),
        }


def analyze_html(html: str) -> dict:
    """Quick analysis of HTML structure."""
    soup = BeautifulSoup(html, "html.parser")

    # Count nesting depth
    def max_depth(element, current=0):
        children = [c for c in element.children if hasattr(c, 'children')]
        if not children:
            return current
        return max(max_depth(c, current + 1) for c in children)

    body = soup.find("body")
    depth = max_depth(body) if body else 0

    # Detect frameworks
    frameworks = []
    if soup.find(attrs={"class": lambda c: c and any(x in str(c) for x in ["react", "jsx"])}):
        frameworks.append("React")
    if soup.find(attrs={"ng-app": True}) or soup.find(attrs={"ng-controller": True}):
        frameworks.append("Angular")
    if soup.find(attrs={"v-if": True}) or soup.find(attrs={"v-for": True}):
        frameworks.append("Vue")
    if soup.find(attrs={"class": lambda c: c and any(x in str(c) for x in ["bootstrap", "btn-", "col-md"])}):
        frameworks.append("Bootstrap")
    if soup.find(attrs={"class": lambda c: c and any(x in str(c) for x in ["tailwind", "tw-"])}):
        frameworks.append("Tailwind")

    # Check for JS-rendered content indicators
    js_indicators = []
    if soup.find("div", id="root") and not soup.find("div", id="root").get_text(strip=True):
        js_indicators.append("empty_root_div")
    if soup.find(string=lambda s: s and "Loading" in s):
        js_indicators.append("loading_placeholder")
    if soup.find("noscript"):
        js_indicators.append("noscript_fallback")

    # Script analysis
    scripts = soup.find_all("script")
    script_count = len(scripts)
    inline_script_chars = sum(len(s.string or "") for s in scripts if s.string)

    return {
        "size_bytes": len(html),
        "size_kb": round(len(html) / 1024, 2),
        "max_nesting_depth": depth,
        "frameworks_detected": frameworks,
        "js_indicators": js_indicators,
        "script_count": script_count,
        "inline_script_chars": inline_script_chars,
        "has_json_ld": bool(soup.find("script", type="application/ld+json")),
        "tag_counts": {
            "div": len(soup.find_all("div")),
            "span": len(soup.find_all("span")),
            "a": len(soup.find_all("a")),
            "img": len(soup.find_all("img")),
            "table": len(soup.find_all("table")),
            "form": len(soup.find_all("form")),
        }
    }


def save_html(html: str, output_path: str, metadata: dict, analysis: dict) -> None:
    """Save HTML and metadata to files."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save HTML
    output_path.write_text(html, encoding="utf-8")

    # Save metadata alongside
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "fetch": metadata,
        "analysis": analysis,
    }, indent=2), encoding="utf-8")

    print(f"Saved: {output_path}")
    print(f"  Size: {analysis['size_kb']} KB")
    print(f"  Depth: {analysis['max_nesting_depth']} levels")
    print(f"  Frameworks: {analysis['frameworks_detected'] or 'none detected'}")
    print(f"  JS indicators: {analysis['js_indicators'] or 'none'}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    url = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Fetching: {url}")
    html, metadata = fetch_html(url)

    if html is None:
        print(f"Error: {metadata.get('error')}")
        sys.exit(1)

    analysis = analyze_html(html)
    save_html(html, output_path, metadata, analysis)


if __name__ == "__main__":
    main()
