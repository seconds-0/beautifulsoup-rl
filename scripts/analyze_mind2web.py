#!/usr/bin/env python3
"""Analyze Mind2Web dataset to extract real HTML patterns.

This script loads the Mind2Web dataset from HuggingFace and analyzes
the HTML to understand real-world patterns for our generators.
"""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from bs4 import BeautifulSoup
from datasets import load_dataset


def detect_framework(html: str, soup: BeautifulSoup) -> str:
    """Detect the likely frontend framework from HTML patterns."""
    # React patterns
    react_patterns = [
        "data-reactroot",
        "data-reactid",
        "__NEXT_DATA__",
        "react-root",
        "_next/static",
    ]
    for pattern in react_patterns:
        if pattern in html:
            return "react"

    # Angular patterns
    angular_patterns = [
        "_ngcontent-",
        "ng-version",
        "ng-reflect-",
        "ng-star-inserted",
    ]
    for pattern in angular_patterns:
        if pattern in html:
            return "angular"

    # Vue patterns
    vue_patterns = [
        "data-v-",
        "__vue__",
        "v-cloak",
    ]
    for pattern in vue_patterns:
        if pattern in html:
            return "vue"

    # Bootstrap patterns (check class names)
    bootstrap_classes = ["container", "row", "col-", "btn-", "navbar", "card-"]
    all_classes = " ".join([" ".join(tag.get("class", [])) for tag in soup.find_all(class_=True)])
    bootstrap_matches = sum(1 for c in bootstrap_classes if c in all_classes)
    if bootstrap_matches >= 3:
        return "bootstrap"

    # Tailwind patterns (utility class explosion)
    tailwind_indicators = [
        "flex",
        "items-center",
        "justify-",
        "bg-",
        "text-",
        "px-",
        "py-",
        "rounded",
    ]
    tailwind_matches = sum(1 for c in tailwind_indicators if c in all_classes)
    if tailwind_matches >= 4:
        return "tailwind"

    return "traditional"


def extract_class_patterns(soup: BeautifulSoup, framework: str) -> list[str]:
    """Extract interesting class name patterns from the HTML."""
    classes = []
    for tag in soup.find_all(class_=True):
        tag_classes = tag.get("class", [])
        if isinstance(tag_classes, list):
            classes.extend(tag_classes)
    return classes


def analyze_structure(soup: BeautifulSoup) -> dict:
    """Analyze HTML structure metrics."""
    # Count elements by tag
    tag_counts = Counter(tag.name for tag in soup.find_all())

    # Calculate nesting depth
    def get_depth(element, current_depth=0):
        max_depth = current_depth
        for child in element.children:
            if hasattr(child, "children"):
                child_depth = get_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
        return max_depth

    max_depth = get_depth(soup) if soup.body else 0

    # Count elements with IDs
    elements_with_id = len(soup.find_all(id=True))

    # Count elements with classes
    elements_with_class = len(soup.find_all(class_=True))

    return {
        "tag_counts": dict(tag_counts.most_common(20)),
        "max_depth": max_depth,
        "elements_with_id": elements_with_id,
        "elements_with_class": elements_with_class,
        "total_elements": len(list(soup.find_all())),
    }


def analyze_head(soup: BeautifulSoup) -> dict:
    """Analyze head section patterns."""
    head = soup.head
    if not head:
        return {}

    # Count meta tags
    meta_tags = head.find_all("meta")
    meta_names = [m.get("name") or m.get("property") or "unknown" for m in meta_tags]

    # Count scripts
    scripts = head.find_all("script")
    script_srcs = [s.get("src", "inline") for s in scripts]

    # Count stylesheets
    stylesheets = head.find_all("link", rel="stylesheet")
    stylesheet_hrefs = [s.get("href", "unknown") for s in stylesheets]

    return {
        "meta_count": len(meta_tags),
        "meta_names": meta_names[:10],  # Top 10
        "script_count": len(scripts),
        "script_srcs": script_srcs[:5],
        "stylesheet_count": len(stylesheets),
        "stylesheet_hrefs": stylesheet_hrefs[:5],
    }


def main():
    print("Loading Mind2Web dataset from HuggingFace...")
    print("(This may take a while on first download)")

    # Load dataset - Mind2Web has train/test splits
    try:
        dataset = load_dataset("osunlp/Mind2Web", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying alternative loading method...")
        # Try loading just metadata first
        dataset = load_dataset("osunlp/Mind2Web", trust_remote_code=True)
        print(f"Available splits: {dataset.keys()}")
        dataset = dataset["train"]

    print(f"Loaded {len(dataset)} examples")

    # Check available columns
    print(f"Columns: {dataset.column_names}")

    # Sample for analysis (limit to 100 for speed)
    sample_size = min(100, len(dataset))
    print(f"\nAnalyzing {sample_size} examples...")

    # Collect statistics
    html_sizes = []
    frameworks = Counter()
    all_class_patterns = defaultdict(list)
    structure_stats = []
    head_stats = []

    for i, example in enumerate(dataset.select(range(sample_size))):
        if i % 10 == 0:
            print(f"  Processing {i}/{sample_size}...")

        # Mind2Web has HTML nested in actions list
        # Each action has 'raw_html' and 'cleaned_html' fields
        html = None
        actions = example.get("actions", [])
        if actions and len(actions) > 0:
            # Get HTML from first action (full page state)
            first_action = actions[0]
            if isinstance(first_action, dict):
                html = first_action.get("raw_html") or first_action.get("cleaned_html")

        if not html:
            print(f"  Warning: No HTML found in example {i}")
            continue

        html_sizes.append(len(html))

        # Parse with BeautifulSoup
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception as e:
            print(f"  Error parsing HTML {i}: {e}")
            continue

        # Detect framework
        framework = detect_framework(html, soup)
        frameworks[framework] += 1

        # Extract class patterns
        classes = extract_class_patterns(soup, framework)
        all_class_patterns[framework].extend(classes[:50])  # Limit per page

        # Analyze structure
        structure = analyze_structure(soup)
        structure_stats.append(structure)

        # Analyze head
        head = analyze_head(soup)
        head_stats.append(head)

    # Compute aggregate statistics
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)

    print("\n### HTML Size Distribution ###")
    if html_sizes:
        print(f"  Min:    {min(html_sizes):,} chars")
        print(f"  Max:    {max(html_sizes):,} chars")
        print(f"  Mean:   {statistics.mean(html_sizes):,.0f} chars")
        print(f"  Median: {statistics.median(html_sizes):,.0f} chars")
        if len(html_sizes) > 1:
            print(f"  StdDev: {statistics.stdev(html_sizes):,.0f} chars")

    print("\n### Framework Distribution ###")
    for framework, count in frameworks.most_common():
        pct = count / sum(frameworks.values()) * 100
        print(f"  {framework}: {count} ({pct:.1f}%)")

    print("\n### Class Pattern Samples by Framework ###")
    for framework, classes in all_class_patterns.items():
        class_counts = Counter(classes)
        print(f"\n  {framework.upper()} (top 15 classes):")
        for cls, count in class_counts.most_common(15):
            print(f"    {cls}: {count}")

    print("\n### Structure Statistics ###")
    if structure_stats:
        avg_depth = statistics.mean(s["max_depth"] for s in structure_stats)
        avg_elements = statistics.mean(s["total_elements"] for s in structure_stats)
        avg_with_class = statistics.mean(s["elements_with_class"] for s in structure_stats)
        avg_with_id = statistics.mean(s["elements_with_id"] for s in structure_stats)

        print(f"  Avg max nesting depth: {avg_depth:.1f}")
        print(f"  Avg total elements:    {avg_elements:.0f}")
        print(f"  Avg elements with class: {avg_with_class:.0f}")
        print(f"  Avg elements with ID:    {avg_with_id:.0f}")

        # Aggregate tag counts
        all_tags = Counter()
        for s in structure_stats:
            all_tags.update(s["tag_counts"])
        print("\n  Most common tags:")
        for tag, count in all_tags.most_common(15):
            print(f"    <{tag}>: {count}")

    print("\n### Head Section Statistics ###")
    if head_stats:
        meta_counts = [h.get("meta_count", 0) for h in head_stats]
        script_counts = [h.get("script_count", 0) for h in head_stats]
        stylesheet_counts = [h.get("stylesheet_count", 0) for h in head_stats]

        print(f"  Avg meta tags:    {statistics.mean(meta_counts):.1f}")
        print(f"  Avg scripts:      {statistics.mean(script_counts):.1f}")
        print(f"  Avg stylesheets:  {statistics.mean(stylesheet_counts):.1f}")

        # Common meta names
        all_meta = Counter()
        for h in head_stats:
            all_meta.update(h.get("meta_names", []))
        print("\n  Common meta tags:")
        for name, count in all_meta.most_common(10):
            print(f"    {name}: {count}")

    # Save detailed results to JSON
    output_path = Path(__file__).parent.parent / "outputs" / "mind2web_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "sample_size": sample_size,
        "html_sizes": {
            "min": min(html_sizes) if html_sizes else 0,
            "max": max(html_sizes) if html_sizes else 0,
            "mean": statistics.mean(html_sizes) if html_sizes else 0,
            "median": statistics.median(html_sizes) if html_sizes else 0,
        },
        "frameworks": dict(frameworks),
        "class_samples": {
            k: list(Counter(v).most_common(30)) for k, v in all_class_patterns.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
