"""Real HTML patterns extracted from Mind2Web dataset.

This module contains patterns observed in real websites from the Mind2Web
dataset (osunlp/Mind2Web). Use these to generate realistic HTML for training.

Source: Analysis of 100 samples from Mind2Web train split.
Date: December 2025
"""

# =============================================================================
# HTML Size Statistics (from Mind2Web analysis)
# =============================================================================

HTML_SIZE_STATS = {
    "min": 51_438,  # ~50KB
    "max": 2_665_947,  # ~2.6MB
    "mean": 387_428,  # ~387KB
    "median": 266_423,  # ~266KB
    "p25": 100_000,  # Estimated 25th percentile
    "p75": 500_000,  # Estimated 75th percentile
}

# Our target: Generate HTML in the 50K-500K range for realism
TARGET_HTML_SIZE_MIN = 50_000
TARGET_HTML_SIZE_MAX = 500_000

# =============================================================================
# Framework Distribution (observed frequencies)
# =============================================================================

FRAMEWORK_DISTRIBUTION = {
    "bootstrap": 0.57,  # 57% of real sites
    "traditional": 0.31,  # 31%
    "angular": 0.08,  # 8%
    "react": 0.04,  # 4%
    "tailwind": 0.00,  # Not detected in sample (but common in newer sites)
    "vue": 0.00,  # Not detected in sample
}

# =============================================================================
# Structure Statistics
# =============================================================================

STRUCTURE_STATS = {
    "avg_depth": 25,  # Average nesting depth (!)
    "avg_elements": 2197,  # Average total elements
    "avg_with_class": 1271,  # Average elements with class attribute
    "avg_with_id": 79,  # Average elements with id attribute
}

# Most common HTML tags (from Mind2Web analysis)
COMMON_TAGS = [
    ("div", 55016),
    ("a", 35349),
    ("li", 23394),
    ("span", 16574),
    ("button", 4646),
    ("img", 3662),
    ("ul", 3540),
    ("option", 2918),
    ("p", 2406),
    ("svg", 1241),
    ("input", 1188),
    ("h3", 1087),
    ("source", 994),
]

# =============================================================================
# Real Class Patterns by Framework
# =============================================================================

# Bootstrap patterns (most common framework at 57%)
BOOTSTRAP_CLASSES = [
    # Navigation
    "nav-item",
    "nav-link",
    "navbar",
    "navbar-nav",
    "navbar-brand",
    "navbar-toggler",
    "navbar-collapse",
    "navbar-expand-lg",
    # Grid
    "container",
    "container-fluid",
    "row",
    "col-xs-4",
    "col-sm-6",
    "col-md-4",
    "col-lg-3",
    "col-xl-2",
    # Dropdowns
    "dropdown",
    "dropdown-toggle",
    "dropdown-menu",
    "dropdown-item",
    "dropdown-link",
    # Utilities
    "d-none",
    "d-block",
    "d-flex",
    "d-inline",
    "text-center",
    "text-left",
    "text-right",
    "hide",
    "show",
    "visually-hidden",
    "sr-only",
    "clearfix",
    "float-left",
    "float-right",
    # Components
    "btn",
    "btn-primary",
    "btn-secondary",
    "btn-link",
    "card",
    "card-body",
    "card-header",
    "card-footer",
    "list-group",
    "list-group-item",
    "badge",
    "alert",
    "modal",
    "form-control",
]

# Traditional/semantic patterns
TRADITIONAL_CLASSES = [
    # Layout
    "container",
    "wrapper",
    "content",
    "main-content",
    "header",
    "footer",
    "sidebar",
    "nav",
    "navigation",
    # Custom design systems (like Tesla's tds-*)
    "product-name",
    "product-card",
    "product-image",
    "site-nav",
    "site-nav-item",
    "site-header",
    "banner",
    "hero",
    "section",
    # State
    "active",
    "selected",
    "disabled",
    "hidden",
    "visible",
    # Typography
    "title",
    "subtitle",
    "heading",
    "text",
    "link",
    # Components
    "button",
    "input",
    "form",
    "menu",
    "list",
    "item",
]

# React patterns (styled-components, CSS modules)
REACT_CLASSES = [
    # Styled-components pattern: ComponentName-sc-hash-index
    "Box-sc-kv6pi1-0",
    "Button-sc-abc123-0",
    "Container-sc-xyz789-0",
    # CSS modules pattern: ComponentName_className__hash
    "styles_container__abc12",
    "Header_wrapper__xyz89",
    "Button_primary__def45",
    # Common React patterns
    "theme-light",
    "theme-dark",
    "animation",
    "transition",
    "withFooter",
    "withHeader",
    "skipLink",
    "sr-only",
]

# Angular patterns
ANGULAR_CLASSES = [
    # Core Angular classes
    "ng-star-inserted",
    "ng-tns-c0-0",
    "ng-tns-c0-1",
    "ng-tns-c0-2",
    "ng-pristine",
    "ng-valid",
    "ng-invalid",
    "ng-touched",
    # Material Angular
    "mat-button",
    "mat-icon",
    "mat-card",
    "mat-list",
    "mat-form-field",
    "mat-input",
    "mat-select",
    # Bootstrap-like patterns in Angular
    "dropdown",
    "dropdown-toggle",
    "dropdown-menu",
    "navbar",
    "collapse",
    "w-100",
    "relative",
    "authenticated",
]

# Tailwind utility classes (for future use)
TAILWIND_CLASSES = [
    # Layout
    "flex",
    "block",
    "inline",
    "hidden",
    "grid",
    "items-center",
    "justify-center",
    "justify-between",
    # Spacing
    "p-4",
    "px-4",
    "py-2",
    "m-2",
    "mx-auto",
    "mt-4",
    "mb-2",
    # Sizing
    "w-full",
    "w-1/2",
    "h-auto",
    "max-w-lg",
    # Colors
    "bg-white",
    "bg-gray-100",
    "bg-blue-500",
    "text-white",
    "text-gray-800",
    "text-blue-600",
    # Borders
    "border",
    "border-gray-300",
    "rounded",
    "rounded-lg",
    # Effects
    "shadow",
    "shadow-md",
    "hover:bg-gray-200",
    # Typography
    "font-bold",
    "text-sm",
    "text-lg",
    "tracking-tight",
]

# =============================================================================
# Navigation Patterns (real examples)
# =============================================================================

BOOTSTRAP_NAV_TEMPLATE = """
<nav class="navbar navbar-expand-lg navbar-light bg-light">
  <div class="container-fluid">
    <a class="navbar-brand" href="#">{brand}</a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarNav">
      <ul class="navbar-nav me-auto mb-2 mb-lg-0">
        {nav_items}
      </ul>
    </div>
  </div>
</nav>
"""

BOOTSTRAP_NAV_ITEM_TEMPLATE = """
<li class="nav-item">
  <a class="nav-link{active}" href="{href}">{text}</a>
</li>
"""

BOOTSTRAP_DROPDOWN_TEMPLATE = """
<li class="nav-item dropdown">
  <a class="nav-link dropdown-toggle" href="#" id="navbarDropdown"
     role="button" data-bs-toggle="dropdown" aria-expanded="false">
    {text}
  </a>
  <ul class="dropdown-menu" aria-labelledby="navbarDropdown">
    {dropdown_items}
  </ul>
</li>
"""

ANGULAR_NAV_TEMPLATE = """
<nav class="navbar ng-star-inserted">
  <div class="container ng-tns-c0-0">
    <a class="navbar-brand" routerLink="/">{brand}</a>
    <ul class="navbar-nav">
      {nav_items}
    </ul>
  </div>
</nav>
"""

# =============================================================================
# Footer Patterns
# =============================================================================

BOOTSTRAP_FOOTER_TEMPLATE = """
<footer class="footer mt-auto py-3 bg-light">
  <div class="container">
    <div class="row">
      <div class="col-md-4">
        <h5>{company}</h5>
        <p class="text-muted">{description}</p>
      </div>
      <div class="col-md-4">
        <h5>Links</h5>
        <ul class="list-unstyled">
          {footer_links}
        </ul>
      </div>
      <div class="col-md-4">
        <h5>Contact</h5>
        <p class="text-muted">{contact}</p>
      </div>
    </div>
    <hr>
    <div class="row">
      <div class="col-12 text-center">
        <span class="text-muted">&copy; {year} {company}. All rights reserved.</span>
      </div>
    </div>
  </div>
</footer>
"""

# =============================================================================
# Noise Elements (common in real sites)
# =============================================================================

# Cookie banners, GDPR notices
COOKIE_BANNER_TEMPLATE = """
<div class="cookie-banner visually-hidden" id="cookie-consent" role="dialog" aria-label="Cookie consent">
  <div class="container">
    <p>We use cookies to improve your experience. By continuing to use this site, you agree to our use of cookies.</p>
    <div class="cookie-actions">
      <button class="btn btn-primary" id="accept-cookies">Accept</button>
      <button class="btn btn-secondary" id="decline-cookies">Decline</button>
    </div>
  </div>
</div>
"""

# Skip links for accessibility
SKIP_LINK_TEMPLATE = """
<a class="skip-link visually-hidden-focusable" href="#main-content">Skip to main content</a>
"""

# Social media links (common in footers)
SOCIAL_LINKS = [
    ("facebook", "https://facebook.com/{handle}"),
    ("twitter", "https://twitter.com/{handle}"),
    ("instagram", "https://instagram.com/{handle}"),
    ("linkedin", "https://linkedin.com/company/{handle}"),
    ("youtube", "https://youtube.com/{handle}"),
]

# =============================================================================
# Helper Functions
# =============================================================================


def get_random_classes(rng, framework: str, count: int = 5) -> list[str]:
    """Get random realistic class names for a framework."""
    class_pools = {
        "bootstrap": BOOTSTRAP_CLASSES,
        "traditional": TRADITIONAL_CLASSES,
        "react": REACT_CLASSES,
        "angular": ANGULAR_CLASSES,
        "tailwind": TAILWIND_CLASSES,
    }
    pool = class_pools.get(framework, TRADITIONAL_CLASSES)
    return [rng.choice(pool) for _ in range(count)]


def generate_styled_component_class(rng) -> str:
    """Generate a React styled-components class name."""
    components = ["Box", "Button", "Container", "Wrapper", "Card", "Text", "Link", "Icon"]
    component = rng.choice(components)
    hash_chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    hash_str = "".join(rng.choices(hash_chars, k=7))
    index = rng.randint(0, 5)
    return f"{component}-sc-{hash_str}-{index}"


def generate_css_module_class(rng) -> str:
    """Generate a CSS modules class name."""
    components = ["styles", "Header", "Footer", "Button", "Card", "Layout", "Modal"]
    names = ["container", "wrapper", "inner", "content", "main", "primary", "active"]
    component = rng.choice(components)
    name = rng.choice(names)
    hash_chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    hash_str = "".join(rng.choices(hash_chars, k=5))
    return f"{component}_{name}__{hash_str}"


def generate_angular_scope_class(rng) -> str:
    """Generate an Angular _ngcontent scope class."""
    hash_chars = "abcdefghijklmnopqrstuvwxyz"
    hash_str = "".join(rng.choices(hash_chars, k=3))
    component_id = rng.randint(0, 99)
    return f"_ngcontent-{hash_str}-c{component_id}"
