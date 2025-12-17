# BeautifulSoup Scraping Research Journal

## Purpose
Document real-world scraping challenges to improve the BS4 RL environment.

---

# Day 1: E-commerce (Part 1)

## Sites: Amazon, eBay, Etsy, Walmart, Target

---

## Amazon

### Site Overview
- **URL**: https://www.amazon.com
- **Tech Stack**: Custom Amazon framework, heavy JS
- **Anti-bot**: Medium (accepts requests with good headers)
- **Overall Difficulty**: 4/5

### Key Discovery: Hybrid Static/JS Rendering

Amazon uses a **hybrid approach**:
- **Static HTML**: Title, brand, rating, review count, feature bullets, images
- **JS-rendered**: Price, availability, delivery info

This is a critical pattern for our RL environment!

### Task 1: Extract Product Title ✅

**URL**: https://www.amazon.com/dp/B0BDHWDR12

**HTML Snippet**:
```html
<span id="productTitle" class="a-size-large product-title-word-break">
    Apple AirPods Pro (2nd Gen) Wireless Earbuds...
</span>
```

**Expected Output**: "Apple AirPods Pro (2nd Gen) Wireless Earbuds..."

**Solution**:
```python
title = soup.find("span", id="productTitle")
result = title.get_text(strip=True)
```

**Challenges**:
- [x] Deep nesting (32 levels!)
- [ ] Similar elements
- [ ] Missing data
- [ ] Malformed HTML
- [ ] JS-rendered
- [ ] Unicode issues
- [ ] Parser differences

**BS4 Gotchas Hit**:
- Must use `get_text(strip=True)` due to whitespace padding

**Difficulty**: 2/5 (easy once you know the ID)

**Current Archetype Coverage**: mvp.extract_text_by_id ✅

---

### Task 2: Extract Product Price ❌ (JS-REQUIRED)

**URL**: https://www.amazon.com/dp/B0BDHWDR12

**HTML Snippet**:
```html
<!-- Price div is EMPTY in static HTML! -->
<div id="corePrice_feature_div"></div>
```

**Expected Output**: "$249.00" (but not available)

**Solution**:
```python
# BS4 CANNOT extract this - price is JS-rendered
# This should be recognized as a LIMITATION
```

**Challenges**:
- [ ] Deep nesting
- [ ] Similar elements
- [ ] Missing data
- [ ] Malformed HTML
- [x] JS-rendered ⚠️ CRITICAL
- [ ] Unicode issues
- [ ] Parser differences

**BS4 Gotchas Hit**:
- Price div EXISTS but is EMPTY
- 267 empty placeholder divs throughout page
- Model should recognize this pattern and abstain

**Difficulty**: N/A (unsolvable with BS4)

**Current Archetype Coverage**: mvp.limit_js_required ✅

---

### Task 3: Extract Brand/Store Name ✅

**URL**: https://www.amazon.com/dp/B0BDHWDR12

**HTML Snippet**:
```html
<a id="bylineInfo" class="a-link-normal" href="/stores/Apple/...">
    Visit the Apple Store
</a>
```

**Expected Output**: "Visit the Apple Store" or parsed "Apple"

**Solution**:
```python
brand = soup.find("a", id="bylineInfo")
result = brand.get_text(strip=True)  # "Visit the Apple Store"
# Or parse: "Apple"
```

**Challenges**:
- [x] Deep nesting
- [ ] Similar elements
- [ ] Missing data
- [ ] Malformed HTML
- [ ] JS-rendered
- [ ] Unicode issues
- [ ] Parser differences

**Difficulty**: 2/5

**Current Archetype Coverage**: mvp.extract_text_by_id ✅

---

### Task 4: Extract Rating ✅

**URL**: https://www.amazon.com/dp/B0BDHWDR12

**Expected Output**: "4.7 out of 5 stars"

**Solution**:
```python
rating = soup.find("span", class_="a-icon-alt")
result = rating.get_text(strip=True)
```

**Challenges**:
- [x] Deep nesting
- [x] Similar elements (multiple ratings on page)
- [ ] Missing data
- [ ] Malformed HTML
- [ ] JS-rendered
- [ ] Unicode issues
- [ ] Parser differences

**Difficulty**: 3/5 (need to get the RIGHT rating element)

**Current Archetype Coverage**: mvp.semantic_ambiguity (partially)

---

### Task 5: Extract Feature Bullets ✅

**URL**: https://www.amazon.com/dp/B0BDHWDR12

**Expected Output**: List of 8 feature strings

**Solution**:
```python
bullets = soup.find("div", id="feature-bullets")
items = bullets.find_all("span", class_="a-list-item")
result = [item.get_text(strip=True) for item in items]
```

**Challenges**:
- [x] Deep nesting
- [ ] Similar elements
- [ ] Missing data
- [ ] Malformed HTML
- [ ] JS-rendered
- [ ] Unicode issues
- [ ] Parser differences

**Difficulty**: 2/5

**Current Archetype Coverage**: mvp.list_extraction ✅

---

### Amazon Summary

| Task | Extractable | Archetype Covered |
|------|-------------|-------------------|
| Product title | ✅ | mvp.extract_text_by_id |
| Price | ❌ JS | mvp.limit_js_required |
| Brand | ✅ | mvp.extract_text_by_id |
| Rating | ✅ | mvp.semantic_ambiguity |
| Feature bullets | ✅ | mvp.list_extraction |
| Availability | ❌ JS | mvp.limit_js_required |
| Delivery info | ❌ JS | mvp.limit_js_required |

**New Insight**: Major e-commerce sites render purchase-critical info via JS (anti-scraping?), but metadata is static.

---

## eBay

### Site Overview
- **URL**: https://www.ebay.com
- **Tech Stack**: Custom framework, Bootstrap CSS
- **Anti-bot**: Low (accepts requests easily)
- **Overall Difficulty**: 3/5

### Key Discovery: More Static Data Than Amazon!

eBay differs from Amazon:
- **Static HTML**: Title, **PRICE** (!!), seller name, shipping, condition, item specifics, images
- **JS-rendered**: Reviews, some dynamic offers
- **Uses data-testid extensively** (304 attributes on one page!)
- 35 levels deep nesting (even more than Amazon's 32)
- 280 empty placeholder divs

### Task 1: Extract Product Title ✅

**URL**: https://www.ebay.com/itm/389028214648

**HTML Snippet**:
```html
<h1 class="x-item-title__mainTitle">
    <span class="ux-textspans ux-textspans--BOLD">
        Heath's Deciphering Dice | Complex Math Magic Trick
    </span>
</h1>
```

**Expected Output**: "Heath's Deciphering Dice | Complex Math Magic Trick"

**Solution**:
```python
title = soup.find("h1", class_="x-item-title__mainTitle")
result = title.get_text(strip=True)
```

**Challenges**:
- [x] Deep nesting (35 levels!)
- [ ] Similar elements
- [ ] Missing data
- [ ] Malformed HTML
- [ ] JS-rendered
- [ ] Unicode issues
- [ ] Parser differences

**BS4 Gotchas Hit**:
- Class-based selection (no ID), must use `class_` or `attrs`
- Multiple nested spans inside the h1

**Difficulty**: 2/5

**Current Archetype Coverage**: mvp.extract_text_by_class ✅

---

### Task 2: Extract Product Price ✅ (UNLIKE AMAZON!)

**URL**: https://www.ebay.com/itm/389028214648

**HTML Snippet**:
```html
<div class="x-price-primary">
    <span class="ux-textspans">US $21.37</span>
</div>
```

**Expected Output**: "US $21.37"

**Solution**:
```python
price = soup.find("div", class_="x-price-primary")
result = price.get_text(strip=True)
```

**Challenges**:
- [x] Deep nesting
- [ ] Similar elements
- [ ] Missing data
- [ ] Malformed HTML
- [ ] JS-rendered
- [ ] Unicode issues
- [ ] Parser differences

**BS4 Gotchas Hit**:
- Price IS STATIC unlike Amazon! This is extractable.
- Currency is baked into the string (need parsing for numeric)

**Difficulty**: 2/5

**Current Archetype Coverage**: mvp.extract_text_by_class ✅

**NEW INSIGHT**: eBay doesn't JS-protect prices like Amazon does!

---

### Task 3: Extract Seller Name ✅

**URL**: https://www.ebay.com/itm/389028214648

**HTML Snippet**:
```html
<div class="x-sellercard-atf__info">
    <a href="https://www.ebay.com/str/brainfoodfun">
        Creative Crafthouse
    </a>
</div>
```

**Expected Output**: "Creative Crafthouse"

**Solution**:
```python
seller_section = soup.find("div", class_="x-sellercard-atf__info")
seller_link = seller_section.find("a")
result = seller_link.get_text(strip=True)
```

**Challenges**:
- [x] Deep nesting
- [x] Similar elements (many links on page)
- [ ] Missing data
- [ ] Malformed HTML
- [ ] JS-rendered
- [ ] Unicode issues
- [ ] Parser differences

**BS4 Gotchas Hit**:
- Must navigate through parent container first
- Many other links with similar patterns exist

**Difficulty**: 3/5

**Current Archetype Coverage**: mvp.nested_navigation (need to add)

---

### Task 4: Extract Shipping Info ✅ (with caveats)

**URL**: https://www.ebay.com/itm/389028214648

**HTML Snippet**:
```html
<div class="ux-labels-values--shipping">
    Shipping:US $9.14UPS Ground Saver...Located in: Hudson, Florida
</div>
```

**Expected Output**: {"cost": "US $9.14", "method": "UPS Ground Saver", "location": "Hudson, Florida"}

**Solution**:
```python
shipping = soup.find("div", class_="ux-labels-values--shipping")
raw_text = shipping.get_text(strip=True)
# Text is CONCATENATED - needs string parsing!
# "Shipping:US $9.14UPS Ground Saver.See detailsfor shippingLocated in: Hudson, Florida"
```

**Challenges**:
- [x] Deep nesting
- [ ] Similar elements
- [ ] Missing data
- [ ] Malformed HTML
- [ ] JS-rendered
- [ ] Unicode issues
- [ ] Parser differences

**BS4 Gotchas Hit**:
- `get_text()` CONCATENATES all nested spans without spaces!
- Need `separator` parameter: `get_text(separator=" ", strip=True)`
- Even then, may need regex to parse structure

**Difficulty**: 4/5 (extraction easy, parsing hard)

**Current Archetype Coverage**: NEEDS NEW ARCHETYPE - "text_concatenation_gotcha"

---

### Task 5: Extract Item Specifics (Key-Value Table) ✅

**URL**: https://www.ebay.com/itm/389028214648

**HTML Snippet**:
```html
<div class="ux-labels-values">
    <span class="ux-textspans">Brand</span>
    <span class="ux-textspans--BOLD">Creative Crafthouse</span>
</div>
```

**Expected Output**: {"Brand": "Creative Crafthouse", "Condition": "New", ...}

**Solution**:
```python
specifics = {}
for row in soup.find_all("div", class_="ux-labels-values"):
    label = row.find("span", class_="ux-textspans")
    value = row.find("span", class_="ux-textspans--BOLD")
    if label and value:
        specifics[label.get_text(strip=True)] = value.get_text(strip=True)
```

**Challenges**:
- [x] Deep nesting
- [x] Similar elements (many rows)
- [ ] Missing data
- [ ] Malformed HTML
- [ ] JS-rendered
- [ ] Unicode issues
- [ ] Parser differences

**BS4 Gotchas Hit**:
- Class names have double dashes (CSS BEM convention)
- Label and value are siblings, not parent-child

**Difficulty**: 3/5

**Current Archetype Coverage**: mvp.table_extraction ✅

---

### Task 6: Extract Product Images ✅

**URL**: https://www.ebay.com/itm/389028214648

**Solution**:
```python
images = soup.find_all("img", src=True)
product_images = [
    img.get("src")
    for img in images
    if "i.ebayimg.com" in img.get("src", "")
]
# Returns 13 product images
```

**Challenges**:
- [x] Deep nesting
- [x] Similar elements (15 total images, only 13 are product)
- [ ] Missing data
- [ ] Malformed HTML
- [ ] JS-rendered
- [ ] Unicode issues
- [ ] Parser differences

**BS4 Gotchas Hit**:
- Must filter by domain to exclude icons/logos
- Images use `src` not `data-src` (good!)

**Difficulty**: 2/5

**Current Archetype Coverage**: mvp.image_extraction ✅

---

### eBay Summary

| Task | Extractable | Archetype Covered |
|------|-------------|-------------------|
| Product title | ✅ | mvp.extract_text_by_class |
| Price | ✅ | mvp.extract_text_by_class |
| Seller | ✅ | mvp.nested_navigation (NEW) |
| Shipping | ✅ (messy) | NEW: text_concatenation_gotcha |
| Item specifics | ✅ | mvp.table_extraction |
| Images | ✅ | mvp.image_extraction |
| Reviews | ❓ | Need to investigate |

**Key Insight**: eBay is MORE BS4-friendly than Amazon! Price is static.

**New Archetype Ideas from eBay**:
1. `text_concatenation_gotcha` - when get_text() smashes words together
2. `domain_filtering` - filtering images by URL pattern
3. `data_testid_selection` - using data-testid attributes (304 on this page!)

---

## Etsy

### Site Overview
- **URL**: https://www.etsy.com
- **Tech Stack**: Unknown (blocked before analysis)
- **Anti-bot**: **HIGH - DataDome** 🚫
- **Overall Difficulty**: N/A (blocked)

### Key Discovery: DataDome Bot Protection

Etsy uses **DataDome**, a professional anti-bot service:
- Returns 403 Forbidden immediately
- Response headers reveal protection:
  - `Server: DataDome`
  - `X-DataDome: protected`
  - `X-DataDome-riskscore: 0.9912` (99% confident we're a bot!)
- Even with browser-like headers (Sec-Fetch-*, Accept-CH, etc.), still blocked

### Task 1: Fetch Any Page ❌ (BLOCKED)

**URL**: https://www.etsy.com/listing/1028317492/handmade-ceramic-cup

**Expected Output**: HTML content

**Actual Result**:
```
HTTP 403 Forbidden
Server: DataDome
X-DataDome-riskscore: 0.9912
Content-Length: 779  # Just an error page
```

**Solution**:
```python
# BS4 CANNOT scrape Etsy - they use DataDome bot protection
# Would need:
# 1. Browser automation (Selenium, Playwright)
# 2. Residential proxies
# 3. CAPTCHA solving service
# None of these are BS4 solutions
```

**Challenges**:
- [ ] Deep nesting
- [ ] Similar elements
- [ ] Missing data
- [ ] Malformed HTML
- [ ] JS-rendered
- [ ] Unicode issues
- [ ] Parser differences
- [x] **Anti-bot blocks** ⚠️ CRITICAL

**BS4 Gotchas Hit**:
- BS4 is useless if you can't get the HTML in the first place
- This is a LIMITATION recognition task

**Difficulty**: N/A (unsolvable with BS4)

**Current Archetype Coverage**: mvp.limit_antibot (NEED TO ADD)

---

### Etsy Summary

| Task | Extractable | Archetype Covered |
|------|-------------|-------------------|
| ANY | ❌ BLOCKED | NEW: mvp.limit_antibot |

**Key Insight**: Some sites are completely inaccessible to requests-based scraping. The model should recognize anti-bot error patterns.

**New Archetype Ideas from Etsy**:
1. `limit_antibot` - recognize 403/captcha/bot-detection responses
2. `http_error_recognition` - different error codes mean different things

---

## Walmart

### Site Overview
- **URL**: https://www.walmart.com
- **Tech Stack**: Next.js (SSR) - data is static!
- **Anti-bot**: Low (accepts requests)
- **Overall Difficulty**: 3/5

### Key Discovery: Next.js SSR + Rich Microdata

Walmart uses **Server-Side Rendering** with Next.js:
- `__NEXT_DATA__` script contains ALL product data as JSON!
- Schema.org microdata (`itemprop`) attributes throughout
- 136 `data-testid` attributes, 44 `data-automation-id` attributes
- Price IS STATIC in HTML (unlike Amazon)

### Task 1: Extract Product Title ✅

**URL**: https://www.walmart.com/ip/Apple-AirPods-Pro-2nd-Generation/1752657021

**HTML Snippet**:
```html
<h1 itemprop="name">Apple AirPods Pro (2nd Generation) - Lightning</h1>
```

**Expected Output**: "Apple AirPods Pro (2nd Generation) - Lightning"

**Solution**:
```python
# Method 1: Microdata
title = soup.find(attrs={"itemprop": "name"})
result = title.get_text(strip=True)

# Method 2: h1 tag
title = soup.find("h1")
result = title.get_text(strip=True)
```

**Challenges**:
- [x] Deep nesting (30 levels)
- [ ] Similar elements
- [ ] Missing data
- [ ] Malformed HTML
- [ ] JS-rendered
- [ ] Unicode issues
- [ ] Parser differences

**BS4 Gotchas Hit**:
- `itemprop` is an HTML attribute, use `attrs={"itemprop": "name"}`

**Difficulty**: 1/5 (very easy with microdata)

**Current Archetype Coverage**: mvp.microdata_extraction ✅

---

### Task 2: Extract Product Price ✅

**URL**: https://www.walmart.com/ip/Apple-AirPods-Pro-2nd-Generation/1752657021

**HTML Snippet**:
```html
<span itemprop="price">Now $117.23</span>
```

**Expected Output**: "Now $117.23" or parsed "$117.23"

**Solution**:
```python
price = soup.find(attrs={"itemprop": "price"})
result = price.get_text(strip=True)  # "Now $117.23"
```

**Challenges**:
- [x] Deep nesting
- [ ] Similar elements
- [ ] Missing data
- [ ] Malformed HTML
- [ ] JS-rendered
- [ ] Unicode issues
- [ ] Parser differences

**BS4 Gotchas Hit**:
- Price includes prefix text ("Now ") - need parsing for numeric
- UNLIKE AMAZON, price is in static HTML!

**Difficulty**: 2/5

**Current Archetype Coverage**: mvp.microdata_extraction ✅

**NEW INSIGHT**: Walmart makes scraping EASY with microdata + SSR!

---

### Task 3: Extract from JSON-LD Structured Data ✅

**URL**: https://www.walmart.com/ip/Apple-AirPods-Pro-2nd-Generation/1752657021

**HTML Snippet**:
```html
<script type="application/ld+json">
{
  "@type": "Product",
  "name": "Apple AirPods Pro (2nd Generation)...",
  "offers": {"price": "117.23", "priceCurrency": "USD"}
}
</script>
```

**Solution**:
```python
import json
script = soup.find("script", type="application/ld+json")
data = json.loads(script.string)
name = data["name"]
price = data["offers"]["price"]
```

**Challenges**:
- [ ] Deep nesting
- [ ] Similar elements
- [ ] Missing data
- [ ] Malformed HTML
- [ ] JS-rendered
- [ ] Unicode issues
- [ ] Parser differences

**BS4 Gotchas Hit**:
- JSON-LD is a `<script>` tag, access via `.string` not `.text`
- May need to find the RIGHT json-ld script (multiple on page)
- Data may not match visible page content exactly

**Difficulty**: 2/5

**Current Archetype Coverage**: mvp.json_ld_extraction ✅

---

### Task 4: Extract from __NEXT_DATA__ (React Hydration) ✅

**URL**: https://www.walmart.com/ip/Apple-AirPods-Pro-2nd-Generation/1752657021

**HTML Snippet**:
```html
<script id="__NEXT_DATA__" type="application/json">
{"props": {"pageProps": {"initialData": {...product data...}}}}
</script>
```

**Solution**:
```python
import json
next_data = soup.find("script", id="__NEXT_DATA__")
data = json.loads(next_data.string)
# Navigate deeply nested structure
product = data["props"]["pageProps"]["initialData"]["data"]["product"]
```

**Challenges**:
- [ ] Deep nesting (in JSON!)
- [ ] Similar elements
- [ ] Missing data
- [ ] Malformed HTML
- [ ] JS-rendered
- [ ] Unicode issues
- [ ] Parser differences

**BS4 Gotchas Hit**:
- This is a goldmine - ALL page data in one place!
- Structure varies by site (Next.js, Nuxt.js, etc.)
- Need to know where to look in the JSON tree

**Difficulty**: 3/5 (finding the data in deep JSON)

**Current Archetype Coverage**: NEED NEW: mvp.react_hydration_extraction

---

### Walmart Summary

| Task | Extractable | Archetype Covered |
|------|-------------|-------------------|
| Product title | ✅ | mvp.microdata_extraction |
| Price | ✅ | mvp.microdata_extraction |
| JSON-LD data | ✅ | mvp.json_ld_extraction |
| __NEXT_DATA__ | ✅ | NEW: mvp.react_hydration_extraction |

**Key Insight**: SSR sites (Next.js, Nuxt.js) are EASIER to scrape - all data is in HTML!

**New Archetype Ideas from Walmart**:
1. `microdata_extraction` - using itemprop attributes
2. `json_ld_extraction` - parsing application/ld+json scripts
3. `react_hydration_extraction` - parsing __NEXT_DATA__ and similar
4. `data_attribute_selection` - using data-testid, data-automation-id

---

# Challenge Tally (Running Count)

| Challenge Type | Count | Example Sites |
|----------------|-------|---------------|
| Deep nesting | 3 | Amazon (32), eBay (35), Walmart (30) |
| Similar elements | 2 | Amazon, eBay |
| Missing data | 0 | |
| Malformed HTML | 0 | |
| JS-rendered | 1 | Amazon (price) |
| Unicode issues | 0 | |
| Parser differences | 0 | |
| Anti-bot blocks | 1 | Etsy (DataDome) |
| Other | 0 | |

## Target

### Site Overview
- **URL**: https://www.target.com
- **Tech Stack**: Next.js, but price is CLIENT-SIDE rendered!
- **Anti-bot**: Low (accepts requests)
- **Overall Difficulty**: 4/5

### Key Discovery: Next.js BUT Price is Client-Side

Target uses Next.js like Walmart, BUT:
- **Title**: Available in static HTML via `data-test="product-title"`
- **Price**: NOT in HTML! Client-side rendered like Amazon!
- **No JSON-LD**: Unlike Walmart's rich structured data
- **__NEXT_DATA__**: Contains flag `isProductDetailServerSideRenderPriceEnabled` (revealing!)
- Uses `data-test` (not `data-testid`)

Target intentionally disables server-side price rendering.

### Task 1: Extract Product Title ✅

**URL**: https://www.target.com/p/apple-airpods-4-wireless-earbuds/-/A-93208030

**HTML Snippet**:
```html
<h1 data-test="product-title">Apple AirPods 4 Wireless Earbuds</h1>
```

**Expected Output**: "Apple AirPods 4 Wireless Earbuds"

**Solution**:
```python
title = soup.find(attrs={"data-test": "product-title"})
result = title.get_text(strip=True)
```

**Challenges**:
- [x] Deep nesting (21 levels - least of all sites!)
- [ ] Similar elements
- [ ] Missing data
- [ ] Malformed HTML
- [ ] JS-rendered
- [ ] Unicode issues
- [ ] Parser differences

**BS4 Gotchas Hit**:
- Target uses `data-test` not `data-testid`
- Must use `attrs={"data-test": "..."}` syntax

**Difficulty**: 2/5

**Current Archetype Coverage**: mvp.data_attribute_selection ✅

---

### Task 2: Extract Product Price ❌ (JS-REQUIRED)

**URL**: https://www.target.com/p/apple-airpods-4-wireless-earbuds/-/A-93208030

**HTML Snippet**:
```html
<!-- Price is NOT in static HTML! -->
<!-- No JSON-LD, no microdata, not in __NEXT_DATA__ -->
```

**Expected Output**: Price (not available)

**Solution**:
```python
# BS4 CANNOT extract price from Target
# Despite being Next.js, they disable server-side price rendering
# pageProps.isProductDetailServerSideRenderPriceEnabled flag reveals this
```

**Challenges**:
- [ ] Deep nesting
- [ ] Similar elements
- [ ] Missing data
- [ ] Malformed HTML
- [x] JS-rendered ⚠️
- [ ] Unicode issues
- [ ] Parser differences

**BS4 Gotchas Hit**:
- Just because a site uses SSR framework doesn't mean ALL data is SSR
- Need to recognize when data is intentionally client-side

**Difficulty**: N/A (unsolvable with BS4)

**Current Archetype Coverage**: mvp.limit_js_required ✅

---

### Target Summary

| Task | Extractable | Archetype Covered |
|------|-------------|-------------------|
| Product title | ✅ | mvp.data_attribute_selection |
| Price | ❌ JS | mvp.limit_js_required |
| Product images | ❓ | Need investigation |
| Description | ❓ | Need investigation |

**Key Insight**: Even Next.js sites may intentionally disable SSR for certain data (pricing)!

**New Archetype Ideas from Target**:
1. `data_attribute_selection` - using data-test attributes
2. Recognize that SSR framework != all data is static

---

# Day 1 Summary: E-commerce

## Sites Analyzed

| Site | Anti-bot | Price Static? | Best Selector Pattern | Key Insight |
|------|----------|---------------|----------------------|-------------|
| Amazon | Medium | ❌ JS | ID (`id="productTitle"`) | Hybrid static/JS, price protected |
| eBay | Low | ✅ Yes! | Class (`x-price-primary`) | Most BS4-friendly major e-commerce |
| Etsy | HIGH | N/A | BLOCKED | DataDome kills simple requests |
| Walmart | Low | ✅ Yes! | Microdata + JSON-LD | SSR paradise, __NEXT_DATA__ goldmine |
| Target | Low | ❌ JS | data-test attributes | SSR but price client-side |

## Patterns Discovered

1. **Hybrid Static/JS Rendering**: Major sites use static HTML for metadata (title, brand) but JS for pricing
2. **Anti-bot Variance**: From none (Walmart) to severe (Etsy/DataDome)
3. **SSR != All Static**: Target uses Next.js but disables price SSR
4. **Microdata Rich Sites**: Walmart has excellent Schema.org markup
5. **Testing IDs**: eBay (data-testid), Target (data-test), Walmart (data-automation-id)
6. **Deep Nesting Universal**: 21-35 levels across all sites

## New Archetypes Needed

1. `microdata_extraction` - using itemprop attributes
2. `json_ld_extraction` - parsing structured data scripts
3. `react_hydration_extraction` - __NEXT_DATA__ parsing
4. `data_attribute_selection` - data-testid/data-test selection
5. `text_concatenation_gotcha` - get_text() separator issues
6. `limit_antibot` - recognize bot protection responses

---

# New Archetype Ideas

| Idea | Found On | Description |
|------|----------|-------------|
| microdata_extraction | Walmart | Extract data using itemprop attributes (Schema.org) |
| json_ld_extraction | Walmart | Parse application/ld+json structured data scripts |
| react_hydration_extraction | Walmart, Target | Parse __NEXT_DATA__ or __NUXT_DATA__ scripts |
| data_attribute_selection | eBay, Target, Walmart | Select by data-testid, data-test, data-automation-id |
| text_concatenation_gotcha | eBay | When get_text() smashes words together without separator |
| limit_antibot | Etsy | Recognize 403/CAPTCHA/bot detection responses |
| domain_filtering | eBay | Filter images/links by domain pattern |
| nested_navigation | eBay | Navigate parent → child to find specific element |

---

# Key Learnings

1. **Pricing is protected**: Amazon and Target JS-render prices even with SSR frameworks
2. **eBay is surprisingly easy**: Most BS4-friendly major e-commerce site (static prices!)
3. **Anti-bot varies wildly**: Etsy (DataDome) vs Walmart (wide open)
4. **SSR doesn't guarantee static data**: Target uses Next.js but client-renders prices
5. **Microdata is a goldmine**: When present (Walmart), `itemprop` makes extraction trivial
6. **Deep nesting is universal**: 21-35 levels across all major sites
7. **Testing IDs are selector gold**: data-testid, data-test, data-automation-id
8. **get_text() needs separator**: Without it, spans concatenate without spaces

---

# Day 2: E-commerce (Part 2)

## Sites: Best Buy, Newegg, Wayfair, Zappos

---

## Best Buy

### Site Overview
- **URL**: https://www.bestbuy.com
- **Tech Stack**: Unknown (blocked)
- **Anti-bot**: **EXTREME - Akamai Bot Manager** 🚫
- **Overall Difficulty**: N/A (blocked)

### Key Discovery: Akamai Bot Manager

Best Buy uses **Akamai Bot Manager** - even more aggressive than Etsy's DataDome:
- DNS resolves to Akamai CDN (23.x.x.x)
- Connections **hang indefinitely** - no response at all
- No 403 error, just timeout (stealthier anti-bot)
- curl, wget, requests all fail the same way

### Task 1: Fetch Any Page ❌ (TIMEOUT)

**URL**: https://www.bestbuy.com/site/apple-airpods-4-white/6574420.p

**Solution**:
```python
# BS4 CANNOT scrape Best Buy
# Akamai Bot Manager blocks at TCP/TLS level
# Connection hangs indefinitely - no response
# Would need browser automation + residential proxies
```

**Current Archetype Coverage**: mvp.limit_antibot ✅

---

## Newegg

### Site Overview
- **URL**: https://www.newegg.com
- **Tech Stack**: Custom, Bootstrap CSS
- **Anti-bot**: Low (accepts requests)
- **Overall Difficulty**: 3/5

### Key Discovery: JSON-LD Rich, HTML Price Missing

Newegg has **5 JSON-LD scripts** with rich structured data:
- Product data with price in JSON-LD
- BreadcrumbList, FAQPage, VideoObject schemas
- But visible price element (`price-current`) NOT in static HTML!

### Task 1: Extract Product Title ✅

**URL**: https://www.newegg.com/p/N82E16811139251

**HTML Snippet**:
```html
<h1 class="product-title">Corsair AIR 5400 RS-R ARGB Triple Chamber Mid-Tower PC Case</h1>
```

**Solution**:
```python
title = soup.find("h1", class_="product-title")
result = title.get_text(strip=True)
```

**Difficulty**: 2/5

---

### Task 2: Extract Product Price (JSON-LD only) ✅

**HTML Snippet**:
```html
<script type="application/ld+json">
{"@type": "Product", "offers": {"price": "229.99", "priceCurrency": "USD"}}
</script>
```

**Solution**:
```python
import json
scripts = soup.find_all("script", type="application/ld+json")
for script in scripts:
    data = json.loads(script.string)
    if data.get("@type") == "Product":
        price = data["offers"]["price"]  # "229.99"
```

**BS4 Gotchas**:
- Price NOT in visible HTML (price-current class missing)
- Must parse JSON-LD to get price
- Hybrid pattern: title in HTML, price in structured data

**Current Archetype Coverage**: mvp.json_ld_extraction ✅

---

### Newegg Summary

| Task | Extractable | Method |
|------|-------------|--------|
| Title | ✅ | Class selector |
| Price | ✅ (JSON-LD only) | Structured data |
| Visible price | ❌ | JS-rendered |

---

## Wayfair

### Site Overview
- **URL**: https://www.wayfair.com
- **Tech Stack**: Heavy JS SPA
- **Anti-bot**: Low (accepts requests)
- **Overall Difficulty**: 5/5

### Key Discovery: Pure JS SPA - No Static Content

Wayfair returns an **empty shell**:
- No `<h1>` tag in static HTML
- No JSON-LD structured data
- No microdata
- Depth: 0 levels (body is nearly empty!)
- Everything is client-side rendered

### Task 1: Extract Anything ❌ (JS-REQUIRED)

**Solution**:
```python
# BS4 CANNOT extract ANYTHING from Wayfair
# Page is pure SPA - entire content is JS-rendered
# Even product title is not in static HTML
```

**Current Archetype Coverage**: mvp.limit_js_required ✅

**Key Insight**: Some sites return HTML that's effectively empty - pure SPA architecture.

---

## Zappos

### Site Overview
- **URL**: https://www.zappos.com
- **Tech Stack**: Custom framework, rich microdata
- **Anti-bot**: Low (accepts requests)
- **Overall Difficulty**: 2/5

### Key Discovery: Microdata Paradise

Zappos is **extremely BS4-friendly**:
- 33 `itemprop` attributes throughout the page!
- Price, brand, name all in microdata
- 1.3MB page but well-structured
- Price includes MSRP and discount in text

### Task 1: Extract Product Title ✅

**HTML Snippet**:
```html
<span itemprop="name">Nike Pegasus 41</span>
```

**Solution**:
```python
title = soup.find(attrs={"itemprop": "name"})
result = title.get_text(strip=True)
```

**Difficulty**: 1/5

---

### Task 2: Extract Product Price ✅

**HTML Snippet**:
```html
<span itemprop="price">$134.97MSRP:$145(7%OFF)</span>
```

**Solution**:
```python
price = soup.find(attrs={"itemprop": "price"})
result = price.get_text(strip=True)  # "$134.97MSRP:$145(7%OFF)"
# Need regex to extract just "$134.97"
```

**BS4 Gotchas**:
- Price includes MSRP and discount text concatenated
- Need string parsing for clean numeric value

**Difficulty**: 2/5 (extraction easy, parsing moderate)

---

### Zappos Summary

| Task | Extractable | Method |
|------|-------------|--------|
| Title | ✅ | Microdata |
| Price | ✅ | Microdata |
| Brand | ✅ | Microdata |

**Key Insight**: Zappos is one of the most BS4-friendly e-commerce sites!

---

# Day 2 Summary

| Site | Anti-bot | Price Static? | Method | Key Finding |
|------|----------|---------------|--------|-------------|
| Best Buy | EXTREME | N/A | BLOCKED | Akamai hangs connections |
| Newegg | Low | JSON-LD only | Structured data | Title HTML, price JSON-LD |
| Wayfair | Low | ❌ | NONE | Pure SPA, empty HTML |
| Zappos | Low | ✅ | Microdata | 33 itemprop attrs! |

## Running Totals

- **Sites analyzed**: 9
- **Extractable**: 5 (Amazon, eBay, Walmart, Newegg, Zappos)
- **Blocked**: 2 (Etsy, Best Buy)
- **Pure JS SPA**: 2 (Wayfair, Target-price)

## New Patterns

1. **Akamai vs DataDome**: Akamai hangs silently, DataDome returns 403
2. **JSON-LD as price source**: Newegg has price only in structured data
3. **Pure SPA detection**: Wayfair returns depth=0, no h1
4. **Microdata variance**: Zappos (33 attrs) vs Target (0 attrs)

---

# Day 3: News/Media

## Sites: CNN, BBC, NPR, The Guardian, Reuters

---

## CNN

### Site Overview
- **URL**: https://www.cnn.com
- **Tech Stack**: Pure SPA (React?)
- **Anti-bot**: Low (accepts requests)
- **Overall Difficulty**: 5/5

### Key Discovery: Pure JavaScript SPA

CNN articles are **completely JS-rendered**:
- No `<h1>` in static HTML
- No `<article>` element
- No author, date, or body content
- No JSON-LD structured data
- 609KB of essentially empty shell

### Task 1: Extract Article Title ❌ (JS-REQUIRED)

**Solution**:
```python
# BS4 CANNOT extract ANYTHING from CNN articles
# Entire page content is JavaScript-rendered
```

**Current Archetype Coverage**: mvp.limit_js_required ✅

---

## BBC News

### Site Overview
- **URL**: https://www.bbc.com/news
- **Tech Stack**: Custom framework, rich semantic HTML
- **Anti-bot**: Low (accepts requests)
- **Overall Difficulty**: 2/5

### Key Discovery: Testing ID Paradise

BBC is **extremely BS4-friendly**:
- **818 data-testid attributes!** (most we've seen)
- 52 h2 headlines in static HTML
- 76 news article links
- Well-structured semantic HTML

### Task 1: Extract Headlines ✅

**Solution**:
```python
headlines = soup.find_all("h2")
# Returns 52 headline elements!
```

### Task 2: Extract Article Links ✅

**Solution**:
```python
links = [a for a in soup.find_all("a", href=True)
         if "/news/" in a.get("href", "")]
# Returns 76 article links
```

**Difficulty**: 1/5

**Key Insight**: BBC's 818 data-testid attributes make it perfect for stable selectors!

---

## NPR

### Site Overview
- **URL**: https://www.npr.org
- **Tech Stack**: Bootstrap, semantic HTML
- **Anti-bot**: Low (accepts requests)
- **Overall Difficulty**: 2/5

### Key Discovery: Clean Semantic HTML

NPR has **textbook-perfect HTML structure**:
- 40 `<article>` elements with proper structure
- 40 `<time>` elements with `datetime` attributes
- 24 h2 headlines
- Clean separation of concerns

### Task 1: Extract Article Cards ✅

**HTML Pattern**:
```html
<article>
  <h2>Headline</h2>
  <time datetime="2025-12-17">...</time>
  <p>Teaser text...</p>
</article>
```

**Solution**:
```python
articles = soup.find_all("article")
for article in articles:
    headline = article.find("h2").get_text(strip=True)
    date = article.find("time").get("datetime")
```

**Difficulty**: 1/5

**Key Insight**: NPR is the most semantically correct news site we've tested!

---

## The Guardian

### Site Overview
- **URL**: https://www.theguardian.com
- **Tech Stack**: Custom, data-link-name heavy
- **Anti-bot**: Low (accepts requests)
- **Overall Difficulty**: 2/5

### Key Discovery: data-link-name for Analytics

Guardian uses custom data attributes for analytics:
- 476 `data-link-name` attributes
- 44 `data-component` attributes
- 150 h3 headline elements
- Large page (1.2MB) but well-structured

### Task 1: Extract Headlines ✅

**Solution**:
```python
headlines = soup.find_all("h3")
# Returns 150 headlines
```

**Difficulty**: 2/5

---

## Reuters

### Site Overview
- **URL**: https://www.reuters.com
- **Anti-bot**: **MEDIUM** - Returns 401 Forbidden
- **Overall Difficulty**: N/A (blocked)

### Key Discovery: API Authentication Required

Reuters returns **401 HTTP Forbidden**:
- Not a standard 403 bot block
- Likely requires API authentication
- Different from Akamai/DataDome patterns

---

# Day 3 Summary

| Site | Anti-bot | Content Static? | Best Feature | Key Finding |
|------|----------|----------------|--------------|-------------|
| CNN | Low | ❌ Pure SPA | None | Completely JS-rendered |
| BBC | Low | ✅ | 818 data-testid | Best test selectors |
| NPR | Low | ✅ | Semantic HTML | Perfect article/time structure |
| Guardian | Low | ✅ | 476 data-link-name | Analytics attrs useful |
| Reuters | Medium | N/A | BLOCKED | 401 auth required |

## News Site Patterns

1. **CNN = Wayfair of news**: Pure SPA, zero static content
2. **BBC = Testing heaven**: 818 data-testid for stable selectors
3. **NPR = Semantic perfection**: article/time/h2 structure
4. **Reuters blocks differently**: 401 vs 403 vs timeout

## Running Totals

- **Sites analyzed**: 13
- **Extractable**: 8 (Amazon, eBay, Walmart, Newegg, Zappos, BBC, NPR, Guardian)
- **Blocked**: 3 (Etsy, Best Buy, Reuters)
- **Pure JS SPA**: 3 (Wayfair, Target-price, CNN)

---

# Day 4: Jobs

## Sites: Indeed, Monster, LinkedIn, Glassdoor, ZipRecruiter

---

## Indeed

### Site Overview
- **URL**: https://www.indeed.com
- **Tech Stack**: Unknown - returns compressed/binary content
- **Anti-bot**: Medium (accepts but returns garbage)
- **Overall Difficulty**: 5/5

### Key Discovery: Binary/Compressed Response

Indeed returns **unreadable binary data**:
- 150KB response but text is garbled
- Looks like undecompressed gzip or Brotli encoding issue
- No HTML structure visible
- May require specific Accept-Encoding headers

**Current Archetype Coverage**: NEEDS NEW - encoding_issues

---

## Monster

### Site Overview
- **URL**: https://www.monster.com
- **Tech Stack**: Custom framework
- **Anti-bot**: Low (accepts requests)
- **Overall Difficulty**: 4/5

### Key Discovery: Shell Only - No Job Data

Monster returns **navigation shell without job content**:
- 6 h2 elements (but just navigation headers)
- 43 data-testid attributes
- 0 article elements
- Job listings are JS-rendered

**Current Archetype Coverage**: mvp.limit_js_required ✅

---

## LinkedIn Jobs

### Site Overview
- **URL**: https://www.linkedin.com/jobs
- **Tech Stack**: Bootstrap + custom
- **Anti-bot**: Low (public job search accessible!)
- **Overall Difficulty**: 2/5

### Key Discovery: Public Jobs Are Scrapeable!

LinkedIn's **public job search** (no login required) has:
- 7 job cards in static HTML
- Job titles in h3 elements
- Company names visible
- `data-tracking-id` for analytics

### Task 1: Extract Job Titles ✅

**HTML Pattern**:
```html
<div class="base-card job-search-card">
  <h3 class="base-search-card__title">Software Engineer, Fullstack</h3>
</div>
```

**Solution**:
```python
titles = soup.find_all("h3", class_=lambda c: c and "title" in str(c))
# Returns 9 job titles
```

**Difficulty**: 2/5

**Key Insight**: LinkedIn's PUBLIC job search is BS4-friendly (unlike logged-in feed)!

---

## Glassdoor

### Site Overview
- **URL**: https://www.glassdoor.com
- **Anti-bot**: **HIGH** - Returns 403 Forbidden
- **Overall Difficulty**: N/A (blocked)

### Key Discovery: Cloudflare Protection

Glassdoor blocks with 403:
- Different from Indeed's encoding issue
- Clean block, no timeout

---

## ZipRecruiter

### Site Overview
- **URL**: https://www.ziprecruiter.com
- **Anti-bot**: **HIGH** - Returns 403 Forbidden
- **Overall Difficulty**: N/A (blocked)

---

# Day 4 Summary

| Site | Anti-bot | Content Static? | Method | Key Finding |
|------|----------|----------------|--------|-------------|
| Indeed | Medium | ❌ Encoding | - | Returns binary garbage |
| Monster | Low | ❌ JS Shell | - | Nav only, jobs JS-rendered |
| LinkedIn | Low | ✅ Public jobs | Class selectors | 7 jobs in static HTML! |
| Glassdoor | HIGH | N/A | BLOCKED | 403 Forbidden |
| ZipRecruiter | HIGH | N/A | BLOCKED | 403 Forbidden |

## Jobs Site Patterns

1. **Most job sites heavily protected**: 2/5 blocked outright
2. **LinkedIn public search works!**: Best surprise finding
3. **Indeed encoding weirdness**: May need special handling
4. **Monster = shell only**: Like CNN, all content is JS

## Running Totals

- **Sites analyzed**: 18
- **Extractable**: 9 (+ LinkedIn)
- **Blocked**: 5 (+ Glassdoor, ZipRecruiter)
- **Pure JS SPA**: 4 (+ Monster)
- **Encoding issues**: 1 (Indeed)
