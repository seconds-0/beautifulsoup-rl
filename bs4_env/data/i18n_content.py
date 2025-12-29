"""International content for realistic multilingual HTML generation.

This module contains vocabularies and phrases in multiple languages to train
models on real-world parsing diversity and UTF-8 edge cases.

Languages included:
- Chinese (Simplified) - Han script
- Arabic - RTL script
- Japanese - Hiragana/Katakana/Kanji
- Korean - Hangul
- Russian - Cyrillic
- German - Latin with diacritics
- Spanish - Latin with diacritics
- French - Latin with diacritics
- Hindi - Devanagari
- Hebrew - RTL script

Source: Common web content patterns
"""

import random
from typing import Any

# =============================================================================
# Language Data
# =============================================================================

LANGUAGES: dict[str, dict[str, Any]] = {
    "zh": {  # Chinese (Simplified)
        "name": "Chinese",
        "script": "Han",
        "direction": "ltr",
        "phrases": [
            "欢迎来到我们的网站",  # Welcome to our website
            "产品详情",  # Product details
            "联系我们",  # Contact us
            "购物车",  # Shopping cart
            "用户登录",  # User login
            "立即购买",  # Buy now
            "加入购物车",  # Add to cart
            "查看更多",  # View more
            "热门推荐",  # Hot recommendations
            "新品上市",  # New arrivals
            "限时优惠",  # Limited time offer
            "免费配送",  # Free shipping
            "客户评价",  # Customer reviews
            "关于我们",  # About us
            "帮助中心",  # Help center
        ],
        "words": [
            "商品",
            "价格",
            "数量",
            "总计",
            "订单",
            "客户",
            "服务",
            "质量",
            "尺寸",
            "颜色",
            "品牌",
            "库存",
            "折扣",
            "优惠",
            "运费",
            "地址",
        ],
        "numbers": ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"],
    },
    "ar": {  # Arabic
        "name": "Arabic",
        "script": "Arabic",
        "direction": "rtl",
        "phrases": [
            "مرحبا بكم في موقعنا",  # Welcome to our website
            "تفاصيل المنتج",  # Product details
            "اتصل بنا",  # Contact us
            "سلة التسوق",  # Shopping cart
            "تسجيل الدخول",  # Login
            "اشتر الآن",  # Buy now
            "أضف إلى السلة",  # Add to cart
            "عرض المزيد",  # View more
            "الأكثر مبيعاً",  # Best sellers
            "عروض خاصة",  # Special offers
        ],
        "words": [
            "منتج",
            "سعر",
            "كمية",
            "إجمالي",
            "طلب",
            "عميل",
            "خدمة",
            "جودة",
            "حجم",
            "لون",
            "علامة",
            "مخزون",
            "خصم",
            "شحن",
            "عنوان",
        ],
        "numbers": ["١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩", "١٠"],
    },
    "ja": {  # Japanese
        "name": "Japanese",
        "script": "Hiragana/Katakana/Kanji",
        "direction": "ltr",
        "phrases": [
            "ようこそ",  # Welcome
            "商品詳細",  # Product details
            "お問い合わせ",  # Contact
            "カートに入れる",  # Add to cart
            "今すぐ購入",  # Buy now
            "もっと見る",  # View more
            "人気商品",  # Popular products
            "新着商品",  # New arrivals
            "送料無料",  # Free shipping
            "カスタマーレビュー",  # Customer reviews
            "会社概要",  # About us
            "ヘルプ",  # Help
        ],
        "words": [
            "商品",
            "価格",
            "数量",
            "合計",
            "注文",
            "お客様",
            "サービス",
            "品質",
            "サイズ",
            "カラー",
            "ブランド",
            "在庫",
            "割引",
            "配送",
            "住所",
        ],
        "numbers": ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"],
    },
    "ko": {  # Korean
        "name": "Korean",
        "script": "Hangul",
        "direction": "ltr",
        "phrases": [
            "환영합니다",  # Welcome
            "제품 상세",  # Product details
            "문의하기",  # Contact
            "장바구니",  # Shopping cart
            "로그인",  # Login
            "바로 구매",  # Buy now
            "장바구니 담기",  # Add to cart
            "더 보기",  # View more
            "인기 상품",  # Popular products
            "신상품",  # New arrivals
            "무료 배송",  # Free shipping
        ],
        "words": [
            "상품",
            "가격",
            "수량",
            "합계",
            "주문",
            "고객",
            "서비스",
            "품질",
            "사이즈",
            "색상",
            "브랜드",
            "재고",
            "할인",
            "배송",
            "주소",
        ],
        "numbers": ["일", "이", "삼", "사", "오", "육", "칠", "팔", "구", "십"],
    },
    "ru": {  # Russian
        "name": "Russian",
        "script": "Cyrillic",
        "direction": "ltr",
        "phrases": [
            "Добро пожаловать",  # Welcome
            "Описание товара",  # Product description
            "Связаться с нами",  # Contact us
            "Корзина",  # Cart
            "Войти",  # Login
            "Купить сейчас",  # Buy now
            "В корзину",  # Add to cart
            "Подробнее",  # More details
            "Популярные товары",  # Popular products
            "Новинки",  # New arrivals
            "Бесплатная доставка",  # Free shipping
            "Отзывы покупателей",  # Customer reviews
        ],
        "words": [
            "товар",
            "цена",
            "количество",
            "итого",
            "заказ",
            "клиент",
            "услуга",
            "качество",
            "размер",
            "цвет",
            "бренд",
            "наличие",
            "скидка",
            "доставка",
        ],
        "numbers": ["один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь"],
    },
    "de": {  # German
        "name": "German",
        "script": "Latin",
        "direction": "ltr",
        "phrases": [
            "Willkommen auf unserer Website",  # Welcome to our website
            "Produktdetails",  # Product details
            "Kontaktieren Sie uns",  # Contact us
            "Warenkorb",  # Shopping cart
            "Anmelden",  # Login
            "Jetzt kaufen",  # Buy now
            "In den Warenkorb",  # Add to cart
            "Mehr anzeigen",  # View more
            "Beliebte Produkte",  # Popular products
            "Neuheiten",  # New arrivals
            "Kostenloser Versand",  # Free shipping
            "Kundenbewertungen",  # Customer reviews
            "Über uns",  # About us
        ],
        "words": [
            "Produkt",
            "Preis",
            "Menge",
            "Gesamt",
            "Bestellung",
            "Kunde",
            "Service",
            "Qualität",
            "Größe",
            "Farbe",
            "Marke",
            "Verfügbarkeit",
            "Rabatt",
            "Versand",
        ],
        "numbers": ["eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht"],
    },
    "es": {  # Spanish
        "name": "Spanish",
        "script": "Latin",
        "direction": "ltr",
        "phrases": [
            "Bienvenido a nuestra web",  # Welcome to our website
            "Detalles del producto",  # Product details
            "Contáctenos",  # Contact us
            "Carrito de compras",  # Shopping cart
            "Iniciar sesión",  # Login
            "Comprar ahora",  # Buy now
            "Añadir al carrito",  # Add to cart
            "Ver más",  # View more
            "Productos populares",  # Popular products
            "Novedades",  # New arrivals
            "Envío gratis",  # Free shipping
            "Opiniones de clientes",  # Customer reviews
        ],
        "words": [
            "producto",
            "precio",
            "cantidad",
            "total",
            "pedido",
            "cliente",
            "servicio",
            "calidad",
            "tamaño",
            "color",
            "marca",
            "disponibilidad",
            "descuento",
            "envío",
        ],
        "numbers": ["uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho"],
    },
    "fr": {  # French
        "name": "French",
        "script": "Latin",
        "direction": "ltr",
        "phrases": [
            "Bienvenue sur notre site",  # Welcome to our website
            "Détails du produit",  # Product details
            "Contactez-nous",  # Contact us
            "Panier",  # Cart
            "Se connecter",  # Login
            "Acheter maintenant",  # Buy now
            "Ajouter au panier",  # Add to cart
            "Voir plus",  # View more
            "Produits populaires",  # Popular products
            "Nouveautés",  # New arrivals
            "Livraison gratuite",  # Free shipping
            "Avis clients",  # Customer reviews
        ],
        "words": [
            "produit",
            "prix",
            "quantité",
            "total",
            "commande",
            "client",
            "service",
            "qualité",
            "taille",
            "couleur",
            "marque",
            "disponibilité",
            "remise",
            "livraison",
        ],
        "numbers": ["un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit"],
    },
    "hi": {  # Hindi
        "name": "Hindi",
        "script": "Devanagari",
        "direction": "ltr",
        "phrases": [
            "हमारी वेबसाइट पर आपका स्वागत है",  # Welcome to our website
            "उत्पाद विवरण",  # Product details
            "संपर्क करें",  # Contact us
            "कार्ट",  # Cart
            "लॉगिन",  # Login
            "अभी खरीदें",  # Buy now
            "कार्ट में जोड़ें",  # Add to cart
            "और देखें",  # View more
            "लोकप्रिय उत्पाद",  # Popular products
            "मुफ्त शिपिंग",  # Free shipping
        ],
        "words": [
            "उत्पाद",
            "कीमत",
            "मात्रा",
            "कुल",
            "आदेश",
            "ग्राहक",
            "सेवा",
            "गुणवत्ता",
            "आकार",
            "रंग",
            "ब्रांड",
            "उपलब्धता",
            "छूट",
            "शिपिंग",
        ],
        "numbers": ["एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ", "दस"],
    },
    "he": {  # Hebrew
        "name": "Hebrew",
        "script": "Hebrew",
        "direction": "rtl",
        "phrases": [
            "ברוכים הבאים לאתר שלנו",  # Welcome to our website
            "פרטי המוצר",  # Product details
            "צור קשר",  # Contact us
            "עגלת קניות",  # Shopping cart
            "התחברות",  # Login
            "קנה עכשיו",  # Buy now
            "הוסף לעגלה",  # Add to cart
            "הצג עוד",  # View more
            "מוצרים פופולריים",  # Popular products
            "משלוח חינם",  # Free shipping
        ],
        "words": [
            "מוצר",
            "מחיר",
            "כמות",
            "סכום",
            "הזמנה",
            "לקוח",
            "שירות",
            "איכות",
            "גודל",
            "צבע",
            "מותג",
            "זמינות",
            "הנחה",
            "משלוח",
        ],
        "numbers": ["אחד", "שניים", "שלוש", "ארבע", "חמש", "שש", "שבע", "שמונה"],
    },
}

# RTL languages for quick lookup
RTL_LANGUAGES = {"ar", "he"}

# =============================================================================
# Emoji Categories
# =============================================================================

EMOJI_CATEGORIES = {
    "positive": ["👍", "❤️", "😊", "🎉", "✨", "🌟", "💯", "🔥", "👏", "💪", "🙌", "😍"],
    "commerce": ["🛒", "💰", "🏷️", "📦", "🚚", "💳", "🎁", "⭐", "💵", "🛍️", "📊", "💎"],
    "navigation": ["🏠", "📧", "📞", "🔍", "📋", "⚙️", "👤", "🔔", "📍", "🔗", "📱", "💻"],
    "status": ["✅", "❌", "⚠️", "ℹ️", "🔄", "⏳", "🔒", "🔓", "📌", "🏆", "🎯", "📈"],
    "social": ["👥", "💬", "🗨️", "📣", "🤝", "👋", "🎤", "📸", "🎬", "🎵", "🎮", "📺"],
}

# =============================================================================
# Special Unicode Characters
# =============================================================================

SPECIAL_UNICODE = {
    # Zero-width characters (can cause invisible parsing issues)
    "zero_width": [
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\ufeff",  # BOM / Zero-width no-break space
    ],
    # Common diacritics in European languages
    "diacritics": [
        "é",
        "è",
        "ê",
        "ë",  # French/Portuguese
        "ñ",
        "á",
        "í",
        "ó",
        "ú",  # Spanish
        "ü",
        "ö",
        "ä",
        "ß",  # German
        "ø",
        "å",
        "æ",  # Nordic
        "ç",
        "ã",
        "õ",  # Portuguese
        "ž",
        "č",
        "š",
        "đ",  # Slavic
        "ł",
        "ń",
        "ź",
        "ż",  # Polish
    ],
    # Currency symbols
    "currency": [
        "€",  # Euro
        "£",  # British Pound
        "¥",  # Yen/Yuan
        "₹",  # Indian Rupee
        "₽",  # Russian Ruble
        "₿",  # Bitcoin
        "฿",  # Thai Baht
        "₩",  # Korean Won
        "₴",  # Ukrainian Hryvnia
        "₺",  # Turkish Lira
    ],
    # Mathematical symbols
    "math": ["±", "×", "÷", "≠", "≤", "≥", "∞", "√", "∑", "∏", "∈", "∉", "⊂", "⊃"],
    # Quotation marks (vary by language)
    "quotes": [
        "« »",  # French/Russian guillemets
        '„ "',  # German quotes
        "「 」",  # Japanese brackets
        "『 』",  # Japanese double brackets
        '" "',  # English curly quotes
        "' '",  # English single quotes
    ],
    # Punctuation variants
    "punctuation": [
        "…",  # Ellipsis
        "—",  # Em dash
        "–",  # En dash
        "•",  # Bullet
        "·",  # Middle dot
        "‣",  # Triangular bullet
        "※",  # Reference mark (Japanese)
    ],
}


# =============================================================================
# Helper Functions
# =============================================================================


def get_random_language(rng: random.Random, exclude_rtl: bool = False) -> str:
    """Get a random language code.

    Args:
        rng: Random instance.
        exclude_rtl: If True, exclude RTL languages.

    Returns:
        Language code (e.g., "zh", "ar", "ja").
    """
    languages = list(LANGUAGES.keys())
    if exclude_rtl:
        languages = [lang for lang in languages if lang not in RTL_LANGUAGES]
    return rng.choice(languages)


def get_random_phrase(rng: random.Random, language: str | None = None) -> tuple[str, str]:
    """Get a random phrase in a language.

    Args:
        rng: Random instance.
        language: Language code. If None, randomly selected.

    Returns:
        Tuple of (phrase, language_code).
    """
    if language is None:
        language = get_random_language(rng)

    lang_data = LANGUAGES.get(language)
    if not lang_data:
        language = "en"
        return "Welcome to our website", language

    phrase = rng.choice(lang_data["phrases"])
    return phrase, language


def get_random_word(rng: random.Random, language: str | None = None) -> tuple[str, str]:
    """Get a random word in a language.

    Args:
        rng: Random instance.
        language: Language code. If None, randomly selected.

    Returns:
        Tuple of (word, language_code).
    """
    if language is None:
        language = get_random_language(rng)

    lang_data = LANGUAGES.get(language)
    if not lang_data:
        return "product", "en"

    word = rng.choice(lang_data["words"])
    return word, language


def get_random_emoji(rng: random.Random, category: str | None = None) -> str:
    """Get a random emoji.

    Args:
        rng: Random instance.
        category: Emoji category. If None, randomly selected.

    Returns:
        Single emoji string.
    """
    if category is None:
        category = rng.choice(list(EMOJI_CATEGORIES.keys()))

    emojis = EMOJI_CATEGORIES.get(category, EMOJI_CATEGORIES["positive"])
    return rng.choice(emojis)


def get_random_special_char(rng: random.Random, category: str | None = None) -> str:
    """Get a random special Unicode character.

    Args:
        rng: Random instance.
        category: Character category. If None, randomly selected.

    Returns:
        Single character string.
    """
    if category is None:
        category = rng.choice(list(SPECIAL_UNICODE.keys()))

    chars = SPECIAL_UNICODE.get(category, SPECIAL_UNICODE["diacritics"])
    return rng.choice(chars)


def is_rtl_language(language: str) -> bool:
    """Check if a language uses RTL script.

    Args:
        language: Language code.

    Returns:
        True if RTL, False otherwise.
    """
    return language in RTL_LANGUAGES


def get_language_direction(language: str) -> str:
    """Get text direction for a language.

    Args:
        language: Language code.

    Returns:
        "rtl" or "ltr".
    """
    lang_data = LANGUAGES.get(language, {})
    return lang_data.get("direction", "ltr")
