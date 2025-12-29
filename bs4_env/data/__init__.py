"""Data modules for BeautifulSoup RL environment."""

from bs4_env.data.i18n_content import (
    EMOJI_CATEGORIES,
    LANGUAGES,
    RTL_LANGUAGES,
    SPECIAL_UNICODE,
    get_language_direction,
    get_random_emoji,
    get_random_language,
    get_random_phrase,
    get_random_special_char,
    get_random_word,
    is_rtl_language,
)

__all__ = [
    "LANGUAGES",
    "RTL_LANGUAGES",
    "EMOJI_CATEGORIES",
    "SPECIAL_UNICODE",
    "get_random_language",
    "get_random_phrase",
    "get_random_word",
    "get_random_emoji",
    "get_random_special_char",
    "is_rtl_language",
    "get_language_direction",
]
