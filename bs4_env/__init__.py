"""BeautifulSoup RL Environment package."""

from bs4_env.config import EnvConfig
from bs4_env.registry import get_archetype, list_archetypes, register

__all__ = [
    "EnvConfig",
    "register",
    "get_archetype",
    "list_archetypes",
]
