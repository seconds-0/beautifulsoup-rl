"""Auto-import module to populate the archetype registry.

This module imports all generator modules to ensure their @register
decorators run and populate the registry.

Import this module before accessing the registry to ensure all
archetypes are available.
"""

# Import generator modules to trigger @register decorators
# Add new generator modules here as they are created

# MVP Phase 1 generators
from bs4_env.generators import mvp_core_extraction  # noqa: F401
from bs4_env.generators import mvp_tables  # noqa: F401
from bs4_env.generators import mvp_error_bait  # noqa: F401
from bs4_env.generators import mvp_limitations  # noqa: F401
from bs4_env.generators import mvp_i18n  # noqa: F401

# MVP Phase 2 generators (advanced difficulty)
from bs4_env.generators import mvp_advanced  # noqa: F401

# BS4 Gotcha generators (from documentation research)
from bs4_env.generators import mvp_multivalue_class  # noqa: F401
from bs4_env.generators import mvp_whitespace_sibling  # noqa: F401
from bs4_env.generators import mvp_json_ld  # noqa: F401
from bs4_env.generators import mvp_navigablestring  # noqa: F401


def ensure_registered() -> None:
    """Ensure all generators are imported and registered.

    Call this function to guarantee the registry is populated.
    This is idempotent - safe to call multiple times.
    """
    # The imports above have already run, but this function
    # provides an explicit API for ensuring registration.
    pass
