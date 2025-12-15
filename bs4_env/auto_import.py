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
# from bs4_env.generators import mvp_traversal  # noqa: F401
# from bs4_env.generators import mvp_forms  # noqa: F401
# from bs4_env.generators import mvp_malformed  # noqa: F401
# from bs4_env.generators import mvp_normalization  # noqa: F401
# from bs4_env.generators import mvp_filters  # noqa: F401
from bs4_env.generators import mvp_error_bait  # noqa: F401
from bs4_env.generators import mvp_limitations  # noqa: F401

# Phase 2 generators (uncomment when implemented)
# from bs4_env.generators import phase2_bot_detection  # noqa: F401
# from bs4_env.generators import phase2_obfuscation  # noqa: F401
# from bs4_env.generators import phase2_i18n  # noqa: F401


def ensure_registered() -> None:
    """Ensure all generators are imported and registered.

    Call this function to guarantee the registry is populated.
    This is idempotent - safe to call multiple times.
    """
    # The imports above have already run, but this function
    # provides an explicit API for ensuring registration.
    pass
