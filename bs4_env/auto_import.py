"""Auto-import module to populate the archetype registry.

This module imports all generator modules to ensure their @register
decorators run and populate the registry.

Import this module before accessing the registry to ensure all
archetypes are available.
"""

# Track registration state
_REGISTERED = False


def ensure_registered() -> None:
    """Ensure all archetype generators are imported and registered exactly once.

    Call this function to guarantee the registry is populated.
    This is idempotent - safe to call multiple times.

    The imports happen inside this function (rather than at module level)
    to allow lazy initialization and clearer control flow.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    # Import generator modules to trigger @register decorators
    # Add new generator modules here as they are created

    # Primer archetypes (ultra-simple bootstrap tasks for 0% models)
    # MVP Phase 1 generators
    # MVP Phase 2 generators (advanced difficulty)
    # BS4 Gotcha generators (from documentation research)
    # Hard archetypes (semantic reasoning, aggregation, multi-hop)
    # Multi-step archetypes (navigation between pages)
    from bs4_env.generators import (  # noqa: F401
        mvp_advanced,
        mvp_core_extraction,
        mvp_core_remaining,
        mvp_error_bait,
        mvp_forms,
        mvp_hard,
        mvp_i18n,
        mvp_json_ld,
        mvp_limitations,
        mvp_multistep,
        mvp_multivalue_class,
        mvp_navigablestring,
        mvp_tables,
        mvp_whitespace_sibling,
        primer,
    )

    _REGISTERED = True


# Auto-register on module import for backwards compatibility
ensure_registered()
