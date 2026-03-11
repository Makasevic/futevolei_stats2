"""Configuration modules for the application."""

from .app_settings import AppConfig, get_config, update_config
from .settings import (
    ADMIN_PASSWORD,
    MATCH_ENTRY_PASSWORD,
    SUPABASE_ANON_KEY,
    SUPABASE_SERVICE_KEY,
    SUPABASE_URL,
    optional_secret,
    require_secret,
)
from .ui_config import BrandingConfig, NavigationLink, UI_CONFIG, UiConfig, get_ui_config

__all__ = [
    "ADMIN_PASSWORD",
    "AppConfig",
    "BrandingConfig",
    "MATCH_ENTRY_PASSWORD",
    "NavigationLink",
    "SUPABASE_ANON_KEY",
    "SUPABASE_SERVICE_KEY",
    "SUPABASE_URL",
    "UI_CONFIG",
    "UiConfig",
    "get_config",
    "get_ui_config",
    "optional_secret",
    "require_secret",
    "update_config",
]
