"""Centralização das credenciais usadas pelo aplicativo."""

from __future__ import annotations

import os
from typing import Any, Final, Optional

try:  # pragma: no cover - Streamlit pode não estar disponível em testes automatizados
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - fallback para execução sem Streamlit
    st = None  # type: ignore[assignment]


def _normalize(value: Any) -> Optional[str]:
    """Converte valores em ``str`` e descarta entradas vazias."""

    if value is None:
        return None

    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None

    return str(value).strip() or None


def _load_from_streamlit(key: str) -> Optional[str]:
    """Tenta ler um segredo do ``st.secrets`` quando disponível."""

    if st is None or not hasattr(st, "secrets"):
        return None

    secrets_mapping = getattr(st, "secrets")
    if key not in secrets_mapping:
        return None

    return _normalize(secrets_mapping.get(key))


def _load_from_env(key: str) -> Optional[str]:
    """Obtém o valor correspondente de ``os.environ``."""

    return _normalize(os.getenv(key))


def _get_secret(key: str) -> Optional[str]:
    """Busca um segredo no Streamlit (quando disponível) ou no ambiente."""

    return _load_from_streamlit(key) or _load_from_env(key)


def require_secret(key: str, *, default: str | None = None) -> str:
    """Obtém uma credencial obrigatória ou lança erro com mensagem amigável."""

    value = _get_secret(key)
    if value:
        return value

    if default is not None:
        return default

    raise RuntimeError(
        "A credencial '%s' não foi configurada. Defina-a no .streamlit/secrets.toml "
        "ou exporte uma variável de ambiente com esse nome." % key
    )


def optional_secret(key: str, *, default: str | None = None) -> Optional[str]:
    """Obtém uma credencial opcional com suporte a valor padrão."""

    value = _get_secret(key)
    if value is not None:
        return value

    return default

SUPABASE_URL: Final[str] = require_secret("SUPABASE_URL")
SUPABASE_ANON_KEY: Final[str] = require_secret("SUPABASE_ANON_KEY")

ADMIN_PASSWORD: Final[Optional[str]] = optional_secret("ADMIN_PASSWORD")
MATCH_ENTRY_PASSWORD: Final[Optional[str]] = optional_secret("MATCH_ENTRY_PASSWORD")

__all__ = [
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "ADMIN_PASSWORD",
    "MATCH_ENTRY_PASSWORD",
    "optional_secret",
    "require_secret",
]
