"""Centralizacao das credenciais usadas pelo aplicativo."""

from __future__ import annotations

import os
from typing import Any, Final, Optional

try:  # pragma: no cover - Streamlit pode nao estar disponivel em testes automatizados
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - fallback para execucao sem Streamlit
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
    """Tenta ler um segredo do ``st.secrets`` quando disponivel."""

    if st is None or not hasattr(st, "secrets"):
        return None

    secrets_mapping = getattr(st, "secrets")
    if key not in secrets_mapping:
        return None

    return _normalize(secrets_mapping.get(key))


def _load_from_env(key: str) -> Optional[str]:
    """Obtem o valor correspondente de ``os.environ``."""

    return _normalize(os.getenv(key))


def _get_secret(key: str) -> Optional[str]:
    """Busca um segredo no Streamlit (quando disponivel) ou no ambiente."""

    return _load_from_streamlit(key) or _load_from_env(key)


def require_secret(key: str, *, default: str | None = None) -> str:
    """Obtem uma credencial obrigatoria ou lanca erro com mensagem amigavel."""

    value = _get_secret(key)
    if value:
        return value

    if default is not None:
        return default

    raise RuntimeError(
        "A credencial '%s' nao foi configurada. Defina-a no .streamlit/secrets.toml "
        "ou exporte uma variavel de ambiente com esse nome." % key
    )


def optional_secret(key: str, *, default: str | None = None) -> Optional[str]:
    """Obtem uma credencial opcional com suporte a valor padrao."""

    value = _get_secret(key)
    if value is not None:
        return value

    return default


SUPABASE_URL: Final[str] = require_secret("SUPABASE_URL")
SUPABASE_ANON_KEY: Final[str] = require_secret("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY: Final[str] = require_secret("SUPABASE_SERVICE_KEY")

ADMIN_PASSWORD: Final[Optional[str]] = optional_secret("ADMIN_PASSWORD")
MATCH_ENTRY_PASSWORD: Final[Optional[str]] = optional_secret("MATCH_ENTRY_PASSWORD")
SUPERADMIN_PASSWORD: Final[Optional[str]] = optional_secret("SUPERADMIN_PASSWORD")
RESEND_API_KEY: Final[Optional[str]] = optional_secret("RESEND_API_KEY")
INVITE_FROM_EMAIL: Final[str] = optional_secret("INVITE_FROM_EMAIL") or "noreply@redinha.app"

__all__ = [
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "SUPABASE_SERVICE_KEY",
    "ADMIN_PASSWORD",
    "MATCH_ENTRY_PASSWORD",
    "SUPERADMIN_PASSWORD",
    "optional_secret",
    "require_secret",
]
