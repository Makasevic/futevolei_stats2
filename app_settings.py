"""Gerenciamento de parâmetros configuráveis do aplicativo."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict

try:  # pragma: no cover - Streamlit pode não estar disponível em testes
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - fallback para execução sem Streamlit
    st = None  # type: ignore[assignment]


_STATE_KEY = "app_config"


@dataclass(frozen=True)
class AppConfig:
    """Parâmetros ajustáveis do aplicativo com valores padrão."""

    min_participation_ratio: float = 0.20
    min_duo_matches: int = 5
    tendencia_short_months: int = 3
    tendencia_long_months: int = 12
    tendencia_threshold: float = 1.0


DEFAULT_CONFIG = AppConfig()


def _normalize_dict(values: Dict[str, Any]) -> AppConfig:
    """Converte um dicionário em :class:`AppConfig`, preenchendo padrões."""

    base = asdict(DEFAULT_CONFIG)
    base.update(values)
    return AppConfig(**base)


def _ensure_state() -> None:
    """Garante que o ``st.session_state`` possua a configuração padrão."""

    if st is None:
        return
    if _STATE_KEY not in st.session_state:
        st.session_state[_STATE_KEY] = DEFAULT_CONFIG


def get_config() -> AppConfig:
    """Obtém a configuração atual, inicializando-a se necessário."""

    if st is None:
        return DEFAULT_CONFIG

    _ensure_state()
    atual = st.session_state.get(_STATE_KEY, DEFAULT_CONFIG)

    if isinstance(atual, AppConfig):
        return atual

    if isinstance(atual, dict):
        config = _normalize_dict(atual)
    else:
        config = DEFAULT_CONFIG

    st.session_state[_STATE_KEY] = config
    return config


def update_config(**changes: Any) -> AppConfig:
    """Atualiza a configuração atual com os valores fornecidos."""

    if st is None:
        global DEFAULT_CONFIG
        DEFAULT_CONFIG = replace(DEFAULT_CONFIG, **changes)
        return DEFAULT_CONFIG

    config_atual = get_config()
    novo = replace(config_atual, **changes)
    st.session_state[_STATE_KEY] = novo
    return novo


__all__ = ["AppConfig", "get_config", "update_config"]
