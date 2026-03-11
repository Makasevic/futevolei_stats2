"""Gerenciamento de parametros configuraveis do aplicativo."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict

try:  # pragma: no cover - Streamlit pode nao estar disponivel em testes
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - fallback para execucao sem Streamlit
    st = None  # type: ignore[assignment]


_STATE_KEY = "app_config"


@dataclass(frozen=True)
class AppConfig:
    """Parametros ajustaveis do aplicativo com valores padrao."""

    min_participation_ratio: float = 0.20
    min_duo_matches: int = 5
    tendencia_short_months: int = 3
    tendencia_long_months: int = 12
    tendencia_threshold: float = 1.0


DEFAULT_CONFIG = AppConfig()


def _normalize_dict(values: Dict[str, Any]) -> AppConfig:
    """Converte um dicionario em :class:`AppConfig`, preenchendo padroes."""

    base = asdict(DEFAULT_CONFIG)
    base.update(values)
    return AppConfig(**base)


def _ensure_state() -> None:
    """Garante que o ``st.session_state`` possua a configuracao padrao."""

    if st is None:
        return
    if _STATE_KEY not in st.session_state:
        st.session_state[_STATE_KEY] = DEFAULT_CONFIG


def get_config() -> AppConfig:
    """Obtem a configuracao atual, inicializando-a se necessario."""

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
    """Atualiza a configuracao atual com os valores fornecidos."""

    if st is None:
        global DEFAULT_CONFIG
        DEFAULT_CONFIG = replace(DEFAULT_CONFIG, **changes)
        return DEFAULT_CONFIG

    config_atual = get_config()
    novo = replace(config_atual, **changes)
    st.session_state[_STATE_KEY] = novo
    return novo


__all__ = ["AppConfig", "DEFAULT_CONFIG", "get_config", "update_config", "_normalize_dict"]
