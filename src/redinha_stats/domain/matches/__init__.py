"""Core match preparation, filtering and extraction helpers."""

from .extraction import get_matches
from .preparation import criar_colunas_duplas, preparar_dataframe
from .processing import (
    filtrar_dados,
    preparar_dados_confrontos_duplas,
    preparar_dados_confrontos_jogadores,
    preparar_dados_duplas,
    preparar_dados_individuais,
)

__all__ = [
    "criar_colunas_duplas",
    "filtrar_dados",
    "get_matches",
    "preparar_dataframe",
    "preparar_dados_confrontos_duplas",
    "preparar_dados_confrontos_jogadores",
    "preparar_dados_duplas",
    "preparar_dados_individuais",
]
