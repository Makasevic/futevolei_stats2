from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any, Dict, Tuple

import pandas as pd

from awards import get_awards_for_year
from src.redinha_stats.config.app_settings import get_config
from src.redinha_stats.config.ui_config import get_ui_config
from src.redinha_stats.domain.matches.extraction import get_matches
from src.redinha_stats.domain.matches.preparation import preparar_dataframe
from src.redinha_stats.domain.matches.processing import (
    filtrar_dados,
    preparar_dados_duplas,
    preparar_dados_individuais,
)


def current_ui_config():
    return get_ui_config()


@lru_cache(maxsize=1)
def fetch_base_dataframe() -> pd.DataFrame:
    matches = get_matches()
    df = preparar_dataframe(matches)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index(pd.to_datetime(df["date"], errors="coerce"))
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

    return df.sort_index()


def reset_cached_dataframe() -> None:
    fetch_base_dataframe.cache_clear()


def filtrar_por_intervalo(
    df: pd.DataFrame, inicio: str | None, fim: str | None
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")

    try:
        data_inicio = pd.to_datetime(inicio) if inicio else None
    except (TypeError, ValueError):
        data_inicio = None

    try:
        data_fim = pd.to_datetime(fim) if fim else None
    except (TypeError, ValueError):
        data_fim = None

    if data_inicio is None and data_fim is None:
        return df

    filtrado = df
    if data_inicio is not None:
        filtrado = filtrado[filtrado.index >= data_inicio]
    if data_fim is not None:
        filtrado = filtrado[filtrado.index <= data_fim]
    return filtrado


def excluded_players() -> set:
    return set(current_ui_config().excluded_players)


def filter_rankings(
    modo: str,
    periodo: str | None,
    ano: str | None,
    mes: str | None,
    inicio: str | None,
    fim: str | None,
    data: str | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_df = fetch_base_dataframe()
    config = get_config()

    if modo == "Ano" and ano:
        df_filtrado = filtrar_dados(base_df, "Ano", ano)
    elif modo in {"Mes/Ano", "Mês/Ano"} and mes:
        df_filtrado = filtrar_dados(base_df, "Mês/Ano", mes)
    elif modo == "Intervalo":
        df_filtrado = filtrar_por_intervalo(base_df, inicio, fim)
    elif modo == "Data":
        df_filtrado = filtrar_dados(base_df, "Data", data) if data else base_df.iloc[0:0]
    else:
        df_filtrado = filtrar_dados(base_df, "Dias", periodo) if periodo else base_df

    jogadores = preparar_dados_individuais(df_filtrado)
    duplas = preparar_dados_duplas(df_filtrado)

    excluidos = excluded_players()
    if excluidos:
        jogadores = jogadores[~jogadores["jogadores"].isin(excluidos)]
        duplas = duplas[~duplas["duplas"].isin(excluidos)]

    aplicar_minimos = not ((modo == "Dia" and periodo == "1 dia") or modo == "Data")
    if aplicar_minimos:
        media_top_10 = jogadores["jogos"].nlargest(10).mean()
        if media_top_10 > 0:
            limiar = media_top_10 * config.min_participation_ratio
            jogadores = jogadores[jogadores["jogos"] >= limiar]

        if config.min_duo_matches > 0 and not duplas.empty:
            max_jogos_duplas = int(duplas["jogos"].max())
            limiar_duplas = (
                config.min_duo_matches
                if max_jogos_duplas >= config.min_duo_matches
                else max_jogos_duplas
            )
            if limiar_duplas > 0:
                duplas = duplas[duplas["jogos"] >= limiar_duplas]

    return base_df, jogadores.reset_index(drop=True), duplas.reset_index(drop=True)


def build_awards_data(year: int) -> list[Dict[str, Any]]:
    awards = get_awards_for_year(year)
    for award in awards:
        sorted_votes = sorted(award["votes"].items(), key=lambda item: item[1], reverse=True)
        award["top_three"] = sorted_votes[:3]
        winner_votes = sorted_votes[0][1]
        award["winners"] = [
            {"name": name, "votes": votes}
            for name, votes in sorted_votes
            if votes == winner_votes
        ]
        award["winner_votes"] = winner_votes
        award["total_votes"] = sum(award["votes"].values())
    return awards


def format_champion_names(names: list[str]) -> str:
    if not names:
        return "-"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} e {names[1]}"
    return f"{', '.join(names[:-1])} e {names[-1]}"


def build_awards_champions(awards: list[Dict[str, Any]]) -> Dict[str, Any]:
    trophies = {"positive": Counter(), "negative": Counter()}
    for award in awards:
        category = award.get("category_type", "positive")
        for winner in award.get("winners", []):
            trophies[category][winner["name"]] += 1

    champions: Dict[str, Any] = {}
    for category, counts in trophies.items():
        if not counts:
            champions[category] = {"names": "-", "count": 0}
            continue
        max_count = max(counts.values())
        names = sorted([name for name, count in counts.items() if count == max_count])
        champions[category] = {
            "names": format_champion_names(names),
            "count": max_count,
        }
    return champions
