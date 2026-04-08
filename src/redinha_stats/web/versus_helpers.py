"""Funções de análise de confrontos e estatísticas individuais para a página Versus."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.redinha_stats.domain.matches.processing import (
    preparar_dados_duplas,
    preparar_dados_individuais,
)


# ---------------------------------------------------------------------------
# Estatísticas individuais
# ---------------------------------------------------------------------------

def estatisticas_jogador_individual(
    df: pd.DataFrame,
    jogador: str,
    excluded_players: frozenset[str] | set[str] | None = None,
) -> Dict[str, int | float]:
    tabela = preparar_dados_individuais(df)
    if excluded_players:
        tabela = tabela[~tabela["jogadores"].isin(excluded_players)]
    linha = tabela.loc[tabela["jogadores"] == jogador]

    if linha.empty:
        return {"jogador": jogador, "jogos": 0, "vitorias": 0, "derrotas": 0, "saldo": 0, "aproveitamento": 0.0}

    registro = linha.iloc[0]
    return {
        "jogador": jogador,
        "jogos": int(registro.get("jogos", 0)),
        "vitorias": int(registro.get("vitórias", 0)),
        "derrotas": int(registro.get("derrotas", 0)),
        "saldo": int(registro.get("saldo", 0)),
        "aproveitamento": float(registro.get("aproveitamento", 0.0)),
    }


def estatisticas_dupla(
    df: pd.DataFrame,
    dupla: str,
) -> Dict[str, int | float]:
    tabela = preparar_dados_duplas(df[["winner1", "winner2", "loser1", "loser2"]])
    tabela = tabela[~tabela["duplas"].str.contains("Outro", na=False)]
    linha = tabela.loc[tabela["duplas"] == dupla]

    if linha.empty:
        return {"jogador": dupla, "jogos": 0, "vitorias": 0, "derrotas": 0, "saldo": 0, "aproveitamento": 0.0}

    registro = linha.iloc[0]
    return {
        "jogador": dupla,
        "jogos": int(registro.get("jogos", 0)),
        "vitorias": int(registro.get("vitórias", 0)),
        "derrotas": int(registro.get("derrotas", 0)),
        "saldo": int(registro.get("saldo", 0)),
        "aproveitamento": float(registro.get("aproveitamento", 0.0)),
    }


# ---------------------------------------------------------------------------
# Confrontos diretos
# ---------------------------------------------------------------------------

def _ordenar_por_data(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    if df.empty:
        return df
    datas = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index, errors="coerce")
    result = df.copy()
    result["data_partida"] = datas
    return result.dropna(subset=["data_partida"]).sort_values("data_partida", ascending=ascending)


def _jogos_recentes_jogadores(confrontos_df: pd.DataFrame, jogador1: str, limite: int = 30) -> List[Dict[str, Any]]:
    if confrontos_df.empty:
        return []
    df_ord = _ordenar_por_data(confrontos_df, ascending=False)
    jogos = []
    for row in df_ord.head(limite).itertuples():
        jogos.append({
            "data": row.data_partida.strftime("%d/%m/%Y"),
            "vencedores": " e ".join([row.winner1, row.winner2]),
            "perdedores": " e ".join([row.loser1, row.loser2]),
            "venceu": jogador1 in [row.winner1, row.winner2],
        })
    return jogos


def _jogos_recentes_duplas(confrontos_df: pd.DataFrame, dupla1: str, limite: int = 30) -> List[Dict[str, Any]]:
    if confrontos_df.empty:
        return []
    df_ord = _ordenar_por_data(confrontos_df, ascending=False)
    jogos = []
    for row in df_ord.head(limite).itertuples():
        jogos.append({
            "data": row.data_partida.strftime("%d/%m/%Y"),
            "vencedores": " e ".join([row.winner1, row.winner2]),
            "perdedores": " e ".join([row.loser1, row.loser2]),
            "venceu": row.dupla_winner == dupla1,
        })
    return jogos


def _serie_mensal(confrontos_df: pd.DataFrame) -> tuple[list, pd.Series]:
    if not isinstance(confrontos_df.index, pd.DatetimeIndex):
        confrontos_df = confrontos_df.copy()
        confrontos_df.index = pd.to_datetime(confrontos_df.index, errors="coerce")
    if getattr(confrontos_df.index, "tz", None) is not None:
        confrontos_df.index = confrontos_df.index.tz_convert(None)
    confrontos_df = confrontos_df[~confrontos_df.index.isna()]
    if confrontos_df.empty:
        return [], confrontos_df

    inicio = confrontos_df.index.min().normalize().replace(day=1)
    fim = pd.Timestamp.now().normalize().replace(day=1)
    meses_range = pd.date_range(start=inicio, end=fim, freq="MS")
    mensal = (
        confrontos_df["resultado_j1"]
        .resample("MS")
        .sum()
        .reindex(meses_range, fill_value=0)
        .sort_index()
    )
    saldo_acumulado = mensal.cumsum()
    serie = [
        {"label": p.strftime("%b/%Y"), "saldo": int(mensal.loc[p]), "acumulado": int(saldo_acumulado.loc[p])}
        for p in mensal.index
    ]
    return serie[-12:], confrontos_df


_EMPTY_CONFRONTO = {"total": 0, "vitorias_j1": 0, "vitorias_j2": 0, "saldo": 0, "serie_mensal": [], "jogos_recentes": []}


def confronto_direto(df: pd.DataFrame, jogador1: str, jogador2: str) -> Dict[str, Any]:
    if jogador1 == jogador2:
        return dict(_EMPTY_CONFRONTO)

    mask_j1 = (
        ((df["winner1"] == jogador1) | (df["winner2"] == jogador1))
        & ((df["loser1"] == jogador2) | (df["loser2"] == jogador2))
    )
    mask_j2 = (
        ((df["winner1"] == jogador2) | (df["winner2"] == jogador2))
        & ((df["loser1"] == jogador1) | (df["loser2"] == jogador1))
    )
    confrontos_df = df[mask_j1 | mask_j2].copy()

    if confrontos_df.empty:
        return dict(_EMPTY_CONFRONTO)

    confrontos_df["resultado_j1"] = confrontos_df.apply(
        lambda row: 1 if (row["winner1"] == jogador1 or row["winner2"] == jogador1) else -1,
        axis=1,
    )
    serie, confrontos_df = _serie_mensal(confrontos_df)

    vitorias_j1 = int(mask_j1.sum())
    vitorias_j2 = int(mask_j2.sum())
    return {
        "total": len(confrontos_df) if not confrontos_df.empty else (vitorias_j1 + vitorias_j2),
        "vitorias_j1": vitorias_j1,
        "vitorias_j2": vitorias_j2,
        "saldo": vitorias_j1 - vitorias_j2,
        "serie_mensal": serie,
        "jogos_recentes": _jogos_recentes_jogadores(confrontos_df, jogador1),
    }


def confronto_direto_duplas(df: pd.DataFrame, dupla1: str, dupla2: str) -> Dict[str, Any]:
    if dupla1 == dupla2:
        return dict(_EMPTY_CONFRONTO)

    mask_d1 = (df["dupla_winner"] == dupla1) & (df["dupla_loser"] == dupla2)
    mask_d2 = (df["dupla_winner"] == dupla2) & (df["dupla_loser"] == dupla1)
    confrontos_df = df[mask_d1 | mask_d2].copy()

    if confrontos_df.empty:
        return dict(_EMPTY_CONFRONTO)

    confrontos_df["resultado_j1"] = confrontos_df["dupla_winner"].apply(
        lambda v: 1 if v == dupla1 else -1
    )
    serie, confrontos_df = _serie_mensal(confrontos_df)

    vitorias_d1 = int(mask_d1.sum())
    vitorias_d2 = int(mask_d2.sum())
    return {
        "total": len(confrontos_df) if not confrontos_df.empty else (vitorias_d1 + vitorias_d2),
        "vitorias_j1": vitorias_d1,
        "vitorias_j2": vitorias_d2,
        "saldo": vitorias_d1 - vitorias_d2,
        "serie_mensal": serie,
        "jogos_recentes": _jogos_recentes_duplas(confrontos_df, dupla1),
    }
