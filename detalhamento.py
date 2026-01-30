from __future__ import annotations

"""Helpers para montar a aba de detalhamento sem Streamlit."""

from collections import defaultdict
import math
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app_settings import get_config
from processing import (
    preparar_dados_confrontos_duplas,
    preparar_dados_confrontos_jogadores,
    preparar_dados_individuais,
)


def _formatar_estrelas(percentual: Optional[float]) -> str:
    if percentual is None or pd.isna(percentual):
        return "—"

    rating = 1
    for limite in (20, 40, 60, 80):
        if percentual >= limite:
            rating += 1

    rating = max(1, min(5, rating))
    return "⭐" * rating


def _calcular_tendencia(
    score_diario: Optional[pd.Series], config
) -> Tuple[str, Optional[float], Optional[str]]:
    tendencia = "Neutro"
    tendencia_delta = None
    tendencia_observacao = None

    if score_diario is not None and isinstance(score_diario.index, pd.DatetimeIndex):
        score_diario = score_diario.sort_index()
        if not score_diario.empty:
            ultimo = score_diario.index.max()
            inicio_curto = ultimo - pd.DateOffset(months=config.tendencia_short_months)
            inicio_longo = ultimo - pd.DateOffset(months=config.tendencia_long_months)

            periodo_curto = score_diario[score_diario.index >= inicio_curto]
            periodo_longo = score_diario[score_diario.index >= inicio_longo]
            if periodo_longo.empty:
                periodo_longo = score_diario

            if periodo_curto.empty:
                tendencia_observacao = (
                    f"Sem jogos nos últimos {config.tendencia_short_months} meses"
                )
            else:
                media_curto = periodo_curto.mean()
                media_longo = periodo_longo.mean()
                diff = media_curto - media_longo
                tendencia_delta = diff
                if diff > config.tendencia_threshold:
                    tendencia = "Alta"
                elif diff < -config.tendencia_threshold:
                    tendencia = "Baixa"
                else:
                    tendencia = "Neutro"
    elif score_diario is None:
        tendencia_observacao = None
    else:
        tendencia_observacao = "Sem dados suficientes"

    return tendencia, tendencia_delta, tendencia_observacao


def _agrupar_score_mensal(score_percentual: pd.Series) -> List[Dict[str, object]]:
    if score_percentual is None or score_percentual.empty:
        return []

    if not isinstance(score_percentual.index, pd.DatetimeIndex):
        return []

    score_percentual = score_percentual.sort_index()
    score_mensal = score_percentual.resample("MS").mean().dropna()
    if score_mensal.empty:
        return []

    ultimo_mes = score_mensal.index.max()
    inicio_periodo = (ultimo_mes - pd.DateOffset(months=11)).normalize()
    score_mensal = score_mensal[score_mensal.index >= inicio_periodo]
    media_movel_3m = score_mensal.rolling(3, min_periods=1).mean()

    return [
        {
            "data": idx.strftime("%m/%Y"),
            "score": float(valor),
            "media_movel": float(media_movel_3m.get(idx, np.nan))
            if not pd.isna(media_movel_3m.get(idx, np.nan))
            else None,
        }
        for idx, valor in score_mensal.items()
    ]


def _ordenar_partidas_por_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        datas_partidas = df.index
    else:
        datas_partidas = pd.to_datetime(df.index, errors="coerce")

    df_ordenado = df.copy()
    df_ordenado["data_partida"] = datas_partidas
    df_ordenado = df_ordenado.dropna(subset=["data_partida"])
    return df_ordenado.sort_values("data_partida", ascending=False)


def _montar_jogos_recentes_jogador(df: pd.DataFrame, jogador: str, limite: int = 30) -> List[Dict[str, object]]:
    if df.empty:
        return []

    df_ordenado = _ordenar_partidas_por_data(df)
    if df_ordenado.empty:
        return []

    colunas = ["winner1", "winner2", "loser1", "loser2"]
    if not all(coluna in df_ordenado.columns for coluna in colunas):
        return []

    mask = df_ordenado[colunas].eq(jogador).any(axis=1)
    partidas = df_ordenado[mask].head(limite)
    if partidas.empty:
        return []

    jogos = []
    for row in partidas.itertuples():
        vencedores = " e ".join([row.winner1, row.winner2])
        perdedores = " e ".join([row.loser1, row.loser2])
        venceu = jogador in [row.winner1, row.winner2]
        data_partida = row.data_partida.strftime("%d/%m/%Y")
        jogos.append(
            {
                "data": data_partida,
                "vencedores": vencedores,
                "perdedores": perdedores,
                "venceu": venceu,
            }
        )
    return jogos


def _montar_jogos_recentes_dupla(
    df: pd.DataFrame, jogador1: str, jogador2: str, limite: int = 30
) -> List[Dict[str, object]]:
    if df.empty:
        return []

    dupla_selecionada = " e ".join(sorted([jogador1, jogador2]))
    df_ordenado = _ordenar_partidas_por_data(df)
    if df_ordenado.empty:
        return []

    colunas = ["dupla_winner", "dupla_loser", "winner1", "winner2", "loser1", "loser2"]
    if not all(coluna in df_ordenado.columns for coluna in colunas):
        return []

    mask = (df_ordenado["dupla_winner"] == dupla_selecionada) | (
        df_ordenado["dupla_loser"] == dupla_selecionada
    )
    partidas = df_ordenado[mask].head(limite)
    if partidas.empty:
        return []

    jogos = []
    for row in partidas.itertuples():
        vencedores = " e ".join([row.winner1, row.winner2])
        perdedores = " e ".join([row.loser1, row.loser2])
        venceu = row.dupla_winner == dupla_selecionada
        data_partida = row.data_partida.strftime("%d/%m/%Y")
        jogos.append(
            {
                "data": data_partida,
                "vencedores": vencedores,
                "perdedores": perdedores,
                "venceu": venceu,
            }
        )
    return jogos


def _aplicar_regras_ranking(tabela: pd.DataFrame, config) -> pd.DataFrame:
    ranking = tabela.copy()
    if ranking.empty or "jogadores" not in ranking.columns:
        return ranking.iloc[0:0]

    ranking = ranking[~ranking["jogadores"].str.contains("Outro", na=False)].copy()
    if ranking.empty:
        return ranking

    media_top_10 = ranking["jogos"].nlargest(10).mean()
    if media_top_10 > 0:
        limiar = media_top_10 * config.min_participation_ratio
        ranking = ranking[ranking["jogos"] >= limiar].copy()

    if ranking.empty:
        return ranking

    ranking = ranking.sort_values(
        ["aproveitamento", "saldo", "vitórias"], ascending=[False, False, False]
    ).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1
    return ranking


def _formatar_parcerias(
    parcerias: Dict[str, Dict[str, int]],
    total_jogos: int,
    parceiros_validos: Optional[set[str]] = None,
    jogador: Optional[str] = None,
    limite: int = 10,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if parceiros_validos:
        parceiros_base = [nome for nome in parceiros_validos if nome and nome != jogador]
    else:
        parceiros_base = [nome for nome in parcerias.keys() if nome and nome != jogador]

    if not parceiros_base:
        return [], []

    parceiros_series = pd.Series(
        {nome: int(parcerias.get(nome, {}).get("jogos", 0)) for nome in parceiros_base},
        dtype=int,
    )

    def _montar_tabela(series: pd.Series) -> List[Dict[str, object]]:
        if series.empty:
            return []
        registros = []
        for parceiro, jogos in series.items():
            stats = parcerias.get(parceiro, {})
            vitorias = int(stats.get("vitorias", 0))
            aproveitamento = (vitorias / jogos * 100) if jogos else None
            participacao = f"{(jogos / total_jogos * 100):.1f}%" if total_jogos else "0%"
            registros.append(
                {
                    "jogador": parceiro,
                    "jogos": int(jogos),
                    "participacao": participacao,
                    "aproveitamento": f"{aproveitamento:.1f}%" if aproveitamento is not None else "-",
                }
            )
        return registros

    parceiros_series = parceiros_series.sort_values(ascending=False)
    total_parceiros = int(len(parceiros_series))
    if total_parceiros <= limite * 2:
        top_count = int(math.ceil(total_parceiros / 2))
        bottom_count = total_parceiros - top_count
    else:
        top_count = limite
        bottom_count = limite

    top_series = parceiros_series.head(top_count)
    bottom_series = parceiros_series.tail(bottom_count) if bottom_count else parceiros_series.iloc[0:0]

    maiores_parcerias = _montar_tabela(top_series)
    menores_parcerias = _montar_tabela(bottom_series)
    return maiores_parcerias, menores_parcerias


def calcular_metricas_jogador(df: pd.DataFrame, jogador: str) -> Dict[str, object]:
    config = get_config()

    vitorias_mask = df[["winner1", "winner2"]] == jogador
    derrotas_mask = df[["loser1", "loser2"]] == jogador

    vitorias_por_dia = vitorias_mask.sum(axis=1).groupby(df.index).sum()
    derrotas_por_dia = derrotas_mask.sum(axis=1).groupby(df.index).sum()

    jogos_totais_por_dia = vitorias_por_dia + derrotas_por_dia
    score_percentual = (vitorias_por_dia / jogos_totais_por_dia * 100).dropna().sort_index()
    total_jogos = int(vitorias_mask.values.sum() + derrotas_mask.values.sum())
    total_vitorias = int(vitorias_mask.values.sum())
    total_derrotas = int(derrotas_mask.values.sum())
    total_score = (total_vitorias / total_jogos * 100) if total_jogos else 0

    parcerias_detalhes: Dict[str, Dict[str, int]] = {}
    confrontos_detalhes: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: {"vitorias": 0, "derrotas": 0})

    def registrar_parceria(parceiro: str, venceu: bool) -> None:
        if not parceiro or parceiro == jogador:
            return
        if "Outro" in parceiro:
            return
        stats = parcerias_detalhes.setdefault(parceiro, {"jogos": 0, "vitorias": 0, "derrotas": 0})
        stats["jogos"] += 1
        if venceu:
            stats["vitorias"] += 1
        else:
            stats["derrotas"] += 1

    def registrar_confronto(adversario: str, venceu: bool) -> None:
        if not adversario or adversario == jogador:
            return
        if "Outro" in adversario:
            return
        registro = confrontos_detalhes[adversario]
        if venceu:
            registro["vitorias"] += 1
        else:
            registro["derrotas"] += 1

    for row in df.itertuples():
        dupla_venc = [row.winner1, row.winner2]
        dupla_perd = [row.loser1, row.loser2]

        if jogador in dupla_venc:
            parceiro = next((p for p in dupla_venc if p != jogador), None)
            registrar_parceria(parceiro, True)
            for adversario in dupla_perd:
                registrar_confronto(adversario, True)

        if jogador in dupla_perd:
            parceiro = next((p for p in dupla_perd if p != jogador), None)
            registrar_parceria(parceiro, False)
            for adversario in dupla_venc:
                registrar_confronto(adversario, False)

    parceiros_series = (
        pd.Series({nome: stats["jogos"] for nome, stats in parcerias_detalhes.items()}, dtype=int)
        if parcerias_detalhes
        else pd.Series(dtype=int)
    )
    parceiros_series = parceiros_series[parceiros_series > 0].sort_values(ascending=False)

    confrontos_dict = {nome: dict(stats) for nome, stats in confrontos_detalhes.items()}

    metricas = _obter_metricas_jogador(
        df,
        jogador,
        total_jogos,
        total_vitorias,
        total_derrotas,
        total_score,
        jogos_totais_por_dia,
        parceiros_series,
        parcerias_detalhes,
        confrontos_dict,
        score_percentual,
    )

    destaque_fregueses, destaque_carrascos = _saldo_confrontos_jogador(df, jogador)

    df_base = df[["winner1", "winner2", "loser1", "loser2"]]
    tabela_geral = preparar_dados_individuais(df_base)
    tabela_geral = _aplicar_regras_ranking(tabela_geral, config)
    parceiros_validos = set(tabela_geral["jogadores"].tolist()) if not tabela_geral.empty else set()

    parcerias_top, parcerias_bottom = _formatar_parcerias(
        parcerias_detalhes,
        total_jogos,
        parceiros_validos,
        jogador=jogador,
        limite=10,
    )

    serie_score = _agrupar_score_mensal(score_percentual)
    jogos_recentes = _montar_jogos_recentes_jogador(df, jogador, limite=30)

    return {
        "metricas": metricas,
        "score_series": serie_score,
        "fregueses": destaque_fregueses,
        "carrascos": destaque_carrascos,
        "parcerias_top": parcerias_top,
        "parcerias_bottom": parcerias_bottom,
        "jogos_recentes": jogos_recentes,
    }


def _obter_metricas_jogador(
    df: pd.DataFrame,
    jogador: str,
    total_jogos: int,
    total_vitorias: int,
    total_derrotas: int,
    total_score: float,
    jogos_por_dia: pd.Series,
    parceiros_series: pd.Series,
    parcerias_detalhes: dict,
    confrontos_detalhes: Optional[dict] = None,
    score_diario: Optional[pd.Series] = None,
) -> dict:
    config = get_config()
    saldo = total_vitorias - total_derrotas

    ranking_geral = None
    rankings_periodo = {"30": None, "60": None, "90": None, "geral": None}
    medalha = "🏐"

    df_base = df[["winner1", "winner2", "loser1", "loser2"]]
    tabela_geral = preparar_dados_individuais(df_base)
    tabela_geral = _aplicar_regras_ranking(tabela_geral, config)
    linha_geral = tabela_geral[tabela_geral["jogadores"] == jogador]
    if not linha_geral.empty:
        ranking_geral = int(linha_geral["rank"].iloc[0])
        if ranking_geral == 1:
            medalha = "🥇"
        elif ranking_geral == 2:
            medalha = "🥈"
        elif ranking_geral == 3:
            medalha = "🥉"

    rankings_periodo["geral"] = ranking_geral

    def _calcular_ranking_periodo(df_periodo: pd.DataFrame) -> Optional[int]:
        if df_periodo.empty:
            return None

        tabela_periodo = preparar_dados_individuais(df_periodo[["winner1", "winner2", "loser1", "loser2"]])
        tabela_periodo = _aplicar_regras_ranking(tabela_periodo, config)
        linha_periodo = tabela_periodo[tabela_periodo["jogadores"] == jogador]
        if linha_periodo.empty:
            return None
        return int(linha_periodo["rank"].iloc[0])

    if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
        ultimo_registro = df.index.max()
        for chave, dias in {"30": 30, "60": 60, "90": 90}.items():
            limite = ultimo_registro - pd.Timedelta(days=dias)
            df_periodo = df[df.index >= limite]
            rankings_periodo[chave] = _calcular_ranking_periodo(df_periodo)

    assiduidade_texto = "—"
    assiduidade_percentual = None
    media_partidas_por_dia = None
    media_partidas_por_dia_texto = "—"
    if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
        if not jogos_por_dia.empty:
            jogos_por_data = jogos_por_dia.groupby(jogos_por_dia.index.normalize()).sum()
            dias_com_participacao = jogos_por_data[jogos_por_data > 0]
            if not dias_com_participacao.empty:
                primeira_data = dias_com_participacao.index.min()
                jogos_considerados = jogos_por_data[jogos_por_data.index >= primeira_data]
                dias_jogados = int((jogos_considerados > 0).sum())
                dias_registrados = int(len(jogos_considerados))
                if dias_registrados > 0:
                    assiduidade_percentual = dias_jogados / dias_registrados * 100
                    assiduidade_texto = f"{assiduidade_percentual:.1f}% ({dias_jogados}/{dias_registrados} dias)"
                if dias_jogados > 0:
                    total_partidas_consideradas = float(jogos_considerados.sum())
                    media_partidas_por_dia = total_partidas_consideradas / dias_jogados

    rotatividade_texto = "—"
    rotatividade_media = None
    if total_jogos > 0:
        base_partidas = df.loc[:, ["winner1", "winner2", "loser1", "loser2"]]
        if isinstance(base_partidas.index, pd.DatetimeIndex):
            datas_partidas = base_partidas.index.normalize()
        else:
            datas_partidas = pd.to_datetime(base_partidas.index, errors="coerce").normalize()

        parceiros_por_dia: DefaultDict[pd.Timestamp, set] = defaultdict(set)
        jogos_por_dia_map: DefaultDict[pd.Timestamp, int] = defaultdict(int)

        for valores, data_partida in zip(base_partidas.values, datas_partidas):
            if pd.isna(data_partida):
                continue

            duplas = [valores[0:2], valores[2:4]]
            for dupla in duplas:
                jogadores_validos = [j for j in dupla if pd.notna(j)]
                if jogador not in jogadores_validos:
                    continue

                jogos_por_dia_map[data_partida] += 1
                parceiros_validos = [j for j in jogadores_validos if j != jogador]
                parceiros_por_dia[data_partida].update(parceiros_validos)

        indicadores_diarios = []
        for dia, jogos_dia in jogos_por_dia_map.items():
            if jogos_dia <= 0:
                continue
            parceiros_dia = parceiros_por_dia.get(dia, set())
            indicadores_diarios.append(len(parceiros_dia) / jogos_dia)

        if indicadores_diarios:
            rotatividade_media = float(sum(indicadores_diarios) / len(indicadores_diarios))

    parceiro_frequente = None
    parceiro_entrosado = None
    if parcerias_detalhes:
        parcerias_validas = {
            nome: stats for nome, stats in parcerias_detalhes.items() if stats.get("jogos", 0) > 0
        }

        if parcerias_validas:
            parceiro_frequente_nome, parceiro_frequente_stats = max(
                parcerias_validas.items(), key=lambda item: (item[1]["jogos"], item[1]["vitorias"])
            )
            parceiro_frequente = {
                "nome": parceiro_frequente_nome,
                "jogos": parceiro_frequente_stats["jogos"],
            }

            parcerias_entrosamento = {
                nome: stats
                for nome, stats in parcerias_validas.items()
                if stats.get("jogos", 0) >= max(1, config.min_duo_matches)
            }

            if parcerias_entrosamento:
                parceiro_entrosado_nome, parceiro_entrosado_stats = max(
                    parcerias_entrosamento.items(),
                    key=lambda item: (
                        item[1]["vitorias"] / item[1]["jogos"],
                        item[1]["jogos"],
                    ),
                )
                parceiro_entrosado = {
                    "nome": parceiro_entrosado_nome,
                    "jogos": parceiro_entrosado_stats["jogos"],
                    "score": (parceiro_entrosado_stats["vitorias"] / parceiro_entrosado_stats["jogos"]) * 100,
                }

    ranking_parceiro_percentil = None
    ranking_parceiro_estrelas = "—"
    ranking_parceiro_percentil_texto = "—"
    ranking_parceiro_medio_texto = "—"
    if not tabela_geral.empty and "rank" in tabela_geral.columns:
        ranking_map = dict(zip(tabela_geral["jogadores"], tabela_geral["rank"]))
        jogadores_validos = set(ranking_map.keys())
        parceiros_por_jogador: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for row in df.itertuples():
            duplas = ([row.winner1, row.winner2], [row.loser1, row.loser2])
            for dupla in duplas:
                jogadores_dupla = [j for j in dupla if j in jogadores_validos]
                if len(jogadores_dupla) != 2:
                    continue
                jogador_a, jogador_b = jogadores_dupla
                parceiros_por_jogador[jogador_a][jogador_b] += 1
                parceiros_por_jogador[jogador_b][jogador_a] += 1

        ranking_medio_parceiros: Dict[str, float] = {}
        for jogador_base, parceiros in parceiros_por_jogador.items():
            soma_ponderada = 0.0
            total_jogos_rank = 0
            for parceiro, jogos_parceiro in parceiros.items():
                if jogos_parceiro <= 0:
                    continue
                ranking_parceiro = ranking_map.get(parceiro)
                if ranking_parceiro is None or pd.isna(ranking_parceiro):
                    continue
                soma_ponderada += float(ranking_parceiro) * jogos_parceiro
                total_jogos_rank += jogos_parceiro
            if total_jogos_rank > 0:
                ranking_medio_parceiros[jogador_base] = soma_ponderada / total_jogos_rank

        ranking_parceiro_medio = ranking_medio_parceiros.get(jogador)
        if ranking_parceiro_medio is not None:
            ranking_parceiro_medio_texto = f"{ranking_parceiro_medio:.1f}"
            valores_rank = pd.Series(ranking_medio_parceiros.values(), dtype=float).dropna()
            if not valores_rank.empty:
                maiores_ou_iguais = (valores_rank >= ranking_parceiro_medio - 1e-9).sum()
                ranking_parceiro_percentil = maiores_ou_iguais / len(valores_rank) * 100
                ranking_parceiro_estrelas = _formatar_estrelas(ranking_parceiro_percentil)
                ranking_parceiro_percentil_texto = f"Percentil {ranking_parceiro_percentil:.1f}"

    rotatividade_percentil = None
    if rotatividade_media is not None and not linha_geral.empty:
        rotatividade_base = float(linha_geral["rotatividade"].iloc[0])
        rotatividade_amostra = (
            tabela_geral["rotatividade"].astype(float).dropna()
            if "rotatividade" in tabela_geral
            else pd.Series(dtype=float)
        )
        if not rotatividade_amostra.empty:
            menores_ou_iguais = (rotatividade_amostra <= rotatividade_base + 1e-9).sum()
            rotatividade_percentil = menores_ou_iguais / len(rotatividade_amostra) * 100

    if media_partidas_por_dia is not None:
        media_partidas_por_dia_texto = f"{media_partidas_por_dia:.2f} jogos/dia"

    if rotatividade_media is not None:
        rotatividade_texto = _formatar_estrelas(rotatividade_percentil)

    maior_pato = None
    maior_carrasco = None
    if confrontos_detalhes:
        for adversario, dados in confrontos_detalhes.items():
            jogos_totais = dados.get("vitorias", 0) + dados.get("derrotas", 0)
            if jogos_totais <= 0:
                continue
            saldo_contra = dados.get("vitorias", 0) - dados.get("derrotas", 0)
            registro = {
                "nome": adversario,
                "vitorias": dados.get("vitorias", 0),
                "derrotas": dados.get("derrotas", 0),
                "saldo": saldo_contra,
            }
            if saldo_contra > 0:
                if not maior_pato or saldo_contra > maior_pato["saldo"]:
                    maior_pato = registro
            elif saldo_contra < 0:
                if not maior_carrasco or saldo_contra < maior_carrasco["saldo"]:
                    maior_carrasco = registro

    tendencia, tendencia_delta, tendencia_observacao = _calcular_tendencia(score_diario, config)

    return {
        "medalha": medalha,
        "jogador": jogador,
        "jogos": total_jogos,
        "vitorias": total_vitorias,
        "derrotas": total_derrotas,
        "saldo": saldo,
        "ranking_geral": ranking_geral,
        "rankings_periodo": rankings_periodo,
        "assiduidade_texto": assiduidade_texto,
        "assiduidade_percentual": assiduidade_percentual,
        "rotatividade_texto": rotatividade_texto,
        "rotatividade_percentil": rotatividade_percentil,
        "media_partidas_por_dia": media_partidas_por_dia,
        "media_partidas_por_dia_texto": media_partidas_por_dia_texto,
        "score": total_score,
        "parceiro_frequente": parceiro_frequente,
        "parceiro_entrosado": parceiro_entrosado,
        "min_jogos_entrosamento": config.min_duo_matches,
        "ranking_parceiro_percentil": ranking_parceiro_percentil,
        "ranking_parceiro_estrelas": ranking_parceiro_estrelas,
        "ranking_parceiro_percentil_texto": ranking_parceiro_percentil_texto,
        "ranking_parceiro_medio_texto": ranking_parceiro_medio_texto,
        "maior_pato": maior_pato,
        "maior_carrasco": maior_carrasco,
        "tendencia": tendencia,
        "tendencia_delta": tendencia_delta,
        "tendencia_observacao": tendencia_observacao,
    }


def _contar_jogos_confrontos_jogadores(df: pd.DataFrame) -> pd.DataFrame:
    jogadores = sorted(
        set(df["winner1"].tolist() + df["winner2"].tolist() + df["loser1"].tolist() + df["loser2"].tolist())
    )
    jogos = pd.DataFrame(0, index=jogadores, columns=jogadores)

    for _, row in df.iterrows():
        winners = [row["winner1"], row["winner2"]]
        losers = [row["loser1"], row["loser2"]]
        for winner in winners:
            for loser in losers:
                jogos.at[winner, loser] += 1
                jogos.at[loser, winner] += 1

    jogos = jogos.loc[
        ~jogos.index.astype(str).str.contains("Outro"),
        ~jogos.columns.astype(str).str.contains("Outro"),
    ]
    return jogos


def _contar_jogos_confrontos_duplas(df: pd.DataFrame) -> pd.DataFrame:
    duplas = sorted(set(df["dupla_winner"].tolist() + df["dupla_loser"].tolist()))
    jogos = pd.DataFrame(0, index=duplas, columns=duplas)

    for _, row in df.iterrows():
        winner_dupla = row["dupla_winner"]
        loser_dupla = row["dupla_loser"]
        jogos.at[winner_dupla, loser_dupla] += 1
        jogos.at[loser_dupla, winner_dupla] += 1

    jogos = jogos.loc[
        ~jogos.index.astype(str).str.contains("Outro"),
        ~jogos.columns.astype(str).str.contains("Outro"),
    ]
    return jogos


def _saldo_confrontos_jogador(df: pd.DataFrame, jogador: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    df_saldo = preparar_dados_confrontos_jogadores(df)
    df_jogos = _contar_jogos_confrontos_jogadores(df)
    if jogador not in df_saldo.index:
        return [], []

    saldo_jogador = df_saldo.loc[jogador, :]
    jogos_jogador = df_jogos.loc[jogador, :] if jogador in df_jogos.index else pd.Series(dtype=int)
    fregueses = saldo_jogador[saldo_jogador > 0].sort_values(ascending=False)
    fregueses = (
        fregueses.to_frame("saldo")
        .join(jogos_jogador.rename("jogos"))
        .reset_index()
        .rename(columns={"index": "nome"})
    )
    carrascos = saldo_jogador[saldo_jogador < 0].sort_values()
    carrascos = (
        carrascos.to_frame("saldo")
        .join(jogos_jogador.rename("jogos"))
        .reset_index()
        .rename(columns={"index": "nome"})
    )
    return fregueses.to_dict(orient="records"), carrascos.to_dict(orient="records")


def calcular_metricas_dupla(df: pd.DataFrame, jogador1: str, jogador2: str) -> Dict[str, object]:
    dupla_selecionada = " e ".join(sorted([jogador1, jogador2]))

    dupla_vitorias = (df["dupla_winner"] == dupla_selecionada).astype(int)
    dupla_derrotas = (df["dupla_loser"] == dupla_selecionada).astype(int)

    vitorias_por_dia = dupla_vitorias.groupby(df.index).sum()
    derrotas_por_dia = dupla_derrotas.groupby(df.index).sum()

    jogos_totais = vitorias_por_dia + derrotas_por_dia
    score_percentual = (vitorias_por_dia / jogos_totais * 100).dropna().sort_index()
    score_por_dia = score_percentual.round(2)
    total_jogos = int(vitorias_por_dia.sum() + derrotas_por_dia.sum())
    total_vitorias = int(vitorias_por_dia.sum())
    total_derrotas = int(derrotas_por_dia.sum())
    total_score = (total_vitorias / total_jogos * 100) if total_jogos else 0

    colunas_participacao = ["winner1", "winner2", "loser1", "loser2"]
    jogador1_present = df[colunas_participacao].eq(jogador1).any(axis=1)
    jogador2_present = df[colunas_participacao].eq(jogador2).any(axis=1)

    presenca_j1_por_dia = jogador1_present.groupby(df.index).any()
    presenca_j2_por_dia = jogador2_present.groupby(df.index).any()
    dias_juntos = presenca_j1_por_dia & presenca_j2_por_dia
    total_dias_juntos = int(dias_juntos.sum())

    dupla_jogou_junta = (df["dupla_winner"] == dupla_selecionada) | (df["dupla_loser"] == dupla_selecionada)
    dias_dupla_jogou_junto = dupla_jogou_junta.groupby(df.index).any()
    total_dias_dupla = int(dias_dupla_jogou_junto.sum())

    percentual_dias_jogando_juntos = (total_dias_dupla / total_dias_juntos * 100) if total_dias_juntos else 0

    ultima_data_juntos = df.index[dupla_jogou_junta].max() if dupla_jogou_junta.any() else pd.NaT
    ultima_data_formatada = (
        ultima_data_juntos.strftime("%d/%m/%Y") if pd.notna(ultima_data_juntos) else "Nunca jogaram juntos"
    )

    config = get_config()
    tendencia_dupla, _, tendencia_dupla_observacao = _calcular_tendencia(score_por_dia, config)

    df_saldo_duplas = preparar_dados_confrontos_duplas(df)
    df_jogos_duplas = _contar_jogos_confrontos_duplas(df)
    saldo_dupla = (
        df_saldo_duplas.loc[dupla_selecionada, :] if dupla_selecionada in df_saldo_duplas.index else pd.Series(dtype=float)
    )
    jogos_dupla = (
        df_jogos_duplas.loc[dupla_selecionada, :] if dupla_selecionada in df_jogos_duplas.index else pd.Series(dtype=int)
    )

    maior_fregues = None
    if not saldo_dupla.empty:
        saldo_positivo = saldo_dupla[saldo_dupla > 0].sort_values(ascending=False)
        if not saldo_positivo.empty:
            maior_fregues = {"nome": saldo_positivo.index[0], "saldo": int(saldo_positivo.iloc[0])}

    maior_carrasco = None
    if not saldo_dupla.empty:
        saldo_negativo = saldo_dupla[saldo_dupla < 0].sort_values()
        if not saldo_negativo.empty:
            maior_carrasco = {"nome": saldo_negativo.index[0], "saldo": int(saldo_negativo.iloc[0])}

    dias_juntos_texto = (
        f"{total_dias_juntos} dia{'s' if total_dias_juntos != 1 else ''}" if total_dias_juntos else "—"
    )
    dias_dupla_texto = (
        f"{total_dias_dupla} dia{'s' if total_dias_dupla != 1 else ''}" if total_dias_dupla else "—"
    )

    if total_dias_juntos:
        percentual_dias_texto = f"{percentual_dias_jogando_juntos:.2f}% ({total_dias_dupla}/{total_dias_juntos} dias)"
        percentual_dias_valor = percentual_dias_jogando_juntos
    else:
        percentual_dias_texto = "—"
        percentual_dias_valor = None

    metricas_dupla = {
        "dupla": dupla_selecionada,
        "jogos": int(total_jogos),
        "vitorias": int(total_vitorias),
        "derrotas": int(total_derrotas),
        "saldo": int(total_vitorias - total_derrotas),
        "score": float(total_score),
        "ultimo_jogo": ultima_data_formatada,
        "dias_juntos_texto": dias_juntos_texto,
        "dias_dupla_texto": dias_dupla_texto,
        "percentual_dias_texto": percentual_dias_texto,
        "percentual_dias_valor": percentual_dias_valor,
        "maior_fregues": maior_fregues,
        "maior_carrasco": maior_carrasco,
        "tendencia": tendencia_dupla,
        "tendencia_observacao": tendencia_dupla_observacao,
    }

    fregueses = saldo_dupla[saldo_dupla > 0].sort_values(ascending=False)
    fregueses = (
        fregueses.to_frame("saldo")
        .join(jogos_dupla.rename("jogos"))
        .reset_index()
        .rename(columns={"index": "nome"})
    )
    carrascos = saldo_dupla[saldo_dupla < 0].sort_values()
    carrascos = (
        carrascos.to_frame("saldo")
        .join(jogos_dupla.rename("jogos"))
        .reset_index()
        .rename(columns={"index": "nome"})
    )

    serie_score = _agrupar_score_mensal(score_percentual)
    jogos_recentes = _montar_jogos_recentes_dupla(df, jogador1, jogador2, limite=30)

    return {
        "metricas": metricas_dupla,
        "fregueses": fregueses.to_dict(orient="records"),
        "carrascos": carrascos.to_dict(orient="records"),
        "score_series": serie_score,
        "jogos_recentes": jogos_recentes,
    }
