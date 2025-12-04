from datetime import timedelta
from collections import defaultdict
from typing import DefaultDict

import pandas as pd
import numpy as np


# ----------------------------------------------------------------------
# 1) FILTRO ÚNICO  ------------------------------------------------------
# ----------------------------------------------------------------------
def filtrar_dados(df: pd.DataFrame, modo: str, valor: str) -> pd.DataFrame:
    """
    Filtra o DataFrame de acordo com o *modo*:

    • 'Dias' – textos fixos (1 dia, 1 mês, …, Todos)
    • 'Mês/Ano'          – string 'YYYY-MM'
    • 'Ano'              – string 'YYYY'
    """

    if modo == "Data":
        if valor in (None, ""):
            return df.iloc[0:0]

        if not isinstance(df.index, pd.DatetimeIndex):
            return df.iloc[0:0]

        try:
            data_alvo = pd.to_datetime(valor).normalize()
        except (TypeError, ValueError):
            return df.iloc[0:0]

        tzinfo = getattr(df.index, "tz", None)
        if tzinfo is not None:
            try:
                data_alvo = data_alvo.tz_localize(tzinfo)
            except TypeError:
                data_alvo = data_alvo.tz_convert(tzinfo)

        indice_normalizado = df.index.normalize()
        return df[indice_normalizado == data_alvo]

    if modo == "Dias":
        indice = df.index

        if isinstance(indice, pd.DatetimeIndex):
            tzinfo = getattr(indice, "tz", None)
        else:
            tzinfo = None

        if tzinfo is not None:
            hoje = pd.Timestamp.now(tz=tzinfo)
        else:
            hoje = pd.Timestamp.now()

        if valor == "1 dia":
            data_ini = df.index.max()
            return df[df.index >= data_ini]

        elif valor == "30 dias":
            data_ini = hoje - timedelta(days=30)

        elif valor == "60 dias":
            data_ini = hoje - timedelta(days=60)

        elif valor == "90 dias":
            data_ini = hoje - timedelta(days=90)

        elif valor == "180 dias":
            data_ini = hoje - timedelta(days=180)

        elif valor == "360 dias":
            data_ini = hoje - timedelta(days=360)

        else:  # "Todos"
            return df

        return df[df.index >= data_ini]

    elif modo == "Mês/Ano":
        periodo = pd.Period(valor, freq="M")
        return df[df.index.to_period("M") == periodo]

    else:  # 'Ano'
        ano_int = int(valor)
        return df[df.index.year == ano_int]


# ----------------------------------------------------------------------
# 2) FUNÇÕES DE PREPARAÇÃO DE DADOS ------------------------------------
# ----------------------------------------------------------------------
def preparar_dados_individuais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula vitórias, derrotas, score (%) e rank para cada jogador.
    Espera que df tenha 4 colunas com nomes qualquer, onde:
        df.iloc[:, 0:2] → jogadores vencedores
        df.iloc[:, 2:4] → jogadores derrotados
    """
    vitorias = pd.Series(df.iloc[:, 0:2].values.ravel()).value_counts()
    derrotas = pd.Series(df.iloc[:, 2:4].values.ravel()).value_counts()

    jogadores = sorted(set(vitorias.index) | set(derrotas.index))
    vitorias = vitorias.reindex(jogadores, fill_value=0)
    derrotas = derrotas.reindex(jogadores, fill_value=0)

    totais = vitorias + derrotas
    aproveitamento = (vitorias / totais * 100).fillna(0)
    saldo = (vitorias - derrotas).astype(int)

    parceiros_diarios: DefaultDict[str, DefaultDict[pd.Timestamp, set]] = defaultdict(
        lambda: defaultdict(set)
    )
    jogos_diarios: DefaultDict[str, DefaultDict[pd.Timestamp, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    base_partidas = df.iloc[:, 0:4]
    if isinstance(base_partidas.index, pd.DatetimeIndex):
        datas_partidas = base_partidas.index.normalize()
    else:
        datas_partidas = pd.to_datetime(base_partidas.index, errors="coerce").normalize()

    for valores, data_partida in zip(base_partidas.values, datas_partidas):
        if pd.isna(data_partida):
            continue

        duplas = [valores[0:2], valores[2:4]]
        for dupla in duplas:
            jogadores_validos = [j for j in dupla if pd.notna(j)]
            if len(jogadores_validos) != 2:
                continue

            jogador_a, jogador_b = jogadores_validos

            jogos_diarios[jogador_a][data_partida] += 1
            jogos_diarios[jogador_b][data_partida] += 1
            parceiros_diarios[jogador_a][data_partida].add(jogador_b)
            parceiros_diarios[jogador_b][data_partida].add(jogador_a)

    rotatividade_por_jogador = {}
    for jogador in jogadores:
        dias_jogados = jogos_diarios.get(jogador, {})
        if not dias_jogados:
            rotatividade_por_jogador[jogador] = 0.0
            continue

        indicadores_diarios = []
        parceiros_do_jogador = parceiros_diarios.get(jogador, {})
        for dia, jogos_dia in dias_jogados.items():
            if jogos_dia <= 0:
                continue
            parceiros_dia = parceiros_do_jogador.get(dia, set())
            indicadores_diarios.append(len(parceiros_dia) / jogos_dia)

        rotatividade_por_jogador[jogador] = (
            float(sum(indicadores_diarios) / len(indicadores_diarios))
            if indicadores_diarios
            else 0.0
        )

    rotatividade_media_series = pd.Series(rotatividade_por_jogador, dtype="float64")
    rotatividade_percentil_series = pd.Series(dtype="float64")

    if not rotatividade_media_series.empty:
        distribuicao = rotatividade_media_series.rank(method="max", pct=True)
        menor_percentil = float(distribuicao.min()) if not distribuicao.empty else 0.0

        if np.isclose(menor_percentil, 1.0):
            rotatividade_percentil_series = pd.Series(
                100.0, index=rotatividade_media_series.index, dtype="float64"
            )
        else:
            rotatividade_percentil_series = (
                (distribuicao - menor_percentil) / (1 - menor_percentil)
            ).clip(lower=0.0, upper=1.0) * 100

    tabela = pd.concat([vitorias, derrotas, totais, saldo, aproveitamento], axis=1).reset_index()
    tabela.columns = ["jogadores", "vitórias", "derrotas", "jogos", "saldo", "aproveitamento"]

    tabela["vitórias"] = tabela["vitórias"].astype(int)
    tabela["derrotas"] = tabela["derrotas"].astype(int)
    tabela["jogos"] = tabela["jogos"].astype(int)

    tabela = tabela.sort_values(
        ["aproveitamento", "saldo", "vitórias"], ascending=[False, False, False]
    ).reset_index(drop=True)

    tabela["rank"] = tabela.index + 1
    tabela["score"] = tabela["aproveitamento"].round().astype(int).astype(str) + "%"
    tabela["rotatividade_media"] = (
        tabela["jogadores"].map(rotatividade_media_series).fillna(0).round(3)
    )
    tabela["rotatividade_percentil"] = (
        tabela["jogadores"].map(rotatividade_percentil_series).fillna(0).round(0)
    )
    tabela["rotatividade"] = tabela["rotatividade_percentil"]

    return tabela


# ----------------------------------------------------------------------
# 3) FUNÇÕES DE CONFRONTO E ESTATÍSTICAS -------------------------------
# ----------------------------------------------------------------------

def preparar_dados_duplas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara dados de vitórias, derrotas e score para duplas.
    """
    duplas_v = pd.Series([f"{a} e {b}" for a, b in df.iloc[:, 0:2].values]).value_counts()
    duplas_d = pd.Series([f"{a} e {b}" for a, b in df.iloc[:, 2:4].values]).value_counts()

    duplas = sorted(set(duplas_v.index) | set(duplas_d.index))
    duplas_v = duplas_v.reindex(duplas, fill_value=0)
    duplas_d = duplas_d.reindex(duplas, fill_value=0)

    totais = duplas_v + duplas_d
    aproveitamento = (duplas_v / totais * 100).fillna(0)
    saldo = (duplas_v - duplas_d).astype(int)

    tabela = pd.concat([duplas_v, duplas_d, totais, saldo, aproveitamento], axis=1).reset_index()

    tabela.columns = ["duplas", "vitórias", "derrotas", "jogos", "saldo", "aproveitamento"]

    tabela["vitórias"] = tabela["vitórias"].astype(int)
    tabela["derrotas"] = tabela["derrotas"].astype(int)
    tabela["jogos"] = tabela["jogos"].astype(int)

    tabela = tabela.sort_values(
        ["aproveitamento", "saldo", "vitórias"], ascending=[False, False, False]
    ).reset_index(drop=True)

    tabela["rank"] = tabela.index + 1
    tabela["score"] = tabela["aproveitamento"].round().astype(int).astype(str) + "%"

    return tabela


def preparar_dados_confrontos_jogadores(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna DataFrame com saldo de confrontos entre jogadores (linha vs coluna)."""

    jogadores = sorted(
        set(df["winner1"].tolist() + df["winner2"].tolist() + df["loser1"].tolist() + df["loser2"].tolist())
    )
    saldos = pd.DataFrame(0, index=jogadores, columns=jogadores)

    for _, row in df.iterrows():
        winners = [row["winner1"], row["winner2"]]
        losers = [row["loser1"], row["loser2"]]
        for winner in winners:
            for loser in losers:
                saldos.at[winner, loser] += 1
                saldos.at[loser, winner] -= 1

    saldo_final = saldos.reset_index().rename(columns={"index": "Jogador"}).set_index("Jogador")
    saldo_final = saldo_final.loc[
        ~saldo_final.index.astype(str).str.contains("Outro"),
        ~saldo_final.columns.astype(str).str.contains("Outro"),
    ]
    return saldo_final


def preparar_dados_confrontos_duplas(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna DataFrame com saldo de confrontos entre duplas (linha vs coluna)."""

    duplas = sorted(set(df["dupla_winner"].tolist() + df["dupla_loser"].tolist()))
    saldos_duplas = pd.DataFrame(0, index=duplas, columns=duplas)

    for _, row in df.iterrows():
        winner_dupla = row["dupla_winner"]
        loser_dupla = row["dupla_loser"]
        saldos_duplas.at[winner_dupla, loser_dupla] += 1
        saldos_duplas.at[loser_dupla, winner_dupla] -= 1

    saldo_final_duplas = saldos_duplas.reset_index().rename(columns={"index": "Dupla"}).set_index("Dupla")
    saldo_final_duplas = saldo_final_duplas.loc[
        ~saldo_final_duplas.index.astype(str).str.contains("Outro"),
        ~saldo_final_duplas.columns.astype(str).str.contains("Outro"),
    ]
    return saldo_final_duplas
