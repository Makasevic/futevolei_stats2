import pandas as pd
import numpy as np


def criar_colunas_duplas(df):
    """
    Cria colunas para duplas vencedoras e perdedoras, garantindo consistência de ordenação.

    Args:
        df (DataFrame): DataFrame com colunas 'winner1', 'winner2', 'loser1', 'loser2'.

    Returns:
        DataFrame: DataFrame atualizado com colunas de duplas ordenadas.
    """
    # Criar as colunas de duplas vencedoras e perdedoras
    df["dupla_winner"] = df.apply(lambda row: " e ".join(sorted([row["winner1"], row["winner2"]])), axis=1)
    df["dupla_loser"] = df.apply(lambda row: " e ".join(sorted([row["loser1"], row["loser2"]])), axis=1)

    # Ordenar vencedores e perdedores nas colunas individuais
    df[["winner1", "winner2"]] = np.sort(df[["winner1", "winner2"]], axis=1)
    df[["loser1", "loser2"]] = np.sort(df[["loser1", "loser2"]], axis=1)

    return df


def preparar_dataframe(matches):
    """Converte a lista de partidas do Supabase em um ``DataFrame`` pronto para uso."""

    base_columns = ["winner1", "winner2", "loser1", "loser2", "date"]
    df = pd.DataFrame(matches or [])

    id_columns = ["id", "match_id"]

    for coluna in ["winner1", "winner2", "loser1", "loser2"]:
        if coluna not in df:
            df[coluna] = ""
        df[coluna] = df[coluna].fillna("").astype(str).str.strip()

    if "date" not in df:
        df["date"] = pd.NaT

    for coluna in id_columns:
        if coluna not in df:
            df[coluna] = None

    ordered_columns = base_columns + id_columns + [
        col for col in df.columns if col not in base_columns + id_columns
    ]
    df = df[ordered_columns]

    df = df.set_index("date")

    # Adicionar colunas de duplas e garantir consistência
    df = criar_colunas_duplas(df)

    df.index = pd.to_datetime(df.index, errors="coerce")

    return df
