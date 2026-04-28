from __future__ import annotations

import pandas as pd

import detalhamento


def test_calcular_metricas_gerais_groups_partner_quality_by_star_rating(monkeypatch):
    df = pd.DataFrame(
        [
            {"winner1": "Ana", "winner2": "Bia", "loser1": "Caio", "loser2": "Duda"},
        ]
    )
    tabela = pd.DataFrame(
        [
            {"jogadores": "Ana", "rank": 1},
            {"jogadores": "Bia", "rank": 2},
            {"jogadores": "Caio", "rank": 3},
            {"jogadores": "Duda", "rank": 4},
            {"jogadores": "Eva", "rank": 5},
            {"jogadores": "Fabi", "rank": 6},
        ]
    )
    monkeypatch.setattr(detalhamento, "preparar_dados_individuais", lambda df_base: tabela)
    monkeypatch.setattr(detalhamento, "_aplicar_regras_ranking", lambda tabela_geral, config: tabela_geral)
    monkeypatch.setattr(
        detalhamento,
        "_calcular_ranking_medio_parceiros",
        lambda df_base, tabela_geral: {
            "Ana": 1.0,
            "Bia": 2.0,
            "Caio": 3.0,
            "Duda": 4.0,
            "Eva": 5.0,
            "Fabi": 6.0,
        },
    )

    resultado = detalhamento.calcular_metricas_gerais(df)

    colunas = resultado["qualidade_parceiros_colunas"]
    assert [coluna["rating"] for coluna in colunas] == [5, 4, 3, 2, 1]
    assert colunas[0]["jogadores"][0]["nome"] == "Ana"
    assert colunas[-1]["jogadores"][0]["nome"] == "Fabi"
    tamanhos = [len(coluna["jogadores"]) for coluna in colunas]
    assert max(tamanhos) - min(tamanhos) <= 1


def test_calcular_metricas_gerais_groups_opponent_quality_by_star_rating(monkeypatch):
    df = pd.DataFrame(
        [{"winner1": "Ana", "winner2": "Bia", "loser1": "Caio", "loser2": "Duda"}] * 30
        + [{"winner1": "Ana", "winner2": "Caio", "loser1": "Eva", "loser2": "Fabi"}] * 30
    )
    tabela = pd.DataFrame(
        [
            {"jogadores": "Ana", "rank": 1},
            {"jogadores": "Bia", "rank": 2},
            {"jogadores": "Caio", "rank": 3},
            {"jogadores": "Duda", "rank": 4},
            {"jogadores": "Eva", "rank": 5},
            {"jogadores": "Fabi", "rank": 6},
        ]
    )
    monkeypatch.setattr(detalhamento, "preparar_dados_individuais", lambda df_base: tabela)
    monkeypatch.setattr(detalhamento, "_aplicar_regras_ranking", lambda tabela_geral, config: tabela_geral)
    monkeypatch.setattr(detalhamento, "_calcular_ranking_medio_parceiros", lambda df_base, tabela_geral: {})

    resultado = detalhamento.calcular_metricas_gerais(df)

    colunas = resultado["qualidade_adversarios_colunas"]
    assert [coluna["rating"] for coluna in colunas] == [5, 4, 3, 2, 1]
    assert colunas[0]["jogadores"][0]["nome"] == "Duda"
    assert colunas[-1]["jogadores"][0]["nome"] == "Ana"


def test_partner_and_opponent_quality_ignore_matches_with_outro():
    df = pd.DataFrame(
        [{"winner1": "Ana", "winner2": "Bia", "loser1": "Caio", "loser2": "Duda"}] * 30
        + [{"winner1": "Ana", "winner2": "Outro_1", "loser1": "Caio", "loser2": "Duda"}] * 30
    )
    tabela = pd.DataFrame(
        [
            {"jogadores": "Ana", "rank": 1},
            {"jogadores": "Bia", "rank": 2},
            {"jogadores": "Caio", "rank": 3},
            {"jogadores": "Duda", "rank": 4},
        ]
    )

    parceiros = detalhamento._calcular_ranking_medio_parceiros(df, tabela)
    adversarios = detalhamento._calcular_ranking_medio_adversarios(df, tabela)

    assert parceiros["Ana"] == 2.0
    assert adversarios["Ana"] == 3.5


def test_opponent_quality_averages_opponent_pair_rank_by_match():
    df = pd.DataFrame(
        [{"winner1": "Ana", "winner2": "Bia", "loser1": "Caio", "loser2": "Duda"}] * 15
        + [{"winner1": "Ana", "winner2": "Bia", "loser1": "Eva", "loser2": "Fabi"}] * 15
    )
    tabela = pd.DataFrame(
        [
            {"jogadores": "Ana", "rank": 1},
            {"jogadores": "Bia", "rank": 2},
            {"jogadores": "Caio", "rank": 3},
            {"jogadores": "Duda", "rank": 5},
            {"jogadores": "Eva", "rank": 7},
            {"jogadores": "Fabi", "rank": 9},
        ]
    )

    adversarios = detalhamento._calcular_ranking_medio_adversarios(df, tabela)

    assert adversarios["Ana"] == 6.0
    assert adversarios["Bia"] == 6.0
    assert adversarios["Caio"] == 1.5
    assert adversarios["Eva"] == 1.5


def test_opponent_quality_uses_players_with_fewer_than_30_valid_matches():
    df = pd.DataFrame(
        [{"winner1": "Ana", "winner2": "Bia", "loser1": "Caio", "loser2": "Duda"}] * 2
        + [{"winner1": "Ana", "winner2": "Outro_1", "loser1": "Caio", "loser2": "Duda"}]
    )
    tabela = pd.DataFrame(
        [
            {"jogadores": "Ana", "rank": 1},
            {"jogadores": "Bia", "rank": 2},
            {"jogadores": "Caio", "rank": 3},
            {"jogadores": "Duda", "rank": 4},
        ]
    )

    adversarios = detalhamento._calcular_ranking_medio_adversarios(df, tabela)

    assert adversarios["Ana"] == 3.5


def test_montar_series_score_returns_monthly_and_daily_series():
    score = pd.Series(
        [100.0, 0.0, 50.0],
        index=pd.to_datetime(["2026-01-01", "2026-01-01", "2026-02-01"]),
    )

    series = detalhamento._montar_series_score(score)

    assert set(series) == {"mes", "dia"}
    assert [item["data"] for item in series["mes"]] == ["01/2026", "02/2026"]
    assert [item["score"] for item in series["mes"]] == [50.0, 50.0]
    assert [item["data"] for item in series["dia"]] == ["01/01/2026", "01/02/2026"]
    assert [item["score"] for item in series["dia"]] == [50.0, 50.0]
