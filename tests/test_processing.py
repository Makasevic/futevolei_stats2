import unittest
from unittest.mock import patch

import pandas as pd

from preparation import preparar_dataframe
from processing import filtrar_dados, preparar_dados_duplas, preparar_dados_individuais


def _sample_matches() -> pd.DataFrame:
    return preparar_dataframe(
        [
            {
                "winner1": "Ana",
                "winner2": "Bia",
                "loser1": "Caio",
                "loser2": "Duda",
                "date": "2026-01-10",
            },
            {
                "winner1": "Ana",
                "winner2": "Caio",
                "loser1": "Bia",
                "loser2": "Duda",
                "date": "2026-02-05",
            },
            {
                "winner1": "Caio",
                "winner2": "Duda",
                "loser1": "Ana",
                "loser2": "Bia",
                "date": "2026-03-01",
            },
        ]
    )


class FiltrarDadosTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _sample_matches()

    def test_filtrar_dados_by_exact_date(self) -> None:
        resultado = filtrar_dados(self.df, "Data", "2026-02-05")

        self.assertEqual(len(resultado), 1)
        self.assertEqual(resultado.index[0], pd.Timestamp("2026-02-05"))

    def test_filtrar_dados_by_mes_ano_accepts_ascii_label(self) -> None:
        resultado = filtrar_dados(self.df, "Mes/Ano", "2026-03")

        self.assertEqual(len(resultado), 1)
        self.assertEqual(resultado.index[0], pd.Timestamp("2026-03-01"))

    def test_filtrar_dados_by_ano(self) -> None:
        resultado = filtrar_dados(self.df, "Ano", "2026")

        self.assertEqual(len(resultado), 3)

    @patch("processing.pd.Timestamp.now", return_value=pd.Timestamp("2026-03-10"))
    def test_filtrar_dados_by_30_days(self, _mock_now) -> None:
        resultado = filtrar_dados(self.df, "Dias", "30 dias")

        self.assertEqual(len(resultado), 1)
        self.assertEqual(resultado.index[0], pd.Timestamp("2026-03-01"))

    def test_filtrar_dados_by_1_day_uses_latest_match_date(self) -> None:
        resultado = filtrar_dados(self.df, "Dias", "1 dia")

        self.assertEqual(len(resultado), 1)
        self.assertEqual(resultado.index[0], pd.Timestamp("2026-03-01"))


class RankingPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _sample_matches()

    def test_preparar_dados_individuais_calculates_stats_and_rank(self) -> None:
        tabela = preparar_dados_individuais(self.df)
        ana = tabela.loc[tabela["jogadores"] == "Ana"].iloc[0]
        duda = tabela.loc[tabela["jogadores"] == "Duda"].iloc[0]

        self.assertEqual(int(ana["vitórias"]), 2)
        self.assertEqual(int(ana["derrotas"]), 1)
        self.assertEqual(int(ana["jogos"]), 3)
        self.assertEqual(str(ana["score"]), "67%")
        self.assertEqual(int(duda["vitórias"]), 1)
        self.assertEqual(int(duda["derrotas"]), 2)
        self.assertIn("rotatividade", tabela.columns)
        self.assertTrue(tabela["rank"].between(1, len(tabela)).all())

    def test_preparar_dados_duplas_calculates_aggregate_stats(self) -> None:
        tabela = preparar_dados_duplas(self.df)
        dupla = tabela.loc[tabela["duplas"] == "Ana e Bia"].iloc[0]

        self.assertEqual(int(dupla["vitórias"]), 1)
        self.assertEqual(int(dupla["derrotas"]), 1)
        self.assertEqual(int(dupla["jogos"]), 2)
        self.assertEqual(str(dupla["score"]), "50%")


if __name__ == "__main__":
    unittest.main()
