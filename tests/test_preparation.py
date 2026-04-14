import unittest

import pandas as pd

from src.redinha_stats.domain.matches.preparation import (
    criar_colunas_duplas,
    preparar_dataframe,
)


class PreparationTests(unittest.TestCase):
    def test_criar_colunas_duplas_orders_players_and_builds_pair_columns(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "winner1": "Bruno",
                    "winner2": "Ana",
                    "loser1": "Carlos",
                    "loser2": "Beto",
                }
            ]
        )

        resultado = criar_colunas_duplas(df.copy())

        self.assertEqual(resultado.loc[0, "winner1"], "Ana")
        self.assertEqual(resultado.loc[0, "winner2"], "Bruno")
        self.assertEqual(resultado.loc[0, "loser1"], "Beto")
        self.assertEqual(resultado.loc[0, "loser2"], "Carlos")
        self.assertEqual(resultado.loc[0, "dupla_winner"], "Ana e Bruno")
        self.assertEqual(resultado.loc[0, "dupla_loser"], "Beto e Carlos")

    def test_preparar_dataframe_fills_missing_columns_and_sets_datetime_index(self) -> None:
        matches = [
            {
                "winner1": "  Ana ",
                "winner2": None,
                "loser1": "Carlos",
                "loser2": "Beto",
                "date": "2026-03-01",
                "score": "21-18",
            }
        ]

        resultado = preparar_dataframe(matches)

        self.assertIsInstance(resultado.index, pd.DatetimeIndex)
        self.assertEqual(resultado.index[0], pd.Timestamp("2026-03-01"))
        self.assertEqual(resultado.iloc[0]["winner1"], "")
        self.assertEqual(resultado.iloc[0]["winner2"], "Ana")
        self.assertEqual(resultado.iloc[0]["id"], None)
        self.assertEqual(resultado.iloc[0]["match_id"], None)
        self.assertEqual(resultado.iloc[0]["dupla_winner"], " e Ana")
        self.assertEqual(resultado.iloc[0]["dupla_loser"], "Beto e Carlos")
        self.assertEqual(resultado.iloc[0]["score"], "21-18")

    def test_preparar_dataframe_handles_empty_payload(self) -> None:
        resultado = preparar_dataframe([])

        self.assertListEqual(
            list(resultado.columns[:8]),
            [
                "winner1",
                "winner2",
                "loser1",
                "loser2",
                "id",
                "match_id",
                "dupla_winner",
                "dupla_loser",
            ],
        )
        self.assertEqual(len(resultado), 0)


if __name__ == "__main__":
    unittest.main()
