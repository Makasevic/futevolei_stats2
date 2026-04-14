import os
import unittest

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon-key")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service-key")

from championship import service
from src.redinha_stats.infrastructure.supabase import championship_repository


class ChampionshipRepositoryTests(unittest.TestCase):
    def test_db_championship_key_normalizes_six_digit_values(self) -> None:
        self.assertEqual(championship_repository._db_championship_key("202602"), "20260201")
        self.assertEqual(championship_repository._db_championship_key(" 2026-02 "), "20260201")
        self.assertEqual(championship_repository._db_championship_key("custom-key"), "custom-key")


class ChampionshipServiceTests(unittest.TestCase):
    def test_parse_score_accepts_blank_and_integer_values(self) -> None:
        self.assertIsNone(service._parse_score(""))
        self.assertEqual(service._parse_score("12"), 12)

    def test_parse_score_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "inteiros"):
            service._parse_score("abc")
        with self.assertRaisesRegex(ValueError, "negativo"):
            service._parse_score("-1")

    def test_winner_and_loser_require_played_non_draw_match(self) -> None:
        match = {"team_a": "A", "team_b": "B", "score_a": 21, "score_b": 18}
        draw = {"team_a": "A", "team_b": "B", "score_a": 20, "score_b": 20}

        self.assertEqual(service._winner(match), "A")
        self.assertEqual(service._loser(match), "B")
        self.assertIsNone(service._winner(draw))
        self.assertIsNone(service._loser(draw))

    def test_accumulate_stats_computes_team_totals(self) -> None:
        matches = [
            {"team_a": "A", "team_b": "B", "score_a": 21, "score_b": 18},
            {"team_a": "B", "team_b": "C", "score_a": 21, "score_b": 19},
        ]

        stats = service._accumulate_stats(matches)

        self.assertEqual(stats["A"]["vitorias"], 1)
        self.assertEqual(stats["A"]["saldo"], 3)
        self.assertEqual(stats["B"]["jogos"], 2)
        self.assertEqual(stats["B"]["derrotas"], 1)
        self.assertEqual(stats["C"]["derrotas"], 1)

    def test_head_to_head_uses_wins_then_point_difference(self) -> None:
        matches = [
            {"team_a": "A", "team_b": "B", "score_a": 21, "score_b": 15},
            {"team_a": "B", "team_b": "A", "score_a": 21, "score_b": 19},
        ]

        self.assertEqual(service._head_to_head("A", "B", matches), 4)
        self.assertEqual(service._head_to_head("B", "A", matches), -4)


if __name__ == "__main__":
    unittest.main()
