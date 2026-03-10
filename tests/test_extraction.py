import os
import unittest
from datetime import date, datetime
from unittest.mock import patch

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon-key")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service-key")

import extraction


class ExtractionTests(unittest.TestCase):
    def test_normalize_date_supports_date_types_and_strings(self) -> None:
        self.assertEqual(
            extraction._normalize_date(datetime(2026, 3, 9, 10, 30, 0)),
            "2026-03-09T10:30:00",
        )
        self.assertEqual(extraction._normalize_date(date(2026, 3, 9)), "2026-03-09")
        self.assertEqual(extraction._normalize_date("2026-03-09"), "2026-03-09")
        self.assertIsNone(extraction._normalize_date(None))

    def test_normalize_match_trims_required_fields_and_keeps_optional_ones(self) -> None:
        record = {
            "winner1": " Ana ",
            "winner2": None,
            "loser1": " Caio ",
            "loser2": "Duda",
            "date": date(2026, 3, 9),
            "match_id": "abc",
            "score": "21-19",
        }

        normalized = extraction._normalize_match(record)

        self.assertEqual(
            normalized,
            {
                "winner1": "Ana",
                "winner2": "",
                "loser1": "Caio",
                "loser2": "Duda",
                "date": "2026-03-09",
                "match_id": "abc",
                "score": "21-19",
            },
        )

    def test_get_matches_normalizes_repository_payload(self) -> None:
        payload = [
            {
                "winner1": " Ana ",
                "winner2": "Bia",
                "loser1": "Caio",
                "loser2": "Duda",
                "date": "2026-03-09",
            }
        ]

        with patch("extraction.fetch_matches", return_value=payload):
            matches = extraction.get_matches()

        self.assertEqual(matches[0]["winner1"], "Ana")
        self.assertEqual(matches[0]["winner2"], "Bia")


if __name__ == "__main__":
    unittest.main()
