import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from src.redinha_stats.infrastructure.local import player_registry_store as player_registry


class PlayerRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests") / "_tmp" / self.id().replace(".", "_")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.store_path = self.temp_dir / "players_registry.json"
        self.store_patch = patch.object(player_registry, "_STORE_PATH", self.store_path)
        self.store_patch.start()

    def tearDown(self) -> None:
        self.store_patch.stop()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_load_registered_players_returns_empty_list_for_missing_store(self) -> None:
        self.assertEqual(player_registry.load_registered_players(), [])

    def test_add_player_persists_sorted_unique_names(self) -> None:
        self.assertTrue(player_registry.add_player("Carlos"))
        self.assertTrue(player_registry.add_player("ana"))
        self.assertFalse(player_registry.add_player("Carlos"))

        self.assertEqual(player_registry.load_registered_players(), ["ana", "Carlos"])

    def test_add_player_collapses_repeated_spaces(self) -> None:
        self.assertTrue(player_registry.add_player("J. Victor  Adão"))
        self.assertFalse(player_registry.add_player("J. Victor Adão"))

        self.assertEqual(player_registry.load_registered_players(), ["J. Victor Adão"])

    def test_add_player_rejects_blank_names(self) -> None:
        self.assertFalse(player_registry.add_player("   "))
        self.assertFalse(self.store_path.exists())

    def test_load_registered_players_handles_invalid_json(self) -> None:
        self.store_path.write_text("{invalido", encoding="utf-8")

        self.assertEqual(player_registry.load_registered_players(), [])

    def test_ensure_store_creates_json_array_file(self) -> None:
        player_registry._ensure_store()

        self.assertTrue(self.store_path.exists())
        self.assertEqual(json.loads(self.store_path.read_text(encoding="utf-8")), [])


if __name__ == "__main__":
    unittest.main()
