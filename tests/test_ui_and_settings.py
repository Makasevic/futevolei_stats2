import importlib
import unittest
from unittest.mock import patch

import app_settings
import ui_config


class UiConfigTests(unittest.TestCase):
    def test_build_ui_config_applies_overrides(self) -> None:
        config = ui_config.build_ui_config(
            {
                "branding": {"site_name": "Meu Site"},
                "ranking_periods": ["7 dias", "30 dias"],
                "games_periods": "90 dias",
                "excluded_players": ["X", "Y"],
                "default_games_period": "90 dias",
                "average_match_minutes": "25",
            }
        )

        self.assertEqual(config.branding.site_name, "Meu Site")
        self.assertEqual(config.ranking_periods, ("7 dias", "30 dias"))
        self.assertEqual(config.games_periods, ("90 dias",))
        self.assertEqual(config.excluded_players, ("X", "Y"))
        self.assertEqual(config.default_games_period, "90 dias")
        self.assertEqual(config.average_match_minutes, 25)

    def test_build_ui_config_without_overrides_returns_default_singleton(self) -> None:
        config = ui_config.build_ui_config()

        self.assertIs(config, ui_config.UI_CONFIG)


class AppSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        importlib.reload(app_settings)
        self.original_default = app_settings.DEFAULT_CONFIG

    def tearDown(self) -> None:
        app_settings.DEFAULT_CONFIG = self.original_default

    def test_get_config_returns_default_when_streamlit_is_unavailable(self) -> None:
        with patch.object(app_settings, "st", None):
            config = app_settings.get_config()

        self.assertEqual(config, app_settings.DEFAULT_CONFIG)

    def test_update_config_replaces_default_when_streamlit_is_unavailable(self) -> None:
        with patch.object(app_settings, "st", None):
            updated = app_settings.update_config(min_duo_matches=8, tendencia_threshold=2.5)

        self.assertEqual(updated.min_duo_matches, 8)
        self.assertEqual(updated.tendencia_threshold, 2.5)
        self.assertEqual(app_settings.DEFAULT_CONFIG, updated)

    def test_normalize_dict_fills_missing_values_from_default(self) -> None:
        config = app_settings._normalize_dict({"min_duo_matches": 9})

        self.assertEqual(config.min_duo_matches, 9)
        self.assertEqual(
            config.min_participation_ratio,
            app_settings.DEFAULT_CONFIG.min_participation_ratio,
        )


if __name__ == "__main__":
    unittest.main()
