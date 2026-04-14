import os

import pandas as pd
import pytest


os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon-key")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service-key")


@pytest.fixture
def sample_matches_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "winner1": "Ana",
                "winner2": "Bia",
                "loser1": "Caio",
                "loser2": "Duda",
                "score": "21-18",
            },
            {
                "winner1": "Caio",
                "winner2": "Duda",
                "loser1": "Ana",
                "loser2": "Bia",
                "score": "21-19",
            },
        ],
        index=pd.to_datetime(["2026-03-09", "2026-03-01"]),
    )


@pytest.fixture
def sample_players_ranking() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "jogadores": "Ana",
                "aproveitamento": 67,
                "vitórias": 2,
                "derrotas": 1,
                "saldo": 1,
                "jogos": 3,
            },
            {
                "jogadores": "Caio",
                "aproveitamento": 33,
                "vitórias": 1,
                "derrotas": 2,
                "saldo": -1,
                "jogos": 3,
            },
        ]
    )


@pytest.fixture
def sample_duos_ranking() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "duplas": "Ana e Bia",
                "aproveitamento": 50,
                "vitórias": 1,
                "derrotas": 1,
                "saldo": 0,
                "jogos": 2,
            }
        ]
    )


@pytest.fixture
def main_api_module():
    import main_api

    main_api.app.config.update(TESTING=True)
    return main_api


@pytest.fixture
def client(main_api_module):
    return main_api_module.app.test_client()
