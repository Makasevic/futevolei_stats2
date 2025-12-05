"""Configurações de marca e parâmetros de interface customizáveis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class NavigationLink:
    """Link de navegação exibido no topo da página."""

    label: str
    url: str
    page_id: str | None = None


@dataclass(frozen=True)
class BrandingConfig:
    """Textos e elementos de identidade visual."""

    site_name: str = "Redinha Stats"
    site_tagline: str = "Jogos feios, site bonito."
    brand_mark: str = "RS"
    hero_eyebrow: str = "ranking em tempo real"
    hero_title: str = "Redinha Stats"
    hero_description: str = (
        "Aqui mora a verdade: sai o blá-blá-blá, entram os números."
        """In God we trust. All others must bring data."""
    )


@dataclass(frozen=True)
class UiConfig:
    """Parâmetros ajustáveis da interface HTML/Flask."""

    branding: BrandingConfig = field(default_factory=BrandingConfig)
    navigation: Tuple[NavigationLink, ...] = field(
        default_factory=lambda: (
            NavigationLink("Ranking", "/", "ranking"),
            NavigationLink("Infos", "/infos", "infos"),
            NavigationLink("Detalhamento", "/detalhamento", "detalhamento"),
            NavigationLink("Jogos", "/jogos", "jogos"),
            NavigationLink("Config", "#", "config"),
            NavigationLink("Admin", "/admin", "admin"),
        )
    )
    ranking_periods: Tuple[str, ...] = (
        "1 dia",
        "30 dias",
        "60 dias",
        "90 dias",
        "180 dias",
        "360 dias",
        "Todos",
    )
    games_periods: Tuple[str, ...] = (
        "1 dia",
        "30 dias",
        "60 dias",
        "90 dias",
        "180 dias",
        "360 dias",
        "Todos",
        "Data",
    )
    default_ranking_period: str = "1 dia"
    default_games_period: str = "90 dias"
    excluded_players: Tuple[str, ...] = ("Outro_1", "Outro_2")
    average_match_minutes: int = 20


UI_CONFIG = UiConfig()


def get_ui_config() -> UiConfig:
    """Retorna a configuração imutável de UI usada pelo Flask."""

    return UI_CONFIG


__all__ = ["NavigationLink", "BrandingConfig", "UiConfig", "UI_CONFIG", "get_ui_config"]
