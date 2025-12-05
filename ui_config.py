"""Configurações de marca e parâmetros de interface customizáveis."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Tuple


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


def _normalize_tuple(values: Iterable[str] | str | None, *, default: Tuple[str, ...]) -> Tuple[str, ...]:
    if values is None:
        return default

    if isinstance(values, str):
        items = [values]
    else:
        items = list(values)

    return tuple(str(item) for item in items if str(item)) or default


def build_ui_config(overrides: Mapping[str, Any] | None = None) -> UiConfig:
    """Cria uma instância de :class:`UiConfig` mesclando *overrides* opcionais."""

    if not overrides:
        return UI_CONFIG

    branding_override = overrides.get("branding")
    branding_cfg = (
        replace(UI_CONFIG.branding, **branding_override)
        if isinstance(branding_override, Mapping)
        else UI_CONFIG.branding
    )

    ranking_periods = _normalize_tuple(
        overrides.get("ranking_periods"), default=UI_CONFIG.ranking_periods
    )
    games_periods = _normalize_tuple(
        overrides.get("games_periods"), default=UI_CONFIG.games_periods
    )
    excluded_players = _normalize_tuple(
        overrides.get("excluded_players"), default=UI_CONFIG.excluded_players
    )

    return replace(
        UI_CONFIG,
        branding=branding_cfg,
        ranking_periods=ranking_periods,
        games_periods=games_periods,
        default_ranking_period=overrides.get(
            "default_ranking_period", UI_CONFIG.default_ranking_period
        ),
        default_games_period=overrides.get(
            "default_games_period", UI_CONFIG.default_games_period
        ),
        excluded_players=excluded_players,
        average_match_minutes=int(
            overrides.get("average_match_minutes", UI_CONFIG.average_match_minutes)
        ),
    )


def get_ui_config(overrides: Mapping[str, Any] | None = None) -> UiConfig:
    """Retorna a configuração imutável de UI usada pelo Flask."""

    return build_ui_config(overrides)


__all__ = ["NavigationLink", "BrandingConfig", "UiConfig", "UI_CONFIG", "get_ui_config"]
