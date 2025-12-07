"""Registro local de jogadores cadastrados manualmente.

Os nomes são armazenados em um arquivo JSON simples para que possam ser
reutilizados na interface de administração, permitindo cadastro rápido de
novos jogadores mesmo antes de terem partidas lançadas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


_STORE_PATH = Path(__file__).resolve().parent.parent / "players_registry.json"


def _ensure_store() -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _STORE_PATH.exists():
        _STORE_PATH.write_text("[]", encoding="utf-8")


def load_registered_players() -> List[str]:
    """Carrega a lista de jogadores cadastrados manualmente."""

    if not _STORE_PATH.exists():
        return []

    try:
        data = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    return [str(item).strip() for item in data if str(item).strip()]


def add_player(name: str) -> bool:
    """Adiciona um novo jogador ao registro persistido.

    Retorna ``True`` se o jogador foi adicionado ou ``False`` quando o nome
    já existia ou estava vazio.
    """

    normalized = name.strip()
    if not normalized:
        return False

    players = load_registered_players()
    if normalized in players:
        return False

    players.append(normalized)
    players.sort(key=str.casefold)

    _ensure_store()
    _STORE_PATH.write_text(
        json.dumps(players, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return True


__all__ = ["add_player", "load_registered_players"]
