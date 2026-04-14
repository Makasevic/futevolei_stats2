from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from typing import Any, Callable, Dict, Iterable, List, Sequence

import pandas as pd


def normalize_admin_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    return None


def is_valid_identifier(value: Any) -> bool:
    return value not in (None, "") and value == value


def identifier_from_match_data(row: Dict[str, Any]) -> tuple[Any | None, str]:
    for field in ("match_id", "id"):
        value = row.get(field)
        if is_valid_identifier(value):
            return value, field
    return None, "id"


def players_from_df(df: pd.DataFrame | None, team_fields: Sequence[str]) -> List[str]:
    if df is None or df.empty:
        return []

    players: List[str] = []
    seen = set()
    for field in team_fields:
        if field not in df.columns:
            continue
        for value in df[field].tolist():
            name = str(value or "").strip()
            if name and name not in seen:
                seen.add(name)
                players.append(name)

    players.sort()
    return players


def players_ranked_by_games(df: pd.DataFrame | None, team_fields: Sequence[str]) -> List[str]:
    if df is None or df.empty:
        return []

    games_by_player: Counter[str] = Counter()
    for field in team_fields:
        if field not in df.columns:
            continue
        for value in df[field].tolist():
            if pd.isna(value):
                continue
            name = str(value).strip()
            if not name or "Outro" in name:
                continue
            games_by_player[name] += 1

    ordered_players = sorted(
        games_by_player.items(),
        key=lambda item: (-item[1], item[0].casefold()),
    )
    return [name for name, _ in ordered_players]


def registered_players(
    df: pd.DataFrame | None,
    *,
    team_fields: Sequence[str],
    load_registered_players: Callable[[], List[str]],
    excluded_players: Callable[[], set],
) -> List[str]:
    base_players = set(players_from_df(df, team_fields))
    manual_players = {name for name in load_registered_players() if name}
    combined = sorted((base_players | manual_players) - excluded_players())
    return combined


def matches_from_df(
    df: pd.DataFrame | None,
    *,
    normalize_admin_date: Callable[[Any], date | None],
    identifier_from_match_data: Callable[[Dict[str, Any]], tuple[Any | None, str]],
) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    registros: List[Dict[str, Any]] = []
    for row in df.reset_index().to_dict("records"):
        match: Dict[str, Any] = {
            "id": row.get("id"),
            "match_id": row.get("match_id"),
            "winner1": str(row.get("winner1") or "").strip(),
            "winner2": str(row.get("winner2") or "").strip(),
            "loser1": str(row.get("loser1") or "").strip(),
            "loser2": str(row.get("loser2") or "").strip(),
            "score": str(row.get("score") or "").strip(),
        }
        score_a = ""
        score_b = ""
        if match["score"]:
            import re

            parsed = re.match(r"^\s*(\d+)\s*[xX-]\s*(\d+)\s*$", match["score"])
            if parsed:
                score_a = parsed.group(1)
                score_b = parsed.group(2)
        match["score_a"] = score_a
        match["score_b"] = score_b
        match["date"] = normalize_admin_date(row.get("date"))
        identifier_value, identifier_field = identifier_from_match_data(row)
        match["_identifier_value"] = identifier_value
        match["_identifier_field"] = identifier_field
        display_identifier = row.get("match_id") or row.get("id")
        match["identifier_display"] = (
            str(display_identifier) if display_identifier not in (None, "") else ""
        )
        formatted_date = match["date"].isoformat() if match["date"] else "Sem data"
        match["label"] = (
            f"{formatted_date}  {match['winner1']} & {match['winner2']} x "
            f"{match['loser1']} & {match['loser2']}"
        )
        if match["identifier_display"]:
            match["label"] += f" (ID: {match['identifier_display']})"
        registros.append(match)

    registros.sort(key=lambda item: item.get("date") or date.min, reverse=True)
    return registros


def validate_registered_players(
    players: Iterable[str],
    *,
    registered_players: Sequence[str],
) -> List[str]:
    registered = set(registered_players)
    return [name for name in players if name not in registered]


def validate_match_data(
    match_id: Any,
    action: str,
    payload: Dict[str, Any],
    *,
    team_fields: Sequence[str],
) -> List[str]:
    errors: List[str] = []

    if action in {"Atualizar", "Excluir"} and match_id is None:
        errors.append("Selecione uma partida para continuar.")

    if action != "Excluir":
        players = [payload.get(field, "") for field in team_fields]
        if any(not player for player in players):
            errors.append("Informe os quatro jogadores da partida.")
        if payload.get("winner1") == payload.get("winner2"):
            errors.append("Os vencedores devem ser jogadores diferentes.")
        if payload.get("loser1") == payload.get("loser2"):
            errors.append("Os perdedores devem ser jogadores diferentes.")
        if len({p for p in players if p}) < 4:
            errors.append("Cada jogador so pode aparecer uma vez na partida.")
        if not isinstance(payload.get("date"), date):
            errors.append("Informe uma data valida.")

    return errors


def parse_form_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def set_admin_feedback(session_obj: Any, level: str, message: str) -> None:
    session_obj["admin_feedback"] = {"status": level, "message": message}


def unlocked_tournament_keys(raw: Any) -> set[str]:
    if not isinstance(raw, list):
        return set()
    return {str(item).strip() for item in raw if str(item).strip()}


def set_unlocked_tournament_keys(session_obj: Any, keys: set[str]) -> None:
    session_obj["tournament_edit_keys"] = sorted(keys)
