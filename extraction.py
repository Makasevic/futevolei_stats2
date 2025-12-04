"""Data extraction helpers backed by Supabase."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from data_access.supabase_repository import fetch_matches


_REQUIRED_FIELDS = ("winner1", "winner2", "loser1", "loser2")
_OPTIONAL_FIELDS = ("id", "match_id", "score")


def _normalize_player(value: Any) -> str:
    return str(value or "").strip()


def _normalize_date(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, str):
        return value

    return str(value)


def _normalize_match(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {field: _normalize_player(record.get(field)) for field in _REQUIRED_FIELDS}
    normalized["date"] = _normalize_date(record.get("date"))

    for optional_field in _OPTIONAL_FIELDS:
        if optional_field in record:
            normalized[optional_field] = record.get(optional_field)

    return normalized


def get_matches() -> List[Dict[str, Any]]:
    """Fetch matches from Supabase ready for dataframe preparation."""

    raw_matches = fetch_matches()
    return [_normalize_match(match) for match in raw_matches]

