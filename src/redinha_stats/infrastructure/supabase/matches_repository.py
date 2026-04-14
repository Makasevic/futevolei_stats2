"""Utilities for interacting with the Supabase ``matches`` table."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, MutableMapping, Optional

from supabase import Client, create_client

from src.redinha_stats.config.settings import SUPABASE_ANON_KEY, SUPABASE_URL


MATCHES_TABLE = "matches"


@lru_cache(maxsize=1)
def _get_client() -> Client:
    """Create (or reuse) a Supabase client instance."""
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def fetch_matches(group_id: Optional[str] = None) -> List[MutableMapping[str, Any]]:
    """Return matches stored in Supabase ordered by date.

    If *group_id* is provided, only matches belonging to that group are returned.
    When None, returns all matches (used during transition before Fase 2 migration).
    """

    query = _get_client().table(MATCHES_TABLE).select("*").order("date", desc=False)

    if group_id is not None:
        query = query.eq("group_id", group_id)

    response = query.execute()
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao buscar partidas no Supabase: {response.error}")
    return list(response.data or [])


def insert_match(match: Dict[str, Any], group_id: Optional[str] = None) -> MutableMapping[str, Any]:
    """Insert a new match in Supabase and return the created record.

    If *group_id* is provided, it is merged into the payload.
    """

    payload = dict(match)
    if group_id is not None:
        payload["group_id"] = group_id

    response = _get_client().table(MATCHES_TABLE).insert(payload).execute()
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao inserir partida no Supabase: {response.error}")

    data = response.data or []
    return data[0] if data else {}


def update_match(
    match_id: Any,
    updates: Dict[str, Any],
    *,
    id_field: str = "id",
) -> MutableMapping[str, Any]:
    """Update an existing match identified by ``match_id``."""

    if match_id is None:
        raise ValueError("match_id é obrigatório para atualização")

    response = (
        _get_client()
        .table(MATCHES_TABLE)
        .update(updates)
        .eq(id_field, match_id)
        .execute()
    )
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao atualizar partida no Supabase: {response.error}")

    data = response.data or []
    return data[0] if data else {}


def delete_match(match_id: Any, *, id_field: str = "id") -> None:
    """Remove uma partida do Supabase pelo identificador fornecido."""

    if match_id is None:
        raise ValueError("match_id é obrigatório para exclusão")

    response = (
        _get_client().table(MATCHES_TABLE).delete().eq(id_field, match_id).execute()
    )
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao excluir partida no Supabase: {response.error}")
