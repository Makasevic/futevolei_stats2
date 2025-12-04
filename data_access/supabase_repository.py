"""Utilities for interacting with the Supabase ``matches`` table."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, MutableMapping

from supabase import Client, create_client

from config import SUPABASE_ANON_KEY, SUPABASE_URL


MATCHES_TABLE = "matches"

@lru_cache(maxsize=1)
def _get_client() -> Client:
    """Create (or reuse) a Supabase client instance."""
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def fetch_matches() -> List[MutableMapping[str, Any]]:
    """Return all matches stored in Supabase ordered by date."""

    response = _get_client().table(MATCHES_TABLE).select("*").order("date", desc=False).execute()
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao buscar partidas no Supabase: {response.error}")
    print(response)
    return list(response.data or [])


def insert_match(match: Dict[str, Any]) -> MutableMapping[str, Any]:
    """Insert a new match in Supabase and return the created record."""

    response = _get_client().table(MATCHES_TABLE).insert(match).execute()
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
    """Update an existing match identified by ``match_id``.

    Parameters
    ----------
    match_id:
        Valor utilizado na cláusula ``eq`` para localizar o registro.
    updates:
        Campos a serem atualizados.
    id_field:
        Nome da coluna utilizada para filtrar o registro (``id`` por padrão).
    """

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

