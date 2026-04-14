"""Repositório de grupos no Supabase."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

from supabase import Client, create_client

from src.redinha_stats.config.settings import SUPABASE_SERVICE_KEY, SUPABASE_URL


GROUPS_TABLE = "groups"


@lru_cache(maxsize=1)
def _get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def fetch_group_by_slug(slug: str) -> Optional[Dict[str, Any]]:
    """Retorna o grupo com o slug fornecido, ou None se não existir."""

    response = (
        _get_client()
        .table(GROUPS_TABLE)
        .select("*")
        .eq("slug", slug)
        .limit(1)
        .execute()
    )
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao buscar grupo '{slug}': {response.error}")

    data = response.data or []
    return data[0] if data else None


def create_group(slug: str, name: str) -> Dict[str, Any]:
    """Cria um novo grupo e retorna o registro criado."""

    response = (
        _get_client()
        .table(GROUPS_TABLE)
        .insert({"slug": slug, "name": name})
        .execute()
    )
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao criar grupo '{slug}': {response.error}")

    data = response.data or []
    return data[0] if data else {}
