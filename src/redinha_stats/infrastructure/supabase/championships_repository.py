"""Repositório de campeonatos configuráveis no Supabase."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from supabase import Client, create_client

from src.redinha_stats.config.settings import SUPABASE_SERVICE_KEY, SUPABASE_URL


CHAMPIONSHIPS_TABLE = "championships"


@lru_cache(maxsize=1)
def _get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def fetch_championships_for_group(group_id: str) -> List[Dict[str, Any]]:
    """Retorna todos os campeonatos do grupo, ordenados do mais recente."""

    response = (
        _get_client()
        .table(CHAMPIONSHIPS_TABLE)
        .select("id,slug,title,description,format,config,edit_password,created_at")
        .eq("group_id", group_id)
        .order("created_at", desc=True)
        .execute()
    )
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao buscar campeonatos: {response.error}")
    return list(response.data or [])


def fetch_championship_by_slug(group_id: str, slug: str) -> Optional[Dict[str, Any]]:
    """Retorna um campeonato pelo slug dentro do grupo, ou None se não existir."""

    response = (
        _get_client()
        .table(CHAMPIONSHIPS_TABLE)
        .select("*")
        .eq("group_id", group_id)
        .eq("slug", slug)
        .limit(1)
        .execute()
    )
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao buscar campeonato '{slug}': {response.error}")
    data = response.data or []
    return data[0] if data else None


def create_championship(
    group_id: str,
    slug: str,
    title: str,
    description: str,
    edit_password: str,
    config: Dict[str, Any],
    format: str = "groups_knockout",
) -> Dict[str, Any]:
    """Insere um novo campeonato e retorna o registro criado."""

    payload = {
        "group_id": group_id,
        "slug": slug,
        "title": title,
        "description": description,
        "edit_password": edit_password or None,
        "config": config,
        "format": format,
    }
    response = (
        _get_client()
        .table(CHAMPIONSHIPS_TABLE)
        .insert(payload)
        .execute()
    )
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao criar campeonato: {response.error}")
    data = response.data or []
    return data[0] if data else {}
