"""Repositório de jogadores cadastrados no Supabase.

Substitui player_registry_store.py (JSON local).
Jogadores são vinculados a um grupo via group_members.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from supabase import Client, create_client

from src.redinha_stats.config.settings import SUPABASE_SERVICE_KEY, SUPABASE_URL


USERS_TABLE = "users"
GROUP_MEMBERS_TABLE = "group_members"


@lru_cache(maxsize=1)
def _get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def fetch_players_for_group(group_id: str) -> List[str]:
    """Retorna a lista de nomes dos jogadores cadastrados no grupo."""

    response = (
        _get_client()
        .table(GROUP_MEMBERS_TABLE)
        .select("users(name)")
        .eq("group_id", group_id)
        .execute()
    )
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao buscar jogadores do grupo: {response.error}")

    names: List[str] = []
    for row in (response.data or []):
        user = row.get("users") or {}
        name = str(user.get("name") or "").strip()
        if name:
            names.append(name)

    return sorted(names, key=str.casefold)


def add_player_to_group(group_id: str, name: str, role: str = "player") -> Optional[Dict[str, Any]]:
    """Cadastra um novo jogador no grupo.

    1. Cria (ou reutiliza) o usuário pelo nome.
    2. Cria o vínculo em group_members.
    Retorna o registro de group_members criado, ou None se o jogador já estava no grupo.
    """

    normalized = name.strip()
    if not normalized:
        raise ValueError("Nome do jogador não pode ser vazio.")

    client = _get_client()

    # Busca usuário existente pelo nome (sem email por enquanto)
    user_resp = (
        client.table(USERS_TABLE)
        .select("id")
        .eq("name", normalized)
        .limit(1)
        .execute()
    )
    if getattr(user_resp, "error", None):
        raise RuntimeError(f"Erro ao buscar usuário: {user_resp.error}")

    user_data = user_resp.data or []
    if user_data:
        user_id = user_data[0]["id"]
    else:
        # Cria usuário sem email (email gerado como placeholder único)
        import uuid
        placeholder_email = f"{uuid.uuid4()}@placeholder.local"
        create_resp = (
            client.table(USERS_TABLE)
            .insert({"name": normalized, "email": placeholder_email})
            .execute()
        )
        if getattr(create_resp, "error", None):
            raise RuntimeError(f"Erro ao criar usuário: {create_resp.error}")
        user_id = (create_resp.data or [{}])[0]["id"]

    # Verifica se já é membro
    member_resp = (
        client.table(GROUP_MEMBERS_TABLE)
        .select("id")
        .eq("group_id", group_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if getattr(member_resp, "error", None):
        raise RuntimeError(f"Erro ao verificar membro: {member_resp.error}")

    if member_resp.data:
        return None  # já é membro

    insert_resp = (
        client.table(GROUP_MEMBERS_TABLE)
        .insert({"group_id": group_id, "user_id": user_id, "role": role})
        .execute()
    )
    if getattr(insert_resp, "error", None):
        raise RuntimeError(f"Erro ao adicionar jogador ao grupo: {insert_resp.error}")

    data = insert_resp.data or []
    return data[0] if data else {}
