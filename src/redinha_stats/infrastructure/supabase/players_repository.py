"""Repositorio de jogadores cadastrados no Supabase.

Substitui ``player_registry_store.py`` (JSON local).
Jogadores sao vinculados a um grupo via ``group_members``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from supabase import Client, create_client

from src.redinha_stats.config.settings import SUPABASE_SERVICE_KEY, SUPABASE_URL


USERS_TABLE = "users"
GROUP_MEMBERS_TABLE = "group_members"
MATCHES_TABLE = "matches"
TEAM_FIELDS = ("winner1", "winner2", "loser1", "loser2")


def _normalize_player_name(value: str) -> str:
    return " ".join(str(value or "").split())


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
        name = _normalize_player_name(str(user.get("name") or ""))
        if name:
            names.append(name)

    return sorted(names, key=str.casefold)


def add_player_to_group(
    group_id: str,
    name: str,
    role: str = "player",
    *,
    user_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Cadastra um novo jogador no grupo.

    Quando ``user_id`` e informado, ele e reutilizado explicitamente para
    evitar colisoes por nome em fluxos de convite e cadastro.
    """

    normalized = _normalize_player_name(name)
    if not normalized:
        raise ValueError("Nome do jogador nao pode ser vazio.")

    client = _get_client()
    resolved_user_id = user_id

    if resolved_user_id is None:
        user_resp = (
            client.table(USERS_TABLE)
            .select("id")
            .eq("name", normalized)
            .limit(1)
            .execute()
        )
        if getattr(user_resp, "error", None):
            raise RuntimeError(f"Erro ao buscar usuario: {user_resp.error}")

        user_data = user_resp.data or []
        if user_data:
            resolved_user_id = user_data[0]["id"]
        else:
            import uuid

            placeholder_email = f"{uuid.uuid4()}@placeholder.local"
            create_resp = (
                client.table(USERS_TABLE)
                .insert({"name": normalized, "email": placeholder_email})
                .execute()
            )
            if getattr(create_resp, "error", None):
                raise RuntimeError(f"Erro ao criar usuario: {create_resp.error}")
            resolved_user_id = (create_resp.data or [{}])[0]["id"]

    member_resp = (
        client.table(GROUP_MEMBERS_TABLE)
        .select("id")
        .eq("group_id", group_id)
        .eq("user_id", resolved_user_id)
        .limit(1)
        .execute()
    )
    if getattr(member_resp, "error", None):
        raise RuntimeError(f"Erro ao verificar membro: {member_resp.error}")

    if member_resp.data:
        return None

    insert_resp = (
        client.table(GROUP_MEMBERS_TABLE)
        .insert({"group_id": group_id, "user_id": resolved_user_id, "role": role})
        .execute()
    )
    if getattr(insert_resp, "error", None):
        raise RuntimeError(f"Erro ao adicionar jogador ao grupo: {insert_resp.error}")

    data = insert_resp.data or []
    return data[0] if data else {}


def is_user_member_of_group(user_id: str, group_id: str) -> bool:
    """Retorna ``True`` quando o usuario participa do grupo informado."""

    response = (
        _get_client()
        .table(GROUP_MEMBERS_TABLE)
        .select("id")
        .eq("group_id", group_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao verificar membro do grupo: {response.error}")
    return bool(response.data)


def _find_user_ids_by_name(client: Client, name: str) -> List[str]:
    """Retorna todos os ``users.id`` cujo nome bate com *name*."""

    target = name.casefold()
    response = client.table(USERS_TABLE).select("id, name").execute()
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao buscar usuarios por nome: {response.error}")
    return [
        row["id"]
        for row in (response.data or [])
        if str(row.get("name") or "").casefold() == target and row.get("id")
    ]


def _name_exists_in_matches(client: Client, name: str) -> bool:
    """True se *name* aparece em qualquer dos 4 campos de time."""

    for field in TEAM_FIELDS:
        response = (
            client.table(MATCHES_TABLE)
            .select("id")
            .eq(field, name)
            .limit(1)
            .execute()
        )
        if getattr(response, "error", None):
            raise RuntimeError(f"Erro ao buscar partidas por nome: {response.error}")
        if response.data:
            return True
    return False


def _update_matches_player_name(
    client: Client,
    old_name: str,
    new_name: str,
) -> int:
    """Substitui *old_name* por *new_name* nos campos de time de todas as matches."""

    total_updated = 0
    for field in TEAM_FIELDS:
        response = (
            client.table(MATCHES_TABLE)
            .update({field: new_name})
            .eq(field, old_name)
            .execute()
        )
        if getattr(response, "error", None):
            raise RuntimeError(
                f"Erro ao atualizar partidas (campo {field}): {response.error}"
            )
        total_updated += len(response.data or [])
    return total_updated


def rename_player(old_name: str, new_name: str) -> Dict[str, Any]:
    """Renomeia um jogador globalmente."""

    old = _normalize_player_name(old_name)
    new = _normalize_player_name(new_name)
    if not old:
        raise ValueError("Nome atual nao pode ser vazio.")
    if not new:
        raise ValueError("Novo nome nao pode ser vazio.")
    if old.casefold() == new.casefold():
        raise ValueError("O novo nome e igual ao atual.")

    client = _get_client()

    new_user_ids = _find_user_ids_by_name(client, new)
    if new_user_ids:
        raise ValueError(
            f"Ja existe um usuario chamado '{new}'. Use a funcao de fusao."
        )
    if _name_exists_in_matches(client, new):
        raise ValueError(
            f"O nome '{new}' ja aparece em partidas. Use a funcao de fusao."
        )

    old_user_ids = _find_user_ids_by_name(client, old)
    has_matches = _name_exists_in_matches(client, old)
    if not old_user_ids and not has_matches:
        raise ValueError(f"Jogador '{old}' nao encontrado.")

    for uid in old_user_ids:
        update_user_resp = (
            client.table(USERS_TABLE)
            .update({"name": new})
            .eq("id", uid)
            .execute()
        )
        if getattr(update_user_resp, "error", None):
            raise RuntimeError(f"Erro ao renomear usuario: {update_user_resp.error}")

    matches_updated = _update_matches_player_name(client, old, new)

    return {"user_ids": old_user_ids, "matches_updated": matches_updated}


def merge_players(kept_name: str, removed_name: str) -> Dict[str, Any]:
    """Funde dois jogadores globalmente."""

    kept = _normalize_player_name(kept_name)
    removed = _normalize_player_name(removed_name)
    if not kept or not removed:
        raise ValueError("Informe os dois nomes para fundir.")
    if kept.casefold() == removed.casefold():
        raise ValueError("Os dois nomes sao iguais - nada a fundir.")

    client = _get_client()

    kept_user_ids = _find_user_ids_by_name(client, kept)
    removed_user_ids = _find_user_ids_by_name(client, removed)

    has_kept = bool(kept_user_ids) or _name_exists_in_matches(client, kept)
    has_removed = bool(removed_user_ids) or _name_exists_in_matches(client, removed)
    if not has_kept:
        raise ValueError(f"Jogador '{kept}' nao encontrado.")
    if not has_removed:
        raise ValueError(f"Jogador '{removed}' nao encontrado.")

    matches_updated = _update_matches_player_name(client, removed, kept)

    user_deleted = False
    for uid in removed_user_ids:
        member_resp = (
            client.table(GROUP_MEMBERS_TABLE)
            .delete()
            .eq("user_id", uid)
            .execute()
        )
        if getattr(member_resp, "error", None):
            raise RuntimeError(
                f"Erro ao remover vinculos do jogador: {member_resp.error}"
            )

        delete_resp = (
            client.table(USERS_TABLE)
            .delete()
            .eq("id", uid)
            .execute()
        )
        if getattr(delete_resp, "error", None):
            raise RuntimeError(f"Erro ao apagar usuario fundido: {delete_resp.error}")
        user_deleted = True

    return {
        "kept_user_ids": kept_user_ids,
        "removed_user_ids": removed_user_ids,
        "matches_updated": matches_updated,
        "user_deleted": user_deleted,
    }
