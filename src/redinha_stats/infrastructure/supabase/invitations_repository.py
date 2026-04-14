"""Repositório de convites no Supabase."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Optional

from supabase import Client, create_client

from src.redinha_stats.config.settings import SUPABASE_SERVICE_KEY, SUPABASE_URL


INVITATIONS_TABLE = "invitations"


@lru_cache(maxsize=1)
def _get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def create_invitation(
    group_id: str,
    *,
    user_id: Optional[str] = None,
    email: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Cria um convite e retorna o registro com o token gerado."""

    payload: Dict[str, Any] = {"group_id": group_id}
    if user_id:
        payload["user_id"] = user_id
    if email:
        payload["email"] = email.strip().lower()
    if name:
        payload["name"] = name.strip()

    response = _get_client().table(INVITATIONS_TABLE).insert(payload).execute()
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao criar convite: {response.error}")
    data = response.data or []
    return data[0] if data else {}


def fetch_invitation_by_token(token: str) -> Optional[Dict[str, Any]]:
    """Retorna o convite pelo token, ou None se não existir."""

    response = (
        _get_client()
        .table(INVITATIONS_TABLE)
        .select("*")
        .eq("token", token)
        .limit(1)
        .execute()
    )
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao buscar convite: {response.error}")
    data = response.data or []
    return data[0] if data else None


def mark_invitation_used(token: str) -> None:
    """Marca o convite como utilizado."""

    now = datetime.now(timezone.utc).isoformat()
    response = (
        _get_client()
        .table(INVITATIONS_TABLE)
        .update({"used_at": now})
        .eq("token", token)
        .execute()
    )
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao marcar convite como usado: {response.error}")


def is_invitation_valid(invitation: Dict[str, Any]) -> tuple[bool, str]:
    """Verifica se o convite é válido. Retorna (válido, motivo)."""

    if invitation.get("used_at"):
        return False, "Este convite já foi utilizado."

    expires_at_raw = invitation.get("expires_at")
    if expires_at_raw:
        try:
            expires_at = datetime.fromisoformat(str(expires_at_raw).replace("Z", "+00:00"))
            if datetime.now(timezone.utc) > expires_at:
                return False, "Este convite expirou."
        except (ValueError, TypeError):
            pass

    return True, ""
