"""Middleware Flask para resolucao de grupos multi-tenant."""

from __future__ import annotations

from flask import abort, g, request, session

from src.redinha_stats.infrastructure.supabase.groups_repository import fetch_group_by_slug


def _clear_cross_group_sessions(group_id: str) -> None:
    """Limpa sessoes de admin e jogador quando pertencem a outro grupo."""

    player_group_id = session.get("player_group_id")
    if player_group_id and player_group_id != group_id:
        session.pop("player_user_id", None)
        session.pop("player_name", None)
        session.pop("player_group_id", None)

    admin_group_id = session.get("admin_group_id")
    if admin_group_id and admin_group_id != group_id:
        session.pop("admin_authenticated", None)
        session.pop("admin_role", None)
        session.pop("admin_group_id", None)
        session.pop("tournament_edit_keys", None)


def resolve_current_group() -> None:
    """Extrai o slug da URL atual e injeta o grupo em ``flask.g.current_group``."""

    slug = request.view_args and request.view_args.get("slug")
    if not slug:
        return

    group = fetch_group_by_slug(slug)
    if group is None:
        abort(404, description=f"Grupo '{slug}' nao encontrado.")

    g.current_group = group
    _clear_cross_group_sessions(group["id"])
