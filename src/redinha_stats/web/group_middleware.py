"""Middleware Flask para resolução de grupos multi-tenant.

O padrão de URL é /g/<slug>/... — o slug identifica o grupo.
O before_request resolve o grupo via Supabase e injeta em flask.g.current_group.
Retorna 404 automaticamente se o slug não existir.
"""

from __future__ import annotations

from flask import abort, g, request

from src.redinha_stats.infrastructure.supabase.groups_repository import fetch_group_by_slug


def resolve_current_group() -> None:
    """Extrai o slug da URL atual e injeta o grupo em flask.g.current_group.

    Deve ser registrado como before_request no blueprint de grupo.
    Só age em rotas que contêm '/g/<slug>' — ignora demais requests.
    """

    # O Flask já decompôs a URL; o view_args contém 'slug' para rotas do blueprint.
    slug = request.view_args and request.view_args.get("slug")
    if not slug:
        return  # rota fora do blueprint de grupo — não faz nada

    group = fetch_group_by_slug(slug)
    if group is None:
        abort(404, description=f"Grupo '{slug}' não encontrado.")

    g.current_group = group
