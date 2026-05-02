"""Blueprint de superadmin — criação de novos grupos (multi-tenant).

Rota pública: /grupos/novo
Protegida por SUPERADMIN_PASSWORD (env var).
"""

from __future__ import annotations

import re

from flask import Blueprint, redirect, render_template, request, url_for

import bcrypt

from src.redinha_stats.config.settings import SUPERADMIN_PASSWORD
from src.redinha_stats.infrastructure.supabase.groups_repository import (
    create_group,
    fetch_group_by_slug,
)
from src.redinha_stats.infrastructure.supabase.invitations_repository import (
    fetch_invitation_by_token,
    is_invitation_valid,
    mark_invitation_used,
)
from src.redinha_stats.infrastructure.supabase.players_repository import add_player_to_group


bp = Blueprint("superadmin", __name__)

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@bp.route("/grupos/novo", methods=["GET", "POST"])
def novo_grupo():
    error = None
    success_url = None

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        slug = request.form.get("slug", "").strip().lower()
        password = request.form.get("superadmin_password", "").strip()

        if not SUPERADMIN_PASSWORD:
            error = "SUPERADMIN_PASSWORD não configurada no servidor."
        elif password != SUPERADMIN_PASSWORD:
            error = "Senha incorreta."
        elif not name:
            error = "Nome do grupo é obrigatório."
        elif not slug:
            error = "Slug é obrigatório."
        elif not _SLUG_RE.match(slug):
            error = "Slug inválido. Use apenas letras minúsculas, números e hífen (ex: meu-grupo)."
        else:
            existing = fetch_group_by_slug(slug)
            if existing:
                error = f"Já existe um grupo com o slug '{slug}'."
            else:
                try:
                    create_group(slug=slug, name=name)
                    success_url = f"/g/{slug}/"
                except Exception as exc:
                    error = f"Erro ao criar grupo: {exc}"

    return render_template(
        "grupo_novo.html",
        error=error,
        success_url=success_url,
        base_url="",
    )


@bp.route("/convite/<token>", methods=["GET", "POST"])
def resgatar_convite(token: str):
    invitation = fetch_invitation_by_token(token)

    if not invitation:
        return render_template("convite_invalido.html", motivo="Link de convite inválido.", base_url=""), 404

    valid, motivo = is_invitation_valid(invitation)
    if not valid:
        return render_template("convite_invalido.html", motivo=motivo, base_url=""), 410

    group_slug = None
    group_name = None
    try:
        from src.redinha_stats.infrastructure.supabase.groups_repository import fetch_group_by_slug
        from supabase import create_client
        from src.redinha_stats.config.settings import SUPABASE_SERVICE_KEY, SUPABASE_URL
        client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        resp = client.table("groups").select("slug,name").eq("id", invitation["group_id"]).limit(1).execute()
        if resp.data:
            group_slug = resp.data[0]["slug"]
            group_name = resp.data[0]["name"]
    except Exception:
        pass

    error = None
    pre_name = invitation.get("name") or ""

    if request.method == "POST":
        name = pre_name  # nome vem do convite, não do form
        email = request.form.get("email", "").strip().lower()
        birthday = request.form.get("birthday", "").strip() or None
        password = request.form.get("password", "")
        password_confirm = request.form.get("password_confirm", "")

        if not email:
            error = "Email é obrigatório."
        elif not birthday:
            error = "Data de aniversário é obrigatória."
        elif not password:
            error = "Senha é obrigatória."
        elif len(password) < 6:
            error = "Senha deve ter pelo menos 6 caracteres."
        elif password != password_confirm:
            error = "Senhas não coincidem."
        else:
            try:
                from src.redinha_stats.config.settings import SUPABASE_SERVICE_KEY, SUPABASE_URL
                from supabase import create_client
                client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

                password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

                # Verifica se email já existe
                existing = client.table("users").select("id").eq("email", email).limit(1).execute()
                if existing.data:
                    error = "Este email já está cadastrado."
                else:
                    # Cria ou atualiza usuário
                    user_id = invitation.get("user_id")
                    if user_id:
                        # Atualiza usuário existente (era placeholder)
                        client.table("users").update({
                            "email": email,
                            "name": name,
                            "password_hash": password_hash,
                            "email_verified": True,
                            "birthday": birthday,
                        }).eq("id", user_id).execute()
                    else:
                        # Cria novo usuário
                        resp = client.table("users").insert({
                            "email": email,
                            "name": name,
                            "password_hash": password_hash,
                            "email_verified": True,
                            "birthday": birthday,
                        }).execute()
                        user_id = (resp.data or [{}])[0].get("id")

                    # Vincula ao grupo se ainda não estiver
                    if user_id and invitation.get("group_id"):
                        add_player_to_group(
                            invitation["group_id"],
                            name,
                            user_id=user_id,
                        )

                    mark_invitation_used(token)
                    redirect_url = f"/g/{group_slug}/" if group_slug else "/"
                    return redirect(redirect_url)

            except Exception as exc:
                error = f"Erro ao criar conta: {exc}"

    return render_template(
        "convite_resgatar.html",
        token=token,
        pre_name=pre_name,
        pre_email=invitation.get("email") or "",
        group_name=group_name,
        error=error,
        base_url="",
    )
