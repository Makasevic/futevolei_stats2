"""Blueprint multi-tenant — rotas prefixadas com /g/<slug>/

Todas as rotas aqui são equivalentes às de main_api.py, mas escopadas por grupo.
O before_request resolve o grupo via slug e injeta flask.g.current_group.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
from flask import Blueprint, g, redirect, render_template, session, url_for

from awards import available_awards_years, get_awards_for_year
from championship import (
    available_championship_keys,
    get_championship_edit_password,
    get_championship_view,
    save_match_score,
)
from src.redinha_stats.config.app_settings import get_config, update_config
from src.redinha_stats.config.settings import ADMIN_PASSWORD, MATCH_ENTRY_PASSWORD
from src.redinha_stats.config.ui_config import get_ui_config
from src.redinha_stats.domain.matches.processing import filtrar_dados
from src.redinha_stats.infrastructure.supabase.matches_repository import (
    delete_match,
    insert_match,
    update_match,
)
from src.redinha_stats.infrastructure.supabase.players_repository import (
    add_player_to_group,
    fetch_players_for_group,
    merge_players,
    rename_player,
)
from src.redinha_stats.infrastructure.supabase.championships_repository import create_championship
from src.redinha_stats.infrastructure.supabase.invitations_repository import (
    create_invitation,
    fetch_invitation_by_token,
    is_invitation_valid,
    mark_invitation_used,
)
from src.redinha_stats.infrastructure.email_service import send_invite_email
from src.redinha_stats.web import admin_helpers, data_helpers, helpers as web_helpers, info_helpers, versus_helpers
from src.redinha_stats.web.group_middleware import resolve_current_group
from src.redinha_stats.web.routes.admin import admin_page_response
from src.redinha_stats.web.routes.api import hidden_players_response, ranking_api_response
from src.redinha_stats.web.routes.config_page import config_page_response
from src.redinha_stats.web.routes.details import detalhamento_page_response
from src.redinha_stats.web.routes.games import render_games_page
from src.redinha_stats.web.routes.pages import (
    awards_page_response,
    championship_page_response,
    infos_page_response,
    ranking_page_response,
)
from src.redinha_stats.web.routes.versus import versus_page_response
from detalhamento import calcular_metricas_dupla, calcular_metricas_gerais, calcular_metricas_jogador


bp = Blueprint("group", __name__, url_prefix="/g")
bp.before_request(resolve_current_group)


@bp.context_processor
def inject_group_context():
    """Injeta group_slug, base_url e dados do jogador logado em todos os templates do blueprint."""
    slug = getattr(g, "current_group", {}).get("slug", "")
    return {
        "group_slug": slug,
        "base_url": f"/g/{slug}" if slug else "",
        "player_logged_in": bool(session.get("player_user_id")),
        "player_name": session.get("player_name", ""),
    }


# ---------------------------------------------------------------------------
# Helpers escopados por grupo
# ---------------------------------------------------------------------------

_TEAM_FIELDS = ("winner1", "winner2", "loser1", "loser2")


def _group_id() -> str:
    return g.current_group["id"]


def _fetch_base_dataframe() -> pd.DataFrame:
    return data_helpers.fetch_base_dataframe(_group_id())


def _reset_cache() -> None:
    data_helpers.reset_cached_dataframe(_group_id())


def _current_ui_config():
    return get_ui_config()


def _excluded_players() -> set:
    return set(_current_ui_config().excluded_players)


def _jogadores_disponiveis(df: pd.DataFrame) -> List[str]:
    jogadores = {
        j
        for j in df[["winner1", "winner2", "loser1", "loser2"]].values.ravel()
        if isinstance(j, str) and "Outro" not in j
    }
    return sorted(jogadores - _excluded_players())


def _duplas_disponiveis(df: pd.DataFrame) -> List[str]:
    duplas = set(df["dupla_winner"].tolist() + df["dupla_loser"].tolist())
    return sorted(d for d in duplas if isinstance(d, str) and "Outro" not in d)


def _parceiros_por_jogador(df: pd.DataFrame) -> Dict[str, List[str]]:
    jogadores = set(_jogadores_disponiveis(df))
    parceiros: Dict[str, set] = {j: set() for j in jogadores}
    for _, row in df.iterrows():
        for par in [("winner1", "winner2"), ("loser1", "loser2")]:
            a, b = row.get(par[0]), row.get(par[1])
            if a in jogadores and b in jogadores:
                parceiros[a].add(b)
                parceiros[b].add(a)
    return {j: sorted(v) for j, v in parceiros.items()}


def _oponentes_por_jogador(df: pd.DataFrame) -> Dict[str, List[str]]:
    jogadores = set(_jogadores_disponiveis(df))
    oponentes: Dict[str, set] = {j: set() for j in jogadores}
    for _, row in df.iterrows():
        ws = [p for p in (row.get("winner1"), row.get("winner2")) if p in jogadores]
        ls = [p for p in (row.get("loser1"), row.get("loser2")) if p in jogadores]
        for w in ws:
            oponentes[w].update(ls)
        for l in ls:
            oponentes[l].update(ws)
    return {j: sorted(v) for j, v in oponentes.items()}


def _oponentes_por_dupla_jogadores(df: pd.DataFrame) -> Dict[str, List[str]]:
    jogadores = set(_jogadores_disponiveis(df))
    duplas = set(_duplas_disponiveis(df))
    oponentes: Dict[str, set] = {d: set() for d in duplas}
    for _, row in df.iterrows():
        dw, dl = row.get("dupla_winner"), row.get("dupla_loser")
        ls = [p for p in (row.get("loser1"), row.get("loser2")) if p in jogadores]
        ws = [p for p in (row.get("winner1"), row.get("winner2")) if p in jogadores]
        if dw in duplas:
            oponentes[dw].update(ls)
        if dl in duplas:
            oponentes[dl].update(ws)
    return {d: sorted(v) for d, v in oponentes.items()}


def _estatisticas_dupla(df: pd.DataFrame, dupla: str) -> Any:
    return versus_helpers.estatisticas_dupla(df, dupla)


def _confronto_direto_duplas(df: pd.DataFrame, dupla1: str, dupla2: str) -> Any:
    return versus_helpers.confronto_direto_duplas(df, dupla1, dupla2)


def _estatisticas_jogador_individual(df: pd.DataFrame, jogador: str) -> Any:
    return versus_helpers.estatisticas_jogador_individual(df, jogador, excluded_players=_excluded_players())


def _confronto_direto(df: pd.DataFrame, jogador1: str, jogador2: str) -> Any:
    return versus_helpers.confronto_direto(df, jogador1, jogador2)


def _filter_rankings(*args):
    return data_helpers.filter_rankings(*args, group_id=_group_id())


def _registered_players(df: pd.DataFrame) -> List[str]:
    return admin_helpers.registered_players(
        df,
        team_fields=_TEAM_FIELDS,
        load_registered_players=lambda: fetch_players_for_group(_group_id()),
        excluded_players=_excluded_players,
    )


def _validate_registered_players(players):
    return admin_helpers.validate_registered_players(
        players,
        registered_players=_registered_players(_fetch_base_dataframe()),
    )


def _validate_match_data(match_id, action, payload):
    return admin_helpers.validate_match_data(match_id, action, payload, team_fields=_TEAM_FIELDS)


def _matches_from_df(df):
    return admin_helpers.matches_from_df(
        df,
        normalize_admin_date=admin_helpers.normalize_admin_date,
        identifier_from_match_data=admin_helpers.identifier_from_match_data,
    )


def _players_ranked_by_games(df):
    return admin_helpers.players_ranked_by_games(df, _TEAM_FIELDS)


def _unlocked_tournament_keys():
    return admin_helpers.unlocked_tournament_keys(session.get("tournament_edit_keys", []))


def _set_unlocked_tournament_keys(keys):
    admin_helpers.set_unlocked_tournament_keys(session, keys)


def _set_admin_feedback(level, message):
    admin_helpers.set_admin_feedback(session, level, message)


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------

@bp.route("/<slug>/")
@bp.route("/<slug>")
def home(slug: str):
    return ranking_page_response(
        current_ui_config=_current_ui_config,
        fetch_base_dataframe=_fetch_base_dataframe,
        normalize_filter_mode=web_helpers.normalize_filter_mode,
        filter_rankings=_filter_rankings,
        describe_period=web_helpers.describe_period,
        format_ranking=web_helpers.format_ranking,
        with_index=web_helpers.with_index,
        build_highlights=web_helpers.build_highlights,
        render_template=render_template,
    )


@bp.route("/<slug>/infos")
def infos(slug: str):
    return infos_page_response(
        fetch_base_dataframe=_fetch_base_dataframe,
        build_infos_summary=info_helpers.build_infos_summary,
        render_template=render_template,
    )


@bp.route("/<slug>/awards")
def awards(slug: str):
    return awards_page_response(
        available_awards_years=available_awards_years,
        safe_int=web_helpers.safe_int,
        build_awards_data=data_helpers.build_awards_data,
        build_awards_champions=data_helpers.build_awards_champions,
        render_template=render_template,
    )


@bp.route("/<slug>/campeonato", methods=["GET", "POST"])
def campeonato(slug: str):
    gid = _group_id()
    return championship_page_response(
        available_championship_keys=lambda: available_championship_keys(group_id=gid),
        get_championship_edit_password=lambda key: get_championship_edit_password(key, group_id=gid),
        unlocked_tournament_keys=_unlocked_tournament_keys,
        set_unlocked_tournament_keys=_set_unlocked_tournament_keys,
        save_match_score=lambda k, m, a, b: save_match_score(k, m, a, b, group_id=gid),
        get_championship_view=lambda key: get_championship_view(key, group_id=gid),
        redirect_to_championship=lambda key: redirect(
            url_for("group.campeonato", slug=slug, championship=key)
        ),
        render_template=render_template,
    )


@bp.route("/<slug>/config", methods=["GET", "POST"])
def config_page(slug: str):
    return config_page_response(
        get_config=get_config,
        update_config=update_config,
        safe_float=web_helpers.safe_float,
        safe_int=web_helpers.safe_int,
        render_template=render_template,
    )


@bp.route("/<slug>/jogos")
def jogos(slug: str):
    return render_games_page(
        current_ui_config=_current_ui_config,
        fetch_base_dataframe=_fetch_base_dataframe,
        filter_interval=data_helpers.filtrar_por_intervalo,
        filter_data=filtrar_dados,
        format_matches=web_helpers.format_matches,
        describe_games=web_helpers.describe_games,
        render_template=render_template,
    )


@bp.route("/<slug>/detalhamento")
def detalhamento(slug: str):
    return detalhamento_page_response(
        fetch_base_dataframe=_fetch_base_dataframe,
        jogadores_disponiveis=_jogadores_disponiveis,
        calcular_metricas_jogador=calcular_metricas_jogador,
        calcular_metricas_dupla=calcular_metricas_dupla,
        calcular_metricas_gerais=calcular_metricas_gerais,
        render_template=render_template,
    )


@bp.route("/<slug>/versus")
def versus(slug: str):
    return versus_page_response(
        fetch_base_dataframe=_fetch_base_dataframe,
        jogadores_disponiveis=_jogadores_disponiveis,
        parceiros_por_jogador=_parceiros_por_jogador,
        oponentes_por_dupla_jogadores=_oponentes_por_dupla_jogadores,
        oponentes_por_jogador=_oponentes_por_jogador,
        estatisticas_dupla=_estatisticas_dupla,
        confronto_direto_duplas=_confronto_direto_duplas,
        estatisticas_jogador_individual=_estatisticas_jogador_individual,
        confronto_direto=_confronto_direto,
        render_template=render_template,
    )


@bp.route("/<slug>/admin", methods=["GET", "POST"])
def admin(slug: str):
    gid = _group_id()
    return admin_page_response(
        admin_password=ADMIN_PASSWORD,
        entry_password=MATCH_ENTRY_PASSWORD,
        set_admin_feedback=_set_admin_feedback,
        available_championship_keys=lambda: available_championship_keys(group_id=gid),
        get_championship_edit_password=lambda key: get_championship_edit_password(key, group_id=gid),
        unlocked_tournament_keys=_unlocked_tournament_keys,
        set_unlocked_tournament_keys=_set_unlocked_tournament_keys,
        reset_cache=_reset_cache,
        fetch_base_dataframe=_fetch_base_dataframe,
        add_player=lambda name: add_player_to_group(gid, name),
        parse_bulk_line=web_helpers.parse_bulk_line,
        validate_registered_players=_validate_registered_players,
        validate_match_data=_validate_match_data,
        insert_match=lambda payload: insert_match(payload, group_id=gid),
        serialize_payload=lambda payload: web_helpers.serialize_payload(payload, _TEAM_FIELDS),
        parse_form_date=admin_helpers.parse_form_date,
        update_match=lambda match_id, payload, id_field: update_match(
            match_id, payload, id_field=id_field
        ),
        delete_match=lambda match_id, id_field: delete_match(match_id, id_field=id_field),
        rename_player=lambda old, new: rename_player(old, new),
        merge_players=lambda kept, removed: merge_players(kept, removed),
        is_valid_identifier=admin_helpers.is_valid_identifier,
        matches_from_df=_matches_from_df,
        registered_players=_registered_players,
        get_championship_view=lambda key: get_championship_view(key, group_id=gid),
        save_match_score=lambda k, m, a, b: save_match_score(k, m, a, b, group_id=gid),
        redirect_to_admin=lambda **params: redirect(
            url_for("group.admin", slug=slug, **params)
        ),
        render_template=render_template,
        team_fields=_TEAM_FIELDS,
    )


@bp.route("/<slug>/admin/convite", methods=["GET", "POST"])
def convite_gerar(slug: str):
    """Página do admin para gerar link de convite."""
    from flask import request as req
    gid = _group_id()
    group_name = g.current_group.get("name", slug)

    if not session.get("admin_authenticated"):
        return redirect(url_for("group.admin", slug=slug))

    invite_url = None
    error = None
    success = None

    # Lista todos os jogadores com partidas registradas no grupo
    players = _jogadores_disponiveis(_fetch_base_dataframe())

    if req.method == "POST":
        player_name = req.form.get("player_name", "").strip() or None

        if not player_name:
            error = "Selecione um jogador."
        else:
            try:
                invitation = create_invitation(group_id=gid, name=player_name)
                token = invitation["token"]
                invite_url = req.host_url.rstrip("/") + f"/convite/{token}"
                success = "Link de convite gerado. Copie e envie pelo WhatsApp."
            except Exception as exc:
                error = f"Erro ao gerar convite: {exc}"

    return render_template(
        "convite_gerar.html",
        slug=slug,
        players=players,
        invite_url=invite_url,
        error=error,
        success=success,
        base_url=f"/g/{slug}",
        active_page="admin",
    )


@bp.route("/<slug>/admin/torneio/novo", methods=["GET", "POST"])
def torneio_novo(slug: str):
    from flask import flash, request as req
    gid = _group_id()
    error = None

    if req.method == "POST":
        title = req.form.get("title", "").strip()
        t_slug = req.form.get("slug", "").strip()
        description = req.form.get("description", "").strip()
        edit_password = req.form.get("edit_password", "").strip()
        admin_password_input = req.form.get("admin_password", "").strip()
        teams_raw = req.form.get("teams", "").strip()
        num_groups = int(req.form.get("num_groups", "4") or "4")

        if admin_password_input != ADMIN_PASSWORD:
            error = "Senha de admin incorreta."
        elif not title:
            error = "Título é obrigatório."
        elif not t_slug:
            error = "Slug é obrigatório."
        elif not teams_raw:
            error = "Adicione pelo menos 2 times."
        else:
            team_names = [l.strip() for l in teams_raw.splitlines() if l.strip()]
            if len(team_names) < 2:
                error = "Adicione pelo menos 2 times."
            else:
                teams = [{"id": f"T{i+1}", "name": name} for i, name in enumerate(team_names)]
                team_by_name = {t["name"].casefold(): t["id"] for t in teams}

                groups = []
                for g_idx in range(num_groups):
                    g_raw = req.form.get(f"group_{g_idx}", "").strip()
                    g_name = req.form.get(f"group_{g_idx}_name", f"Grupo {g_idx+1}").strip()
                    g_team_names = [l.strip() for l in g_raw.splitlines() if l.strip()]
                    team_ids = []
                    unknown = []
                    for tname in g_team_names:
                        tid = team_by_name.get(tname.casefold())
                        if tid:
                            team_ids.append(tid)
                        else:
                            unknown.append(tname)
                    if unknown:
                        error = f"Times não encontrados no grupo {g_name}: {', '.join(unknown)}"
                        break
                    if team_ids:
                        groups.append({"id": f"G{g_idx+1}", "name": g_name, "team_ids": team_ids})

                if not error:
                    config = {"teams": teams, "groups": groups}
                    try:
                        create_championship(
                            group_id=gid,
                            slug=t_slug,
                            title=title,
                            description=description,
                            edit_password=edit_password,
                            config=config,
                        )
                        return redirect(url_for("group.campeonato", slug=slug, championship=t_slug))
                    except Exception as exc:
                        error = f"Erro ao salvar: {exc}"

    return render_template(
        "torneio_novo.html",
        slug=slug,
        error=error,
        base_url=f"/g/{slug}",
        active_page="campeonato",
    )


@bp.route("/<slug>/login", methods=["GET", "POST"])
def login(slug: str):
    from flask import request as req
    import bcrypt

    if session.get("player_user_id"):
        return redirect(url_for("group.perfil", slug=slug))

    error = None

    if req.method == "POST":
        email = req.form.get("email", "").strip().lower()
        password = req.form.get("password", "")

        if not email or not password:
            error = "Preencha email e senha."
        else:
            try:
                from src.redinha_stats.config.settings import SUPABASE_SERVICE_KEY, SUPABASE_URL
                from supabase import create_client
                client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
                resp = client.table("users").select("id,name,password_hash").eq("email", email).limit(1).execute()
                user = (resp.data or [None])[0]

                if not user or not user.get("password_hash"):
                    error = "Email ou senha incorretos."
                elif not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
                    error = "Email ou senha incorretos."
                else:
                    session["player_user_id"] = user["id"]
                    session["player_name"] = user["name"]
                    return redirect(url_for("group.perfil", slug=slug))
            except Exception as exc:
                error = f"Erro ao autenticar: {exc}"

    return render_template(
        "player_login.html",
        error=error,
        base_url=f"/g/{slug}",
        active_page=None,
    )


@bp.route("/<slug>/logout")
def logout(slug: str):
    session.pop("player_user_id", None)
    session.pop("player_name", None)
    return redirect(url_for("group.home", slug=slug))


@bp.route("/<slug>/perfil")
def perfil(slug: str):
    player_name = session.get("player_name")
    if not player_name:
        return redirect(url_for("group.login", slug=slug))

    from urllib.parse import urlencode
    params = urlencode({"tipo": "Jogador", "jogador": player_name})
    return redirect(f"/g/{slug}/detalhamento?{params}")


@bp.route("/<slug>/api/ranking")
def api_ranking(slug: str):
    return ranking_api_response(
        filter_rankings=_filter_rankings,
        describe_period=web_helpers.describe_period,
        format_ranking=web_helpers.format_ranking,
    )


@bp.route("/<slug>/_oculto/jogadores")
def hidden_players(slug: str):
    from flask import current_app
    return hidden_players_response(
        fetch_base_dataframe=_fetch_base_dataframe,
        players_ranked_by_games=_players_ranked_by_games,
        response_factory=lambda content: current_app.response_class(
            content, mimetype="text/plain; charset=utf-8"
        ),
    )
