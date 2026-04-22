from __future__ import annotations
from batch_endpoints import bp as batch_bp

"""Aplicação Flask para exibir o ranking em HTML estático."""

from collections import Counter
import os
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from src.redinha_stats.config.app_settings import get_config, update_config
from awards import available_awards_years, get_awards_for_year
from championship import (
    available_championship_keys,
    get_championship_edit_password,
    get_championship_view,
    save_match_score,
)
from src.redinha_stats.config.settings import ADMIN_PASSWORD, MATCH_ENTRY_PASSWORD
from detalhamento import calcular_metricas_dupla, calcular_metricas_gerais, calcular_metricas_jogador
from src.redinha_stats.domain.matches.extraction import get_matches
from src.redinha_stats.domain.matches.preparation import preparar_dataframe
from src.redinha_stats.domain.matches.processing import (
    filtrar_dados,
    preparar_dados_duplas,
    preparar_dados_individuais,
)
from src.redinha_stats.infrastructure.local.player_registry_store import (
    add_player,
    load_registered_players,
)
from src.redinha_stats.infrastructure.supabase.matches_repository import (
    delete_match,
    insert_match,
    update_match,
)
from src.redinha_stats.config.ui_config import get_ui_config
from src.redinha_stats.web import admin_helpers
from src.redinha_stats.web import data_helpers
from src.redinha_stats.web import helpers as web_helpers
from src.redinha_stats.web import info_helpers
from src.redinha_stats.web.routes.api import (
    hidden_players_response,
    ranking_api_response,
)
from src.redinha_stats.web.routes.games import render_games_page
from src.redinha_stats.web.routes.pages import (
    championship_page_response,
    ranking_page_response,
    awards_page_response,
    infos_page_response,
)
from src.redinha_stats.web.routes.config_page import config_page_response
from src.redinha_stats.web.routes.details import detalhamento_page_response
from src.redinha_stats.web.routes.versus import versus_page_response
from src.redinha_stats.web.routes.admin import admin_page_response
from src.redinha_stats.web import versus_helpers

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")
app.register_blueprint(batch_bp)

from src.redinha_stats.web.group_blueprint import bp as group_bp  # noqa: E402
from src.redinha_stats.web.superadmin_blueprint import bp as superadmin_bp  # noqa: E402
app.register_blueprint(group_bp)
app.register_blueprint(superadmin_bp)


def _normalize_filter_mode(value: str | None) -> str:
    """Normaliza nomes de filtros legados para valores internos estaveis."""

    normalized = (value or "Dia").strip()
    if normalized == "Dias":
        return "Dia"
    if normalized in {"Mes/Ano", "Mês/Ano", "MÃªs/Ano"}:
        return "Mes/Ano"
    return normalized

_current_ui_config = data_helpers.current_ui_config
_build_awards_data = data_helpers.build_awards_data
_format_champion_names = data_helpers.format_champion_names
_build_awards_champions = data_helpers.build_awards_champions


# --------------------------------- Dados ----------------------------------
_fetch_base_dataframe = data_helpers.fetch_base_dataframe
_filtrar_por_intervalo = data_helpers.filtrar_por_intervalo
_excluded_players = data_helpers.excluded_players
_filter_rankings = data_helpers.filter_rankings


def _descricao_periodo(
    modo: str,
    periodo: str | None,
    ano: str | None,
    mes: str | None,
    inicio: str | None,
    fim: str | None,
    data: str | None,
) -> str:
    if modo == "Ano" and ano:
        return f"Ano {ano}"
    if modo == "Mes/Ano" and mes:
        return mes
    if modo == "Intervalo":
        if inicio or fim:
            return f"Intervalo {inicio or '...'} a {fim or '...'}"
        return "Intervalo personalizado"
    if modo == "Data" and data:
        return f"Dia {data}"
    return periodo or "Todos"


def _format_ranking(df: pd.DataFrame, nome_col: str) -> List[Dict[str, str]]:
    medalhas = ["\U0001F947", "\U0001F948", "\U0001F949"]
    linhas: List[Dict[str, str]] = []
    total_linhas = len(df)

    for idx, linha in df.iterrows():
        if idx < len(medalhas):
            posicao = medalhas[idx]
        else:
            posicao = "\U0001F631" if idx == total_linhas - 1 else f"{idx + 1:02d}"
        linhas.append(
            {
                "posicao": posicao,
                "nome": linha.get(nome_col, "-"),
                "score": f"{int(round(linha.get('aproveitamento', 0)))}%",
                "vitorias": int(linha.get("vitórias", 0)),
                "derrotas": int(linha.get("derrotas", 0)),
                "saldo": int(linha.get("saldo", 0)),
                "jogos": int(linha.get("jogos", 0)),
            }
        )

    return linhas


def _build_highlights(linhas: List[Dict[str, str]]) -> List[Dict[str, str]]:
    medals = ["\U0001F947", "\U0001F948", "\U0001F949"]
    destaques = []

    for idx, linha in enumerate(linhas[:3]):
        destaque = {**linha}
        destaque["medal"] = medals[idx] if idx < len(medals) else ""
        destaques.append(destaque)

    return destaques


def _jogadores_disponiveis(df: pd.DataFrame) -> List[str]:
    jogadores = {
        j
        for j in df[["winner1", "winner2", "loser1", "loser2"]].values.ravel()
        if isinstance(j, str) and "Outro" not in j
    }

    excluidos = _excluded_players()
    return sorted(jogadores - excluidos)


def _duplas_disponiveis(df: pd.DataFrame) -> List[str]:
    duplas = set(df["dupla_winner"].tolist() + df["dupla_loser"].tolist())
    duplas = {d for d in duplas if isinstance(d, str) and "Outro" not in d}
    return sorted(duplas)


def _oponentes_por_jogador(df: pd.DataFrame) -> Dict[str, List[str]]:
    jogadores = set(_jogadores_disponiveis(df))
    oponentes: Dict[str, set[str]] = {j: set() for j in jogadores}

    def _limpar_nome(valor: Any) -> str | None:
        if not isinstance(valor, str):
            return None
        nome = valor.strip()
        if not nome or "Outro" in nome or nome not in jogadores:
            return None
        return nome

    for _, linha in df.iterrows():
        w1 = _limpar_nome(linha.get("winner1"))
        w2 = _limpar_nome(linha.get("winner2"))
        l1 = _limpar_nome(linha.get("loser1"))
        l2 = _limpar_nome(linha.get("loser2"))

        winners = [p for p in (w1, w2) if p]
        losers = [p for p in (l1, l2) if p]

        for vencedor in winners:
            oponentes[vencedor].update(losers)
        for perdedor in losers:
            oponentes[perdedor].update(winners)

    return {j: sorted(lista) for j, lista in oponentes.items()}


def _parceiros_por_jogador(df: pd.DataFrame) -> Dict[str, List[str]]:
    jogadores = set(_jogadores_disponiveis(df))
    parceiros: Dict[str, set[str]] = {j: set() for j in jogadores}

    for _, row in df.iterrows():
        duplas_partida = [
            [row.get("winner1"), row.get("winner2")],
            [row.get("loser1"), row.get("loser2")],
        ]
        for jogador_a, jogador_b in duplas_partida:
            if not jogador_a or not jogador_b:
                continue
            if jogador_a not in jogadores or jogador_b not in jogadores:
                continue
            parceiros[jogador_a].add(jogador_b)
            parceiros[jogador_b].add(jogador_a)

    return {j: sorted(lista) for j, lista in parceiros.items()}


def _oponentes_por_dupla_jogadores(df: pd.DataFrame) -> Dict[str, List[str]]:
    jogadores = set(_jogadores_disponiveis(df))
    duplas = set(_duplas_disponiveis(df))
    oponentes: Dict[str, set[str]] = {dupla: set() for dupla in duplas}

    for _, row in df.iterrows():
        dupla_winner = row.get("dupla_winner")
        dupla_loser = row.get("dupla_loser")
        winners = [row.get("winner1"), row.get("winner2")]
        losers = [row.get("loser1"), row.get("loser2")]

        if dupla_winner in duplas:
            oponentes[dupla_winner].update([j for j in losers if j in jogadores])
        if dupla_loser in duplas:
            oponentes[dupla_loser].update([j for j in winners if j in jogadores])

    return {dupla: sorted(lista) for dupla, lista in oponentes.items()}


def _oponentes_por_dupla(df: pd.DataFrame) -> Dict[str, List[str]]:
    duplas = set(_duplas_disponiveis(df))
    oponentes: Dict[str, set[str]] = {d: set() for d in duplas}

    for _, linha in df.iterrows():
        vencedor = linha.get("dupla_winner")
        perdedor = linha.get("dupla_loser")

        if vencedor not in duplas or perdedor not in duplas:
            continue

        oponentes[vencedor].add(perdedor)
        oponentes[perdedor].add(vencedor)

    return {dupla: sorted(lista) for dupla, lista in oponentes.items()}


def _estatisticas_jogador_individual(df: pd.DataFrame, jogador: str) -> Dict[str, int | float]:
    return versus_helpers.estatisticas_jogador_individual(df, jogador, excluded_players=_excluded_players())


def _estatisticas_dupla(df: pd.DataFrame, dupla: str) -> Dict[str, int | float]:
    return versus_helpers.estatisticas_dupla(df, dupla)


def _confronto_direto(df: pd.DataFrame, jogador1: str, jogador2: str) -> Dict[str, Any]:
    return versus_helpers.confronto_direto(df, jogador1, jogador2)


def _confronto_direto_duplas(df: pd.DataFrame, dupla1: str, dupla2: str) -> Dict[str, Any]:
    return versus_helpers.confronto_direto_duplas(df, dupla1, dupla2)

# ------------------------------- Admin ------------------------------------
_TEAM_FIELDS = ("winner1", "winner2", "loser1", "loser2")

_normalize_admin_date = admin_helpers.normalize_admin_date
_is_valid_identifier = admin_helpers.is_valid_identifier
_identifier_from_match_data = admin_helpers.identifier_from_match_data
_players_from_df = lambda df: admin_helpers.players_from_df(df, _TEAM_FIELDS)
_players_ranked_by_games = lambda df: admin_helpers.players_ranked_by_games(df, _TEAM_FIELDS)
_registered_players = lambda df: admin_helpers.registered_players(
    df,
    team_fields=_TEAM_FIELDS,
    load_registered_players=load_registered_players,
    excluded_players=_excluded_players,
)
_matches_from_df = lambda df: admin_helpers.matches_from_df(
    df,
    normalize_admin_date=_normalize_admin_date,
    identifier_from_match_data=_identifier_from_match_data,
)


def _serialize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    data_value = payload.get("date")
    serialized: Dict[str, Any] = {field: payload.get(field, "").strip() for field in _TEAM_FIELDS}
    if isinstance(data_value, date):
        serialized["date"] = data_value.isoformat()
    else:
        serialized["date"] = str(data_value) if data_value is not None else None
    if "score" in payload:
        serialized["score"] = payload.get("score")
    return serialized


def _parse_bulk_line(line: str) -> Dict[str, str] | None:
    """Converte uma linha no formato "a e b x c e d" em um dicionário de campos."""

    import re

    pattern = re.compile(
        r"^(?P<w1>.+?)\s+e\s+(?P<w2>.+?)\s+x\s+(?P<l1>.+?)\s+e\s+(?P<l2>.+)$",
        flags=re.IGNORECASE,
    )
    match = pattern.match(line.strip())
    if not match:
        return None

    return {
        "winner1": match.group("w1").strip(),
        "winner2": match.group("w2").strip(),
        "loser1": match.group("l1").strip(),
        "loser2": match.group("l2").strip(),
    }


_validate_registered_players = lambda players: admin_helpers.validate_registered_players(
    players,
    registered_players=_registered_players(_fetch_base_dataframe()),
)
_validate_match_data = lambda match_id, action, payload: admin_helpers.validate_match_data(
    match_id,
    action,
    payload,
    team_fields=_TEAM_FIELDS,
)


def _reset_cache() -> None:
    _fetch_base_dataframe.cache_clear()

# -------------------------------- Rotas -----------------------------------
@app.route("/")
def home():
    return ranking_page_response(
        current_ui_config=lambda: _current_ui_config(),
        fetch_base_dataframe=lambda: _fetch_base_dataframe(),
        normalize_filter_mode=lambda value: _normalize_filter_mode(value),
        filter_rankings=lambda *args: _filter_rankings(*args),
        describe_period=lambda *args: _descricao_periodo(*args),
        format_ranking=lambda *args: _format_ranking(*args),
        with_index=lambda linhas: dados_with_index(linhas),
        build_highlights=lambda linhas: _build_highlights(linhas),
        render_template=lambda template_name, **context: render_template(template_name, **context),
    )


@app.route("/infos")
def infos():
    return infos_page_response(
        fetch_base_dataframe=lambda: _fetch_base_dataframe(),
        build_infos_summary=lambda df: _resumo_infos(df),
        render_template=lambda template_name, **context: render_template(template_name, **context),
    )


@app.route("/awards")
def awards():
    return awards_page_response(
        available_awards_years=lambda: available_awards_years(),
        safe_int=lambda value, default: _safe_int(value, default),
        build_awards_data=lambda year: _build_awards_data(year),
        build_awards_champions=lambda awards: _build_awards_champions(awards),
        render_template=lambda template_name, **context: render_template(template_name, **context),
    )


@app.route("/campeonato", methods=["GET", "POST"])
def campeonato():
    return championship_page_response(
        available_championship_keys=lambda: available_championship_keys(),
        get_championship_edit_password=lambda key: get_championship_edit_password(key),
        unlocked_tournament_keys=lambda: _unlocked_tournament_keys(),
        set_unlocked_tournament_keys=lambda keys: _set_unlocked_tournament_keys(keys),
        save_match_score=lambda championship_key, match_id, score_a, score_b: save_match_score(
            championship_key, match_id, score_a, score_b
        ),
        get_championship_view=lambda key: get_championship_view(key),
        redirect_to_championship=lambda selected_key: redirect(
            url_for("campeonato", championship=selected_key)
        ),
        render_template=lambda template_name, **context: render_template(template_name, **context),
    )


def _safe_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


@app.route("/config", methods=["GET", "POST"])
def config_page():
    return config_page_response(
        get_config=lambda: get_config(),
        update_config=lambda **kwargs: update_config(**kwargs),
        safe_float=lambda value, default: _safe_float(value, default),
        safe_int=lambda value, default: _safe_int(value, default),
        render_template=lambda template_name, **context: render_template(template_name, **context),
    )


def _descricao_jogos(
    modo: str,
    periodo: str | None,
    ano: str | None,
    mes: str | None,
    data: str | None,
    inicio: str | None,
    fim: str | None,
) -> str:
    if modo == "Mês/Ano" and mes:
        return mes
    if modo == "Ano" and ano:
        return f"Ano {ano}"
    if modo == "Data" and data:
        return f"Dia {data}"
    if modo == "Intervalo":
        if inicio or fim:
            return f"Intervalo {inicio or '...'} a {fim or '...'}"
        return "Intervalo personalizado"
    return periodo or "Todos"


def _formatar_partidas(df: pd.DataFrame) -> List[Dict[str, str]]:
    linhas: List[Dict[str, str]] = []

    for indice, linha in df.iterrows():
        if isinstance(indice, pd.Timestamp) and not pd.isna(indice):
            data_legivel = indice.date().isoformat()
        else:
            data_legivel = str(indice)

        vencedor1 = linha.get("winner1", "")
        vencedor2 = linha.get("winner2", "")
        perdedor1 = linha.get("loser1", "")
        perdedor2 = linha.get("loser2", "")

        score = linha.get("score", "")
        if pd.isna(score) or score == "nan":
            score = ""

        linhas.append(
            {
                "data": data_legivel,
                "vencedores": " & ".join(filter(None, [vencedor1, vencedor2])) or "-",
                "perdedores": " & ".join(filter(None, [perdedor1, perdedor2])) or "-",
                "score": score or "-",
            }
        )

    return linhas


@app.route("/jogos")
def jogos():
    return render_games_page(
        current_ui_config=lambda: _current_ui_config(),
        fetch_base_dataframe=lambda: _fetch_base_dataframe(),
        filter_interval=lambda df, inicio, fim: _filtrar_por_intervalo(df, inicio, fim),
        filter_data=lambda df, modo, valor: filtrar_dados(df, modo, valor),
        format_matches=lambda df: _formatar_partidas(df),
        describe_games=lambda *args: _descricao_jogos(*args),
        render_template=lambda template_name, **context: render_template(template_name, **context),
    )


@app.route("/detalhamento")
def detalhamento():
    return detalhamento_page_response(
        fetch_base_dataframe=lambda: _fetch_base_dataframe(),
        jogadores_disponiveis=lambda df: _jogadores_disponiveis(df),
        calcular_metricas_jogador=lambda df, jogador: calcular_metricas_jogador(df, jogador),
        calcular_metricas_dupla=lambda df, jogador1, jogador2: calcular_metricas_dupla(df, jogador1, jogador2),
        calcular_metricas_gerais=lambda df: calcular_metricas_gerais(df),
        render_template=lambda template_name, **context: render_template(template_name, **context),
    )


@app.route("/versus")
def versus():
    return versus_page_response(
        fetch_base_dataframe=lambda: _fetch_base_dataframe(),
        jogadores_disponiveis=lambda df: _jogadores_disponiveis(df),
        parceiros_por_jogador=lambda df: _parceiros_por_jogador(df),
        oponentes_por_dupla_jogadores=lambda df: _oponentes_por_dupla_jogadores(df),
        oponentes_por_jogador=lambda df: _oponentes_por_jogador(df),
        estatisticas_dupla=lambda df, dupla: _estatisticas_dupla(df, dupla),
        confronto_direto_duplas=lambda df, dupla1, dupla2: _confronto_direto_duplas(df, dupla1, dupla2),
        estatisticas_jogador_individual=lambda df, jogador: _estatisticas_jogador_individual(df, jogador),
        confronto_direto=lambda df, jogador1, jogador2: _confronto_direto(df, jogador1, jogador2),
        render_template=lambda template_name, **context: render_template(template_name, **context),
    )


_parse_form_date = admin_helpers.parse_form_date
_set_admin_feedback = lambda level, message: admin_helpers.set_admin_feedback(session, level, message)
_unlocked_tournament_keys = lambda: admin_helpers.unlocked_tournament_keys(
    session.get("tournament_edit_keys", [])
)
_set_unlocked_tournament_keys = lambda keys: admin_helpers.set_unlocked_tournament_keys(
    session, keys
)


@app.route("/admin", methods=["GET", "POST"])
def admin():
    return admin_page_response(
        admin_password=ADMIN_PASSWORD,
        entry_password=MATCH_ENTRY_PASSWORD,
        set_admin_feedback=lambda level, message: _set_admin_feedback(level, message),
        available_championship_keys=lambda: available_championship_keys(),
        get_championship_edit_password=lambda key: get_championship_edit_password(key),
        unlocked_tournament_keys=lambda: _unlocked_tournament_keys(),
        set_unlocked_tournament_keys=lambda keys: _set_unlocked_tournament_keys(keys),
        reset_cache=lambda: _reset_cache(),
        fetch_base_dataframe=lambda: _fetch_base_dataframe(),
        add_player=lambda name: add_player(name),
        parse_bulk_line=lambda line: _parse_bulk_line(line),
        validate_registered_players=lambda players: _validate_registered_players(players),
        validate_match_data=lambda match_id, action, payload: _validate_match_data(match_id, action, payload),
        insert_match=lambda payload: insert_match(payload),
        serialize_payload=lambda payload: _serialize_payload(payload),
        parse_form_date=lambda value: _parse_form_date(value),
        update_match=lambda match_id, payload, id_field: update_match(match_id, payload, id_field=id_field),
        delete_match=lambda match_id, id_field: delete_match(match_id, id_field=id_field),
        is_valid_identifier=lambda value: _is_valid_identifier(value),
        matches_from_df=lambda df: _matches_from_df(df),
        registered_players=lambda df: _registered_players(df),
        get_championship_view=lambda key: get_championship_view(key),
        save_match_score=lambda championship_key, match_id, score_a, score_b: save_match_score(
            championship_key, match_id, score_a, score_b
        ),
        redirect_to_admin=lambda **params: redirect(url_for("admin", **params)),
        render_template=lambda template_name, **context: render_template(template_name, **context),
        team_fields=_TEAM_FIELDS,
    )


@app.route("/api/ranking")
def api_ranking():
    return ranking_api_response(
        filter_rankings=lambda *args: _filter_rankings(*args),
        describe_period=lambda *args: _descricao_periodo(*args),
        format_ranking=lambda *args: _format_ranking(*args),
    )


@app.route("/_oculto/jogadores")
def hidden_players():
    return hidden_players_response(
        fetch_base_dataframe=lambda: _fetch_base_dataframe(),
        players_ranked_by_games=lambda df: _players_ranked_by_games(df),
        response_factory=lambda content: app.response_class(
            content, mimetype="text/plain; charset=utf-8"
        ),
    )


def dados_with_index(linhas: List[Dict[str, str]]) -> List[Dict[str, str]]:
    for idx, linha in enumerate(linhas, start=1):
        linha["index"] = idx
    return linhas


_resumo_infos = info_helpers.build_infos_summary


# Web helper implementations extracted from this module. Keep the original
# names bound for compatibility while routing the behavior through the new
# package.
_normalize_filter_mode = web_helpers.normalize_filter_mode
_descricao_periodo = web_helpers.describe_period
_format_ranking = web_helpers.format_ranking
_build_highlights = web_helpers.build_highlights
_safe_int = web_helpers.safe_int
_safe_float = web_helpers.safe_float
_descricao_jogos = web_helpers.describe_games
_formatar_partidas = web_helpers.format_matches
dados_with_index = web_helpers.with_index
_serialize_payload = lambda payload: web_helpers.serialize_payload(payload, _TEAM_FIELDS)
_parse_bulk_line = web_helpers.parse_bulk_line


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
