from __future__ import annotations

from datetime import date
from typing import Any, Callable

import pandas as pd
from flask import request, session


def ranking_page_response(
    *,
    current_ui_config: Callable[[], Any],
    fetch_base_dataframe: Callable[[], Any],
    normalize_filter_mode: Callable[[str | None], str],
    filter_rankings: Callable[..., Any],
    describe_period: Callable[..., str],
    format_ranking: Callable[..., list[dict[str, str]]],
    with_index: Callable[[list[dict[str, str]]], list[dict[str, str]]],
    build_highlights: Callable[[list[dict[str, str]]], list[dict[str, str]]],
    render_template: Callable[..., Any],
) -> Any:
    ui_config = current_ui_config()
    periodos_disponiveis = list(ui_config.ranking_periods)

    modo = normalize_filter_mode(request.args.get("modo", "Dia"))
    periodo = request.args.get("periodo", ui_config.default_ranking_period)
    inicio = request.args.get("inicio")
    fim = request.args.get("fim")
    data_escolhida = request.args.get("data")

    base_df = fetch_base_dataframe()
    datas_index = pd.to_datetime(base_df.index, errors="coerce")
    anos_disponiveis = sorted({str(int(dt.year)) for dt in datas_index if pd.notna(dt)})
    meses_disponiveis = sorted({dt.strftime("%Y-%m") for dt in datas_index if pd.notna(dt)}, reverse=True)
    datas_disponiveis = sorted(
        {dt.normalize().date().isoformat() for dt in datas_index if pd.notna(dt)},
        reverse=True,
    )

    ano = request.args.get("ano", anos_disponiveis[-1] if anos_disponiveis else None)
    mes = request.args.get("mes", meses_disponiveis[-1] if meses_disponiveis else None)

    if modo == "Dia" and periodo not in periodos_disponiveis:
        periodo = "1 dia"
    if modo == "Ano" and ano not in anos_disponiveis:
        ano = anos_disponiveis[-1] if anos_disponiveis else None
    if modo == "Mes/Ano" and mes not in meses_disponiveis:
        mes = meses_disponiveis[-1] if meses_disponiveis else None
    if modo == "Data" and data_escolhida not in datas_disponiveis:
        data_escolhida = datas_disponiveis[0] if datas_disponiveis else None

    df, jogadores, duplas = filter_rankings(
        modo, periodo, ano, mes, inicio, fim, data_escolhida
    )
    periodo_legenda = describe_period(modo, periodo, ano, mes, inicio, fim, data_escolhida)

    jogadores_fmt = format_ranking(jogadores, "jogadores")
    duplas_fmt = format_ranking(duplas, "duplas")
    destaques = build_highlights(jogadores_fmt)

    return render_template(
        "ranking.html",
        active_page="ranking",
        periodo_legenda=periodo_legenda,
        periodo_escolhido=periodo,
        modo=modo,
        periodos=periodos_disponiveis,
        datas=datas_disponiveis,
        data_selecionada=data_escolhida,
        anos=anos_disponiveis,
        ano_selecionado=ano,
        meses=meses_disponiveis,
        mes_selecionado=mes,
        inicio=inicio,
        fim=fim,
        jogos_total=len(df),
        jogos_filtrados=len(df),
        jogadores=with_index(jogadores_fmt),
        duplas=with_index(duplas_fmt),
        destaques=destaques,
    )


def infos_page_response(
    *,
    fetch_base_dataframe: Callable[[], Any],
    build_infos_summary: Callable[[Any], dict[str, Any]],
    render_template: Callable[..., Any],
) -> Any:
    df = fetch_base_dataframe()
    infos_payload = build_infos_summary(df)

    return render_template(
        "infos.html",
        active_page="infos",
        **infos_payload,
    )


def awards_page_response(
    *,
    available_awards_years: Callable[[], list[int]],
    safe_int: Callable[[str | None, int], int],
    build_awards_data: Callable[[int], list[dict[str, Any]]],
    build_awards_champions: Callable[[list[dict[str, Any]]], dict[str, Any]],
    render_template: Callable[..., Any],
) -> Any:
    available_years = available_awards_years()
    default_year = available_years[0] if available_years else date.today().year
    selected_year = safe_int(request.args.get("year"), default_year)
    if selected_year not in available_years:
        selected_year = default_year

    awards_data = build_awards_data(selected_year)
    total_votes = sum(award["total_votes"] for award in awards_data)
    champions = build_awards_champions(awards_data)
    positive_awards = [
        award for award in awards_data if award["category_type"] == "positive"
    ]
    negative_awards = [
        award for award in awards_data if award["category_type"] == "negative"
    ]
    return render_template(
        "awards.html",
        awards=awards_data,
        positive_awards=positive_awards,
        negative_awards=negative_awards,
        awards_total_votes=total_votes,
        awards_years=available_years,
        awards_selected_year=selected_year,
        awards_positive_champion=champions["positive"],
        awards_negative_champion=champions["negative"],
        active_page="awards",
    )


def championship_page_response(
    *,
    available_championship_keys: Callable[[], list[str]],
    get_championship_edit_password: Callable[[str], str | None],
    unlocked_tournament_keys: Callable[[], set[str]],
    set_unlocked_tournament_keys: Callable[[set[str]], None],
    save_match_score: Callable[[str, str, str | None, str | None], None],
    get_championship_view: Callable[[str], dict[str, Any]],
    redirect_to_championship: Callable[[str], Any],
    render_template: Callable[..., Any],
) -> Any:
    feedback = session.pop("tournament_feedback", None)
    keys = available_championship_keys()
    default_key = keys[0] if keys else date.today().strftime("%Y%m%d")

    if request.method == "POST":
        action = (request.form.get("action") or "").strip()
        selected_key = (request.form.get("championship_key") or default_key).strip()
        if selected_key not in keys:
            selected_key = default_key

        if action == "unlock_tournament":
            provided_password = (request.form.get("tournament_password") or "").strip()
            expected_password = get_championship_edit_password(selected_key)
            if not expected_password:
                session["tournament_feedback"] = {
                    "status": "error",
                    "message": "Este torneio nao possui senha configurada.",
                }
            elif provided_password != expected_password:
                session["tournament_feedback"] = {
                    "status": "error",
                    "message": "Senha do torneio incorreta.",
                }
            else:
                unlocked = unlocked_tournament_keys()
                unlocked.add(selected_key)
                set_unlocked_tournament_keys(unlocked)
                session["tournament_feedback"] = {
                    "status": "success",
                    "message": "Edicao desbloqueada para este torneio nesta sessao.",
                }
            return redirect_to_championship(selected_key)

        if action == "championship_score":
            if selected_key not in unlocked_tournament_keys():
                session["tournament_feedback"] = {
                    "status": "error",
                    "message": "Desbloqueie o torneio com a senha para editar placares.",
                }
                return redirect_to_championship(selected_key)

            match_id = (request.form.get("championship_match_id") or "").strip()
            score_a_raw = request.form.get("score_a")
            score_b_raw = request.form.get("score_b")
            try:
                save_match_score(selected_key, match_id, score_a_raw, score_b_raw)
                session["tournament_feedback"] = {
                    "status": "success",
                    "message": "Placar atualizado com sucesso.",
                }
            except Exception as exc:
                session["tournament_feedback"] = {
                    "status": "error",
                    "message": f"Erro ao salvar placar: {exc}",
                }
            return redirect_to_championship(selected_key)

        if action == "delete_championship_score":
            if selected_key not in unlocked_tournament_keys():
                session["tournament_feedback"] = {
                    "status": "error",
                    "message": "Desbloqueie o torneio com a senha para apagar placares.",
                }
                return redirect_to_championship(selected_key)

            match_id = (request.form.get("championship_match_id") or "").strip()
            try:
                save_match_score(selected_key, match_id, None, None)
                session["tournament_feedback"] = {
                    "status": "success",
                    "message": "Placar apagado com sucesso.",
                }
            except Exception as exc:
                session["tournament_feedback"] = {
                    "status": "error",
                    "message": f"Erro ao apagar placar: {exc}",
                }
            return redirect_to_championship(selected_key)

    selected_key = (request.args.get("championship") or default_key).strip()
    if selected_key not in keys:
        selected_key = default_key

    payload = get_championship_view(selected_key)
    championship_can_edit = selected_key in unlocked_tournament_keys()
    is_admin_full = bool(
        session.get("admin_authenticated") and session.get("admin_role") == "full"
    )
    saved_matches = [
        m
        for m in payload.get("editable_matches", [])
        if m.get("score_a") is not None and m.get("score_b") is not None
    ]
    return render_template(
        "campeonato.html",
        active_page="campeonato",
        championship=payload,
        championship_keys=keys,
        championship_selected_key=selected_key,
        championship_can_edit=championship_can_edit,
        is_admin_full=is_admin_full,
        tournament_feedback=feedback,
        championship_saved_matches=saved_matches,
    )
