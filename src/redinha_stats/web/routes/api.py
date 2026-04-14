from __future__ import annotations

from typing import Any, Callable

from flask import jsonify, request


def ranking_api_response(
    *,
    filter_rankings: Callable[..., Any],
    describe_period: Callable[..., str],
    format_ranking: Callable[..., list[dict[str, str]]],
):
    periodo = request.args.get("periodo", "90 dias")
    modo = request.args.get("modo", "Dia")
    if modo == "Dias":
        modo = "Dia"
    ano = request.args.get("ano")
    mes = request.args.get("mes")
    inicio = request.args.get("inicio")
    fim = request.args.get("fim")

    df, jogadores, duplas = filter_rankings(modo, periodo, ano, mes, inicio, fim, None)
    periodo_legenda = describe_period(modo, periodo, ano, mes, inicio, fim, None)

    return jsonify(
        {
            "periodo": periodo_legenda,
            "periodo_param": periodo,
            "modo": modo,
            "intervalo": {"inicio": inicio, "fim": fim} if modo == "Intervalo" else None,
            "total_partidas": len(df),
            "jogadores": format_ranking(jogadores, "jogadores"),
            "duplas": format_ranking(duplas, "duplas"),
        }
    )


def hidden_players_response(
    *,
    fetch_base_dataframe: Callable[[], Any],
    players_ranked_by_games: Callable[[Any], list[str]],
    response_factory: Callable[[str], Any],
):
    df = fetch_base_dataframe()
    players = players_ranked_by_games(df)
    output_format = request.args.get("formato", "json").lower()

    if output_format == "txt":
        content = "\n".join(players)
        return response_factory(content)

    return jsonify({"jogadores": players})
