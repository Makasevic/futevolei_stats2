from __future__ import annotations

from typing import Any, Callable

import pandas as pd
from flask import request


def render_games_page(
    *,
    current_ui_config: Callable[[], Any],
    fetch_base_dataframe: Callable[[], pd.DataFrame],
    filter_interval: Callable[[pd.DataFrame, str | None, str | None], pd.DataFrame],
    filter_data: Callable[[pd.DataFrame, str, str], pd.DataFrame],
    format_matches: Callable[[pd.DataFrame], list[dict[str, str]]],
    describe_games: Callable[..., str],
    render_template: Callable[..., Any],
) -> Any:
    ui_config = current_ui_config()
    periodos_disponiveis = list(ui_config.games_periods)

    modo = request.args.get("modo", "Dia")
    if modo == "Dias":
        modo = "Dia"
    periodo = request.args.get("periodo", ui_config.default_games_period)
    data_escolhida = request.args.get("data")
    inicio = request.args.get("inicio")
    fim = request.args.get("fim")

    base_df = fetch_base_dataframe()
    datas_index = pd.to_datetime(base_df.index, errors="coerce")

    anos_disponiveis = sorted({str(int(dt.year)) for dt in datas_index if pd.notna(dt)})
    meses_disponiveis = sorted({dt.strftime("%Y-%m") for dt in datas_index if pd.notna(dt)})
    datas_disponiveis = sorted(
        {dt.normalize().date().isoformat() for dt in datas_index if pd.notna(dt)},
        reverse=True,
    )

    ano = request.args.get("ano", anos_disponiveis[-1] if anos_disponiveis else None)
    mes = request.args.get("mes", meses_disponiveis[-1] if meses_disponiveis else None)

    filtro_modo = modo
    filtro_valor = periodo

    if modo == "Dia":
        if periodo not in periodos_disponiveis:
            periodo = "Todos"
        filtro_valor = periodo
        filtro_modo = "Dias"
    elif modo == "Data":
        if data_escolhida not in datas_disponiveis:
            data_escolhida = datas_disponiveis[0] if datas_disponiveis else None
        filtro_modo = "Data"
        filtro_valor = data_escolhida
    elif modo == "Mes/Ano":
        if mes not in meses_disponiveis:
            mes = meses_disponiveis[-1] if meses_disponiveis else None
        filtro_valor = mes
    elif modo == "Intervalo":
        filtro_modo = "Intervalo"
        filtro_valor = None
    else:
        if ano not in anos_disponiveis:
            ano = anos_disponiveis[-1] if anos_disponiveis else None
        filtro_valor = ano

    if modo == "Intervalo":
        df_filtrado = filter_interval(base_df, inicio, fim)
    elif filtro_valor is None:
        df_filtrado = base_df.iloc[0:0]
    else:
        df_filtrado = filter_data(base_df, filtro_modo, filtro_valor)

    jogadores_unicos = sorted(
        set(
            base_df["winner1"].tolist()
            + base_df["winner2"].tolist()
            + base_df["loser1"].tolist()
            + base_df["loser2"].tolist()
        )
    )
    jogadores_unicos = [j for j in jogadores_unicos if j]

    jogadores_selecionados = request.args.getlist("jogadores")
    mensagem_limite = None

    if len(jogadores_selecionados) > 4:
        mensagem_limite = "Cada partida tem até 4 jogadores. Reduza o número de seleções."
        df_filtrado = df_filtrado.iloc[0:0]
    elif jogadores_selecionados:
        jogadores_alvo = set(jogadores_selecionados)
        colunas_jogadores = ["winner1", "winner2", "loser1", "loser2"]

        def contem_todos_jogadores(row) -> bool:
            jogadores_partida = {valor for valor in row if valor not in (None, "")}
            return jogadores_alvo.issubset(jogadores_partida)

        mask = df_filtrado[colunas_jogadores].apply(contem_todos_jogadores, axis=1)
        df_filtrado = df_filtrado[mask]

    df_ordenado = df_filtrado.sort_index(ascending=False)
    partidas_fmt = format_matches(df_ordenado)
    periodo_legenda = describe_games(modo, periodo, ano, mes, data_escolhida, inicio, fim)

    return render_template(
        "jogos.html",
        active_page="jogos",
        modo=modo,
        periodos=periodos_disponiveis,
        periodo_escolhido=periodo,
        datas=datas_disponiveis,
        data_selecionada=data_escolhida,
        inicio=inicio,
        fim=fim,
        meses=meses_disponiveis,
        mes_selecionado=mes,
        anos=anos_disponiveis,
        ano_selecionado=ano,
        jogadores=jogadores_unicos,
        jogadores_selecionados=jogadores_selecionados,
        partidas=partidas_fmt,
        periodo_legenda=periodo_legenda,
        jogos_filtrados=len(df_filtrado),
        jogos_total=len(base_df),
        mensagem_limite=mensagem_limite,
    )
