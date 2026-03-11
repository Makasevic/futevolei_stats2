from __future__ import annotations

from typing import Any, Callable

from flask import request


def detalhamento_page_response(
    *,
    fetch_base_dataframe: Callable[[], Any],
    jogadores_disponiveis: Callable[[Any], list[str]],
    calcular_metricas_jogador: Callable[[Any, str], Any],
    calcular_metricas_dupla: Callable[[Any, str, str], Any],
    render_template: Callable[..., Any],
) -> Any:
    df = fetch_base_dataframe()
    jogadores = jogadores_disponiveis(df)

    tipo = request.args.get("tipo", "Jogador")
    if tipo not in {"Jogador", "Dupla"}:
        tipo = "Jogador"

    detalhes = None
    jogador_escolhido = request.args.get("jogador") if tipo == "Jogador" else None
    jogador1 = request.args.get("j1") if tipo == "Dupla" else None
    jogador2 = request.args.get("j2") if tipo == "Dupla" else None
    parceiros_validos: list[str] = []

    if tipo == "Jogador":
        if jogador_escolhido not in jogadores:
            jogador_escolhido = None
        if jogador_escolhido:
            detalhes = calcular_metricas_jogador(df, jogador_escolhido)
    else:
        parceiros_por_jogador: dict[str, set[str]] = {j: set() for j in jogadores}
        for _, row in df.iterrows():
            duplas_partida = [
                [row.get("winner1"), row.get("winner2")],
                [row.get("loser1"), row.get("loser2")],
            ]
            for jogador_a, jogador_b in duplas_partida:
                if not jogador_a or not jogador_b:
                    continue
                if "Outro" in str(jogador_a) or "Outro" in str(jogador_b):
                    continue
                parceiros_por_jogador.setdefault(jogador_a, set()).add(jogador_b)
                parceiros_por_jogador.setdefault(jogador_b, set()).add(jogador_a)

        if jogador1 not in jogadores:
            jogador1 = None
        parceiros_validos = sorted(parceiros_por_jogador.get(jogador1, set())) if jogador1 else []
        if jogador2 not in parceiros_validos:
            jogador2 = None
        if jogador1 and jogador2:
            detalhes = calcular_metricas_dupla(df, jogador1, jogador2)

    return render_template(
        "detalhamento.html",
        active_page="detalhamento",
        tipo=tipo,
        jogadores=jogadores,
        jogador_escolhido=jogador_escolhido,
        jogador1=jogador1,
        jogador2=jogador2,
        parceiros_validos=parceiros_validos,
        detalhes=detalhes,
    )
