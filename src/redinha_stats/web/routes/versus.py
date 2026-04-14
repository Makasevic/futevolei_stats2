from __future__ import annotations

from typing import Any, Callable

from flask import request


def versus_page_response(
    *,
    fetch_base_dataframe: Callable[[], Any],
    jogadores_disponiveis: Callable[[Any], list[str]],
    parceiros_por_jogador: Callable[[Any], dict[str, list[str]]],
    oponentes_por_dupla_jogadores: Callable[[Any], dict[str, list[str]]],
    oponentes_por_jogador: Callable[[Any], dict[str, list[str]]],
    estatisticas_dupla: Callable[[Any, str], Any],
    confronto_direto_duplas: Callable[[Any, str, str], Any],
    estatisticas_jogador_individual: Callable[[Any, str], Any],
    confronto_direto: Callable[[Any, str, str], Any],
    render_template: Callable[..., Any],
) -> Any:
    df = fetch_base_dataframe()
    tipo = request.args.get("tipo", "Jogador")
    if tipo not in ("Jogador", "Dupla"):
        tipo = "Jogador"

    parceiros = {}
    duplas_selecao = {
        "d1a": request.args.get("d1a"),
        "d1b": request.args.get("d1b"),
        "d2a": request.args.get("d2a"),
        "d2b": request.args.get("d2b"),
    }
    jogador1 = request.args.get("j1")
    jogador2 = request.args.get("j2")

    if tipo == "Dupla":
        jogadores = jogadores_disponiveis(df)
        parceiros = parceiros_por_jogador(df)
        oponentes_duplas = oponentes_por_dupla_jogadores(df)
        oponentes_jogadores = {}

        for chave, valor in duplas_selecao.items():
            if valor not in jogadores:
                duplas_selecao[chave] = None

        def dupla_valida(primeiro: str | None, segundo: str | None) -> bool:
            if not primeiro or not segundo or primeiro == segundo:
                return False
            return segundo in parceiros.get(primeiro, [])

        if not dupla_valida(duplas_selecao["d1a"], duplas_selecao["d1b"]):
            duplas_selecao["d1b"] = None
        if not dupla_valida(duplas_selecao["d2a"], duplas_selecao["d2b"]):
            duplas_selecao["d2b"] = None
    else:
        jogadores = jogadores_disponiveis(df)
        oponentes_jogadores = oponentes_por_jogador(df)
        oponentes_duplas = {}
        if jogador1 not in jogadores:
            jogador1 = None
        if jogador2 not in jogadores:
            jogador2 = None

    estatisticas = None
    confronto = None
    dupla1_nome = None
    dupla2_nome = None

    if tipo == "Dupla":
        if duplas_selecao["d1a"] and duplas_selecao["d1b"]:
            dupla1_nome = " e ".join(sorted([duplas_selecao["d1a"], duplas_selecao["d1b"]]))
        if duplas_selecao["d2a"] and duplas_selecao["d2b"]:
            dupla2_nome = " e ".join(sorted([duplas_selecao["d2a"], duplas_selecao["d2b"]]))

        if dupla1_nome:
            oponentes_dupla1 = set(oponentes_duplas.get(dupla1_nome, []))
            if duplas_selecao["d2a"] not in oponentes_dupla1:
                duplas_selecao["d2a"] = None
                duplas_selecao["d2b"] = None
            elif duplas_selecao["d2b"] and duplas_selecao["d2b"] not in oponentes_dupla1:
                duplas_selecao["d2b"] = None

        if duplas_selecao["d2a"] and duplas_selecao["d2b"]:
            parceiros_validos_d2a = set(parceiros.get(duplas_selecao["d2a"], []))
            if duplas_selecao["d2b"] not in parceiros_validos_d2a:
                duplas_selecao["d2b"] = None

        if duplas_selecao["d2a"] and duplas_selecao["d2b"]:
            dupla2_nome = " e ".join(sorted([duplas_selecao["d2a"], duplas_selecao["d2b"]]))

        if dupla1_nome and dupla2_nome and dupla1_nome != dupla2_nome:
            estatisticas = {
                "jogador1": estatisticas_dupla(df, dupla1_nome),
                "jogador2": estatisticas_dupla(df, dupla2_nome),
            }
            confronto = confronto_direto_duplas(df, dupla1_nome, dupla2_nome)
    else:
        if jogador1 and jogador2 and jogador1 != jogador2:
            estatisticas = {
                "jogador1": estatisticas_jogador_individual(df, jogador1),
                "jogador2": estatisticas_jogador_individual(df, jogador2),
            }
            confronto = confronto_direto(df, jogador1, jogador2)

    return render_template(
        "versus.html",
        active_page="versus",
        tipo=tipo,
        jogadores=jogadores,
        jogador1=jogador1,
        jogador2=jogador2,
        dupla1_nome=dupla1_nome,
        dupla2_nome=dupla2_nome,
        duplas_selecao=duplas_selecao,
        parceiros_por_jogador=parceiros,
        oponentes_por_dupla_jogadores=oponentes_duplas,
        oponentes_por_jogador=oponentes_jogadores,
        estatisticas=estatisticas,
        confronto=confronto,
    )
