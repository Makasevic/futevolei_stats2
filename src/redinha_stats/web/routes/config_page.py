from __future__ import annotations

from typing import Any, Callable

from flask import request


def config_page_response(
    *,
    get_config: Callable[[], Any],
    update_config: Callable[..., Any],
    safe_float: Callable[[str | None, float], float],
    safe_int: Callable[[str | None, int], int],
    render_template: Callable[..., Any],
) -> Any:
    app_config = get_config()
    mensagem = None

    if request.method == "POST":
        update_config(
            min_participation_ratio=safe_float(
                request.form.get("min_participation_ratio"),
                app_config.min_participation_ratio,
            ),
            min_duo_matches=safe_int(
                request.form.get("min_duo_matches"),
                app_config.min_duo_matches,
            ),
        )
        app_config = get_config()
        mensagem = "Configuracoes atualizadas com sucesso!"

    return render_template(
        "config.html",
        active_page="config",
        min_participation_ratio=app_config.min_participation_ratio,
        min_duo_matches=app_config.min_duo_matches,
        mensagem=mensagem,
    )
