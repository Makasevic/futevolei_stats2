from __future__ import annotations

from typing import Any


def _capture_template(monkeypatch, main_api_module) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_render_template(template_name: str, **context: Any) -> str:
        captured["template"] = template_name
        captured["context"] = context
        return f"TEMPLATE:{template_name}"

    monkeypatch.setattr(main_api_module, "render_template", fake_render_template)
    return captured


def test_home_renders_ranking_context(
    client,
    main_api_module,
    monkeypatch,
    sample_matches_df,
    sample_players_ranking,
    sample_duos_ranking,
):
    captured = _capture_template(monkeypatch, main_api_module)
    monkeypatch.setattr(main_api_module, "_fetch_base_dataframe", lambda: sample_matches_df)
    monkeypatch.setattr(
        main_api_module,
        "_filter_rankings",
        lambda *args, **kwargs: (
            sample_matches_df,
            sample_players_ranking,
            sample_duos_ranking,
        ),
    )
    monkeypatch.setattr(
        main_api_module,
        "_build_highlights",
        lambda linhas: [{"nome": linhas[0]["nome"], "medal": "🥇"}],
    )

    response = client.get("/")

    assert response.status_code == 200
    assert captured["template"] == "ranking.html"
    assert captured["context"]["active_page"] == "ranking"
    assert captured["context"]["jogos_total"] == 2
    assert captured["context"]["jogadores"][0]["nome"] == "Ana"
    assert captured["context"]["duplas"][0]["nome"] == "Ana e Bia"


def test_jogos_renders_formatted_matches(
    client,
    main_api_module,
    monkeypatch,
    sample_matches_df,
):
    captured = _capture_template(monkeypatch, main_api_module)
    monkeypatch.setattr(main_api_module, "_fetch_base_dataframe", lambda: sample_matches_df)

    response = client.get("/jogos")

    assert response.status_code == 200
    assert captured["template"] == "jogos.html"
    assert captured["context"]["active_page"] == "jogos"
    assert captured["context"]["jogos_total"] == 2
    assert captured["context"]["partidas"][0]["vencedores"] == "Ana & Bia"
    assert captured["context"]["partidas"][0]["perdedores"] == "Caio & Duda"


def test_jogos_limits_selected_players(
    client,
    main_api_module,
    monkeypatch,
    sample_matches_df,
):
    captured = _capture_template(monkeypatch, main_api_module)
    monkeypatch.setattr(main_api_module, "_fetch_base_dataframe", lambda: sample_matches_df)

    response = client.get(
        "/jogos?jogadores=Ana&jogadores=Bia&jogadores=Caio&jogadores=Duda&jogadores=Extra"
    )

    assert response.status_code == 200
    assert "Reduza o n" in captured["context"]["mensagem_limite"]
    assert captured["context"]["partidas"] == []


def test_api_ranking_returns_json_payload(
    client,
    main_api_module,
    monkeypatch,
    sample_matches_df,
    sample_players_ranking,
    sample_duos_ranking,
):
    monkeypatch.setattr(
        main_api_module,
        "_filter_rankings",
        lambda *args, **kwargs: (
            sample_matches_df,
            sample_players_ranking,
            sample_duos_ranking,
        ),
    )

    response = client.get("/api/ranking?modo=Mes/Ano&mes=2026-03")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["periodo"] == "2026-03"
    assert payload["total_partidas"] == 2
    assert payload["jogadores"][0]["nome"] == "Ana"
    assert payload["duplas"][0]["nome"] == "Ana e Bia"


def test_hidden_players_supports_text_output(client, main_api_module, monkeypatch):
    monkeypatch.setattr(main_api_module, "_fetch_base_dataframe", lambda: object())
    monkeypatch.setattr(
        main_api_module,
        "_players_ranked_by_games",
        lambda df: ["Ana", "Bia", "Caio"],
    )

    response = client.get("/_oculto/jogadores?formato=txt")

    assert response.status_code == 200
    assert response.mimetype == "text/plain"
    assert response.get_data(as_text=True) == "Ana\nBia\nCaio"


def test_campeonato_get_renders_payload(client, main_api_module, monkeypatch):
    captured = _capture_template(monkeypatch, main_api_module)
    monkeypatch.setattr(main_api_module, "available_championship_keys", lambda: ["202602"])
    monkeypatch.setattr(
        main_api_module,
        "get_championship_view",
        lambda key: {
            "title": "Torneio 2026",
            "editable_matches": [{"id": "QF1", "score_a": 21, "score_b": 18}],
        },
    )

    response = client.get("/campeonato?championship=202602")

    assert response.status_code == 200
    assert captured["template"] == "campeonato.html"
    assert captured["context"]["championship"]["title"] == "Torneio 2026"
    assert len(captured["context"]["championship_saved_matches"]) == 1


def test_campeonato_post_unlock_updates_session(client, main_api_module, monkeypatch):
    monkeypatch.setattr(main_api_module, "available_championship_keys", lambda: ["202602"])
    monkeypatch.setattr(main_api_module, "get_championship_edit_password", lambda key: "secret")

    response = client.post(
        "/campeonato",
        data={
            "action": "unlock_tournament",
            "championship_key": "202602",
            "tournament_password": "secret",
        },
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/campeonato?championship=202602")
    with client.session_transaction() as session:
        assert session["tournament_edit_keys"] == ["202602"]
        assert session["tournament_feedback"]["status"] == "success"
