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


def test_infos_renders_summary_payload(client, main_api_module, monkeypatch):
    captured = _capture_template(monkeypatch, main_api_module)
    monkeypatch.setattr(main_api_module, "_fetch_base_dataframe", lambda: object())
    monkeypatch.setattr(
        main_api_module,
        "_resumo_infos",
        lambda df: {"resumo": {"total_partidas": 12}, "destaques_primarios": []},
    )

    response = client.get("/infos")

    assert response.status_code == 200
    assert captured["template"] == "infos.html"
    assert captured["context"]["active_page"] == "infos"
    assert captured["context"]["resumo"]["total_partidas"] == 12


def test_awards_renders_selected_year(client, main_api_module, monkeypatch):
    captured = _capture_template(monkeypatch, main_api_module)
    monkeypatch.setattr(main_api_module, "available_awards_years", lambda: [2025, 2024])
    monkeypatch.setattr(
        main_api_module,
        "_build_awards_data",
        lambda year: [
            {
                "category_type": "positive",
                "total_votes": 10,
                "winners": [{"name": "Ana", "votes": 6}],
            }
        ],
    )
    monkeypatch.setattr(
        main_api_module,
        "_build_awards_champions",
        lambda awards: {
            "positive": {"names": "Ana", "count": 1},
            "negative": {"names": "-", "count": 0},
        },
    )

    response = client.get("/awards?year=2025")

    assert response.status_code == 200
    assert captured["template"] == "awards.html"
    assert captured["context"]["active_page"] == "awards"
    assert captured["context"]["awards_selected_year"] == 2025
    assert captured["context"]["awards_total_votes"] == 10


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


def test_campeonato_post_score_requires_unlock(client, main_api_module, monkeypatch):
    monkeypatch.setattr(main_api_module, "available_championship_keys", lambda: ["202602"])

    response = client.post(
        "/campeonato",
        data={
            "action": "championship_score",
            "championship_key": "202602",
            "championship_match_id": "QF1",
            "score_a": "21",
            "score_b": "18",
        },
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/campeonato?championship=202602")
    with client.session_transaction() as session:
        assert session["tournament_feedback"]["status"] == "error"
        assert "Desbloqueie o torneio" in session["tournament_feedback"]["message"]


def test_campeonato_post_delete_requires_unlock(client, main_api_module, monkeypatch):
    monkeypatch.setattr(main_api_module, "available_championship_keys", lambda: ["202602"])

    response = client.post(
        "/campeonato",
        data={
            "action": "delete_championship_score",
            "championship_key": "202602",
            "championship_match_id": "QF1",
        },
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/campeonato?championship=202602")
    with client.session_transaction() as session:
        assert session["tournament_feedback"]["status"] == "error"
        assert "Desbloqueie o torneio" in session["tournament_feedback"]["message"]


def test_campeonato_post_score_saves_when_unlocked(client, main_api_module, monkeypatch):
    calls = []
    monkeypatch.setattr(main_api_module, "available_championship_keys", lambda: ["202602"])
    monkeypatch.setattr(
        main_api_module,
        "save_match_score",
        lambda championship_key, match_id, score_a, score_b: calls.append(
            (championship_key, match_id, score_a, score_b)
        ),
    )

    with client.session_transaction() as session:
        session["tournament_edit_keys"] = ["202602"]

    response = client.post(
        "/campeonato",
        data={
            "action": "championship_score",
            "championship_key": "202602",
            "championship_match_id": "QF1",
            "score_a": "21",
            "score_b": "18",
        },
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/campeonato?championship=202602")
    assert calls == [("202602", "QF1", "21", "18")]
    with client.session_transaction() as session:
        assert session["tournament_feedback"]["status"] == "success"


def test_config_page_renders_current_values(client, main_api_module, monkeypatch):
    captured = _capture_template(monkeypatch, main_api_module)

    class DummyConfig:
        min_participation_ratio = 0.5
        min_duo_matches = 4

    monkeypatch.setattr(main_api_module, "get_config", lambda: DummyConfig())

    response = client.get("/config")

    assert response.status_code == 200
    assert captured["template"] == "config.html"
    assert captured["context"]["active_page"] == "config"
    assert captured["context"]["min_participation_ratio"] == 0.5
    assert captured["context"]["min_duo_matches"] == 4


def test_detalhamento_renders_player_metrics(client, main_api_module, monkeypatch, sample_matches_df):
    captured = _capture_template(monkeypatch, main_api_module)
    monkeypatch.setattr(main_api_module, "_fetch_base_dataframe", lambda: sample_matches_df)
    monkeypatch.setattr(main_api_module, "_jogadores_disponiveis", lambda df: ["Ana", "Bia", "Caio"])
    monkeypatch.setattr(
        main_api_module,
        "calcular_metricas_jogador",
        lambda df, jogador: {"metricas": {"jogador": jogador}},
    )

    response = client.get("/detalhamento?tipo=Jogador&jogador=Ana")

    assert response.status_code == 200
    assert captured["template"] == "detalhamento.html"
    assert captured["context"]["active_page"] == "detalhamento"
    assert captured["context"]["detalhes"]["metricas"]["jogador"] == "Ana"


def test_detalhamento_renders_general_metrics(client, main_api_module, monkeypatch, sample_matches_df):
    captured = _capture_template(monkeypatch, main_api_module)
    monkeypatch.setattr(main_api_module, "_fetch_base_dataframe", lambda: sample_matches_df)
    monkeypatch.setattr(main_api_module, "_jogadores_disponiveis", lambda df: ["Ana", "Bia", "Caio"])
    monkeypatch.setattr(
        main_api_module,
        "calcular_metricas_gerais",
        lambda df: {
            "qualidade_parceiros_colunas": [
                {"rating": 5, "estrelas": "*****", "jogadores": [{"nome": "Ana"}]},
            ],
            "qualidade_parceiros": [{"nome": "Ana"}],
        },
    )

    response = client.get("/detalhamento?tipo=Geral")

    assert response.status_code == 200
    assert captured["template"] == "detalhamento.html"
    assert captured["context"]["active_page"] == "detalhamento"
    assert captured["context"]["tipo"] == "Geral"
    assert captured["context"]["detalhes"]["qualidade_parceiros"][0]["nome"] == "Ana"


def test_versus_renders_individual_comparison(client, main_api_module, monkeypatch, sample_matches_df):
    captured = _capture_template(monkeypatch, main_api_module)
    monkeypatch.setattr(main_api_module, "_fetch_base_dataframe", lambda: sample_matches_df)
    monkeypatch.setattr(main_api_module, "_jogadores_disponiveis", lambda df: ["Ana", "Caio"])
    monkeypatch.setattr(main_api_module, "_oponentes_por_jogador", lambda df: {"Ana": ["Caio"], "Caio": ["Ana"]})
    monkeypatch.setattr(
        main_api_module,
        "_estatisticas_jogador_individual",
        lambda df, jogador: {"nome": jogador, "jogos": 2},
    )
    monkeypatch.setattr(
        main_api_module,
        "_confronto_direto",
        lambda df, jogador1, jogador2: {"j1": jogador1, "j2": jogador2},
    )

    response = client.get("/versus?tipo=Jogador&j1=Ana&j2=Caio")

    assert response.status_code == 200
    assert captured["template"] == "versus.html"
    assert captured["context"]["active_page"] == "versus"
    assert captured["context"]["estatisticas"]["jogador1"]["nome"] == "Ana"
    assert captured["context"]["confronto"]["j2"] == "Caio"


def test_admin_get_renders_context(client, main_api_module, monkeypatch):
    captured = _capture_template(monkeypatch, main_api_module)
    monkeypatch.setattr(main_api_module, "ADMIN_PASSWORD", "")
    monkeypatch.setattr(main_api_module, "MATCH_ENTRY_PASSWORD", "")
    monkeypatch.setattr(main_api_module, "_fetch_base_dataframe", lambda: object())
    monkeypatch.setattr(main_api_module, "_matches_from_df", lambda df: [])
    monkeypatch.setattr(main_api_module, "_registered_players", lambda df: ["Ana", "Bia"])
    monkeypatch.setattr(main_api_module, "available_championship_keys", lambda: ["202602"])
    monkeypatch.setattr(main_api_module, "get_championship_view", lambda key: {"title": "Torneio"})

    response = client.get("/admin")

    assert response.status_code == 200
    assert captured["template"] == "admin.html"
    assert captured["context"]["active_page"] == "admin"
    assert captured["context"]["players"] == ["Ana", "Bia"]
    assert captured["context"]["championship_selected_key"] == "202602"


def test_admin_login_sets_session(client, main_api_module, monkeypatch):
    monkeypatch.setattr(main_api_module, "ADMIN_PASSWORD", "secret")
    monkeypatch.setattr(main_api_module, "MATCH_ENTRY_PASSWORD", "entry")

    response = client.post(
        "/admin",
        data={"action": "login", "password": "secret"},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/admin")
    with client.session_transaction() as session:
        assert session["admin_authenticated"] is True
        assert session["admin_role"] == "full"
        assert session["admin_feedback"]["status"] == "success"
