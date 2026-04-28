from __future__ import annotations

from datetime import date
from typing import Any, Callable

from flask import request, session

from src.redinha_stats.web.admin_helpers import normalize_player_name


def admin_page_response(
    *,
    admin_password: str | None,
    entry_password: str | None,
    set_admin_feedback: Callable[[str, str], None],
    available_championship_keys: Callable[[], list[str]],
    get_championship_edit_password: Callable[[str], str | None],
    unlocked_tournament_keys: Callable[[], set[str]],
    set_unlocked_tournament_keys: Callable[[set[str]], None],
    reset_cache: Callable[[], None],
    fetch_base_dataframe: Callable[[], Any],
    add_player: Callable[[str], bool],
    parse_bulk_line: Callable[[str], dict[str, Any] | None],
    validate_registered_players: Callable[[list[str]], list[str]],
    validate_match_data: Callable[[Any, str, dict[str, Any]], list[str]],
    insert_match: Callable[[dict[str, Any]], None],
    serialize_payload: Callable[[dict[str, Any]], dict[str, Any]],
    parse_form_date: Callable[[str | None], date | None],
    update_match: Callable[[Any, dict[str, Any], str], None],
    delete_match: Callable[[Any, str], None],
    rename_player: Callable[[str, str], Any],
    merge_players: Callable[[str, str], Any],
    is_valid_identifier: Callable[[Any], bool],
    matches_from_df: Callable[[Any], list[dict[str, Any]]],
    registered_players: Callable[[Any], list[str]],
    get_championship_view: Callable[[str], dict[str, Any]],
    save_match_score: Callable[[str, str, str | None, str | None], None],
    redirect_to_admin: Callable[..., Any],
    render_template: Callable[..., Any],
    team_fields: list[str],
) -> Any:
    requires_auth = bool(admin_password or entry_password)

    feedback = session.pop("admin_feedback", None)
    authenticated = session.get("admin_authenticated") or not requires_auth
    role = session.get("admin_role", "full") if authenticated else None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "login":
            password = request.form.get("password", "")
            if admin_password and password == admin_password:
                session["admin_authenticated"] = True
                session["admin_role"] = "full"
                set_admin_feedback("success", "Acesso liberado como administrador completo.")
            elif entry_password and password == entry_password:
                session["admin_authenticated"] = True
                session["admin_role"] = "limited"
                set_admin_feedback("success", "Acesso liberado para lancamentos.")
            else:
                set_admin_feedback("error", "Senha incorreta. Tente novamente.")
            return redirect_to_admin()

        if action == "logout":
            session.pop("admin_authenticated", None)
            session.pop("admin_role", None)
            set_admin_feedback("success", "Sessao encerrada.")
            return redirect_to_admin()

        if action == "unlock_tournament":
            available_keys = available_championship_keys()
            default_key = available_keys[0] if available_keys else date.today().strftime("%Y%m%d")
            selected_key = (request.form.get("championship_key", default_key) or "").strip()
            if selected_key not in available_keys:
                selected_key = default_key

            if role == "full":
                set_admin_feedback("success", "Administrador completo ja possui acesso a todos os torneios.")
                return redirect_to_admin(championship_key=selected_key)

            provided_password = request.form.get("tournament_password", "").strip()
            expected_password = get_championship_edit_password(selected_key)
            if not expected_password:
                set_admin_feedback("error", "Este torneio nao possui senha de edicao configurada.")
                return redirect_to_admin(championship_key=selected_key)
            if provided_password != expected_password:
                set_admin_feedback("error", "Senha do torneio incorreta.")
                return redirect_to_admin(championship_key=selected_key)

            unlocked = unlocked_tournament_keys()
            unlocked.add(selected_key)
            set_unlocked_tournament_keys(unlocked)
            set_admin_feedback("success", "Edicao do torneio desbloqueada para esta sessao.")
            return redirect_to_admin(championship_key=selected_key)

        if not authenticated:
            set_admin_feedback("error", "Acesso nao autorizado. Informe a senha para continuar.")
            return redirect_to_admin()

        if action == "refresh":
            try:
                reset_cache()
                fetch_base_dataframe()
                set_admin_feedback("success", "Cache atualizado com sucesso!")
            except Exception as exc:
                set_admin_feedback("error", f"Erro ao atualizar cache: {exc}")
            return redirect_to_admin()

        if action == "add_player":
            player_name = normalize_player_name(request.form.get("player_name", ""))
            if not player_name:
                set_admin_feedback("error", "Informe o nome do jogador.")
                return redirect_to_admin()
            if add_player(player_name):
                set_admin_feedback("success", f"Jogador '{player_name}' cadastrado com sucesso!")
            else:
                set_admin_feedback("error", "Jogador ja cadastrado ou nome invalido.")
            return redirect_to_admin()

        if role == "limited" and action in {"update", "delete", "rename_player", "merge_players"}:
            set_admin_feedback("error", "Somente administradores completos podem editar ou excluir partidas.")
            return redirect_to_admin()

        if action == "rename_player":
            old_name = normalize_player_name(request.form.get("old_name", ""))
            new_name = normalize_player_name(request.form.get("new_name", ""))
            if not old_name or not new_name:
                set_admin_feedback("error", "Selecione o jogador atual e informe o novo nome.")
                return redirect_to_admin()
            try:
                result = rename_player(old_name, new_name)
                reset_cache()
                matches_updated = result.get("matches_updated", 0) if isinstance(result, dict) else 0
                set_admin_feedback(
                    "success",
                    f"Jogador renomeado de '{old_name}' para '{new_name}'. "
                    f"{matches_updated} referencia(s) em partidas atualizada(s).",
                )
            except ValueError as exc:
                set_admin_feedback("error", str(exc))
            except Exception as exc:
                set_admin_feedback("error", f"Erro ao renomear jogador: {exc}")
            return redirect_to_admin()

        if action == "merge_players":
            kept_name = normalize_player_name(request.form.get("kept_name", ""))
            removed_name = normalize_player_name(request.form.get("removed_name", ""))
            if not kept_name or not removed_name:
                set_admin_feedback("error", "Selecione os dois jogadores para fundir.")
                return redirect_to_admin()
            try:
                result = merge_players(kept_name, removed_name)
                reset_cache()
                matches_updated = result.get("matches_updated", 0) if isinstance(result, dict) else 0
                user_deleted = result.get("user_deleted", False) if isinstance(result, dict) else False
                tail = " Usuario antigo apagado." if user_deleted else " Usuario antigo mantido (esta em outros grupos)."
                set_admin_feedback(
                    "success",
                    f"'{removed_name}' fundido em '{kept_name}'. "
                    f"{matches_updated} referencia(s) em partidas atualizada(s).{tail}",
                )
            except ValueError as exc:
                set_admin_feedback("error", str(exc))
            except Exception as exc:
                set_admin_feedback("error", f"Erro ao fundir jogadores: {exc}")
            return redirect_to_admin()

        if action == "championship_score":
            available_keys = available_championship_keys()
            default_key = available_keys[0] if available_keys else date.today().strftime("%Y%m%d")
            selected_key = (request.form.get("championship_key", default_key) or "").strip()
            if selected_key not in available_keys:
                selected_key = default_key
            can_edit_tournament = role == "full" or (selected_key in unlocked_tournament_keys())
            if not can_edit_tournament:
                set_admin_feedback("error", "Desbloqueie este torneio com a senha para editar os placares.")
                return redirect_to_admin(championship_key=selected_key)

            match_id = request.form.get("championship_match_id", "").strip()
            score_a_raw = request.form.get("score_a")
            score_b_raw = request.form.get("score_b")
            try:
                save_match_score(selected_key, match_id, score_a_raw, score_b_raw)
                set_admin_feedback("success", "Placar do torneio atualizado com sucesso.")
            except Exception as exc:
                set_admin_feedback("error", f"Erro ao salvar placar do torneio: {exc}")
            return redirect_to_admin(championship_key=selected_key)

        if action == "bulk_create":
            bulk_matches = request.form.get("bulk_matches", "")
            match_date = parse_form_date(request.form.get("bulk_date")) or date.today()
            linhas = [linha.strip() for linha in bulk_matches.splitlines() if linha.strip()]
            if not linhas:
                set_admin_feedback("error", "Informe ao menos uma linha de partida.")
                return redirect_to_admin()

            parsed_matches: list[dict[str, Any]] = []
            errors: list[str] = []
            missing_players: set[str] = set()

            for idx, linha in enumerate(linhas, start=1):
                parsed = parse_bulk_line(linha)
                if not parsed:
                    errors.append(f"Linha {idx} em formato invalido.")
                    continue
                players = [parsed.get(field, "") for field in team_fields]
                missing_players.update(validate_registered_players(players))
                payload = {**parsed, "date": match_date}
                validation_errors = validate_match_data(None, "Adicionar", payload)
                if validation_errors:
                    errors.extend([f"Linha {idx}: {erro}" for erro in validation_errors])
                else:
                    parsed_matches.append(payload)

            if missing_players:
                errors.append("Jogadores nao cadastrados: " + ", ".join(sorted(missing_players)))
            if errors:
                set_admin_feedback("error", " ".join(errors))
                return redirect_to_admin()

            try:
                for payload in parsed_matches:
                    insert_match(serialize_payload(payload))
                reset_cache()
                set_admin_feedback("success", f"{len(parsed_matches)} partida(s) cadastrada(s) em bloco!")
            except Exception as exc:
                set_admin_feedback("error", f"Erro ao salvar partidas em bloco: {exc}")
            return redirect_to_admin()

        if action in {"create", "update"}:
            score_a_raw = request.form.get("score_a", "").strip()
            score_b_raw = request.form.get("score_b", "").strip()
            score_value = None
            if score_a_raw == "" and score_b_raw == "":
                score_value = ""
            elif score_a_raw and score_b_raw and score_a_raw.isdigit() and score_b_raw.isdigit():
                score_value = f"{int(score_a_raw)}x{int(score_b_raw)}"

            payload = {
                "winner1": normalize_player_name(request.form.get("winner1", "")),
                "winner2": normalize_player_name(request.form.get("winner2", "")),
                "loser1": normalize_player_name(request.form.get("loser1", "")),
                "loser2": normalize_player_name(request.form.get("loser2", "")),
                "date": parse_form_date(request.form.get("date")),
            }
            if score_value is not None:
                payload["score"] = score_value

            match_id = request.form.get("match_id") if action == "update" else None
            id_field = request.form.get("id_field", "id")
            errors = validate_match_data(match_id, "Atualizar" if action == "update" else "Adicionar", payload)
            if errors:
                set_admin_feedback("error", " ".join(errors))
                return redirect_to_admin()

            try:
                if action == "create":
                    insert_match(serialize_payload(payload))
                    set_admin_feedback("success", "Partida cadastrada com sucesso!")
                else:
                    update_match(match_id, serialize_payload(payload), id_field)
                    set_admin_feedback("success", "Partida atualizada com sucesso!")
                reset_cache()
            except Exception as exc:
                set_admin_feedback("error", f"Erro ao salvar partida: {exc}")

            match_date_param = request.form.get("match_date") or None
            return redirect_to_admin(match_date=match_date_param) if match_date_param else redirect_to_admin()

        if action == "delete":
            match_id = request.form.get("match_id")
            id_field = request.form.get("id_field", "id")
            if not is_valid_identifier(match_id):
                set_admin_feedback("error", "Selecione uma partida para excluir.")
                return redirect_to_admin()
            try:
                delete_match(match_id, id_field)
                reset_cache()
                set_admin_feedback("success", "Partida removida com sucesso!")
            except Exception as exc:
                set_admin_feedback("error", f"Erro ao excluir partida: {exc}")

            match_date_param = request.form.get("match_date") or None
            return redirect_to_admin(match_date=match_date_param) if match_date_param else redirect_to_admin()

    df = fetch_base_dataframe()
    matches = matches_from_df(df)
    match_dates = sorted({match["date"] for match in matches if match.get("date")}, reverse=True)
    selected_match_date = parse_form_date(request.args.get("match_date"))
    if match_dates:
        if selected_match_date not in match_dates:
            selected_match_date = match_dates[0]
        matches = [match for match in matches if match.get("date") == selected_match_date]
    else:
        selected_match_date = None
        matches = []

    players = registered_players(df)
    players_text = "\n".join(players)
    championship_keys = available_championship_keys()
    default_key = championship_keys[0] if championship_keys else date.today().strftime("%Y%m%d")
    selected_key = (request.args.get("championship_key", default_key) or "").strip()
    if selected_key not in championship_keys:
        selected_key = default_key
    championship_payload = get_championship_view(selected_key)
    championship_can_edit = role == "full" or (selected_key in unlocked_tournament_keys())

    return render_template(
        "admin.html",
        active_page="admin",
        authenticated=authenticated,
        role=role or "full",
        feedback=feedback,
        requires_auth=requires_auth,
        players=players,
        players_text=players_text,
        matches=matches,
        match_dates=[date_item.isoformat() for date_item in match_dates],
        selected_match_date=selected_match_date.isoformat() if selected_match_date else "",
        today=date.today().isoformat(),
        championship_keys=championship_keys,
        championship_selected_key=selected_key,
        championship_payload=championship_payload,
        championship_can_edit=championship_can_edit,
    )
