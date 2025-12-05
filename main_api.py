from __future__ import annotations

"""Aplicação Flask para exibir o ranking em HTML estático."""

from collections import Counter
import os
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from app_settings import get_config, update_config
from config import ADMIN_PASSWORD, MATCH_ENTRY_PASSWORD
from detalhamento import calcular_metricas_dupla, calcular_metricas_jogador
from extraction import get_matches
from preparation import preparar_dataframe
from processing import filtrar_dados, preparar_dados_duplas, preparar_dados_individuais
from data_access.supabase_repository import delete_match, insert_match, update_match
from ui_config import get_ui_config

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")


def _current_ui_config():
    """Retorna a configuração de UI imutável."""

    return get_ui_config()


# --------------------------------- Dados ----------------------------------
@lru_cache(maxsize=1)
def _fetch_base_dataframe() -> pd.DataFrame:
    """Busca as partidas e devolve o DataFrame com índice datetime."""
    matches = get_matches()
    df = preparar_dataframe(matches)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.set_index(pd.to_datetime(df["date"], errors="coerce"))
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.sort_index()
    return df


def _filtrar_por_intervalo(
    df: pd.DataFrame, inicio: str | None, fim: str | None
) -> pd.DataFrame:
    """Filtra por intervalo customizável."""

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")

    try:
        data_inicio = pd.to_datetime(inicio) if inicio else None
    except (TypeError, ValueError):
        data_inicio = None

    try:
        data_fim = pd.to_datetime(fim) if fim else None
    except (TypeError, ValueError):
        data_fim = None

    if data_inicio is None and data_fim is None:
        return df

    filtrado = df
    if data_inicio is not None:
        filtrado = filtrado[filtrado.index >= data_inicio]
    if data_fim is not None:
        filtrado = filtrado[filtrado.index <= data_fim]

    return filtrado


def _excluded_players() -> set:
    config = _current_ui_config()
    return set(config.excluded_players)


def _filter_rankings(
    modo: str,
    periodo: str | None,
    ano: str | None,
    mes: str | None,
    inicio: str | None,
    fim: str | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_df = _fetch_base_dataframe()
    config = get_config()

    if modo == "Ano" and ano:
        df_filtrado = filtrar_dados(base_df, "Ano", ano)
    elif modo == "Mês/Ano" and mes:
        df_filtrado = filtrar_dados(base_df, "Mês/Ano", mes)
    elif modo == "Intervalo":
        df_filtrado = _filtrar_por_intervalo(base_df, inicio, fim)
    else:
        df_filtrado = filtrar_dados(base_df, "Dias", periodo) if periodo else base_df

    jogadores = preparar_dados_individuais(df_filtrado)
    duplas = preparar_dados_duplas(df_filtrado)

    excluidos = _excluded_players()
    if excluidos:
        jogadores = jogadores[~jogadores["jogadores"].isin(excluidos)]
        duplas = duplas[~duplas["duplas"].isin(excluidos)]

    media_top_10 = jogadores["jogos"].nlargest(10).mean()
    if media_top_10 > 0:
        limiar = media_top_10 * config.min_participation_ratio
        jogadores = jogadores[jogadores["jogos"] >= limiar]

    if config.min_duo_matches > 0:
        duplas = duplas[duplas["jogos"] >= config.min_duo_matches]

    # Garantir que os índices sejam contínuos após filtros, para não quebrar as medalhas
    jogadores = jogadores.reset_index(drop=True)
    duplas = duplas.reset_index(drop=True)

    return base_df, jogadores, duplas


def _descricao_periodo(
    modo: str, periodo: str | None, ano: str | None, mes: str | None, inicio: str | None, fim: str | None
) -> str:
    if modo == "Ano" and ano:
        return f"Ano {ano}"
    if modo == "Mês/Ano" and mes:
        return mes
    if modo == "Intervalo":
        if inicio or fim:
            return f"Intervalo {inicio or '...'} a {fim or '...'}"
        return "Intervalo personalizado"
    return periodo or "Todos"


def _format_ranking(df: pd.DataFrame, nome_col: str) -> List[Dict[str, str]]:
    medalhas = ["🥇", "🥈", "🥉"]
    linhas: List[Dict[str, str]] = []
    total_linhas = len(df)

    for idx, linha in df.iterrows():
        if idx < len(medalhas):
            posicao = medalhas[idx]
        else:
            posicao = "😱" if idx == total_linhas - 1 else f"{idx + 1:02d}"
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
    medals = ["🥇", "🥈", "🥉"]
    destaques = []

    for idx, linha in enumerate(linhas[:3]):
        destaque = {**linha}
        destaque["medal"] = medals[idx] if idx < len(medals) else ""
        destaques.append(destaque)

    return destaques


# ------------------------------- Admin ------------------------------------
_TEAM_FIELDS = ("winner1", "winner2", "loser1", "loser2")


def _normalize_admin_date(value: Any) -> date | None:
    if value is None:
        return None

    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None

    return None


def _is_valid_identifier(value: Any) -> bool:
    return value not in (None, "") and value == value


def _identifier_from_match_data(row: Dict[str, Any]) -> tuple[Any | None, str]:
    for field in ("match_id", "id"):
        value = row.get(field)
        if _is_valid_identifier(value):
            return value, field
    return None, "id"


def _players_from_df(df: pd.DataFrame | None) -> List[str]:
    if df is None or df.empty:
        return []

    players: List[str] = []
    seen = set()
    for field in _TEAM_FIELDS:
        if field not in df.columns:
            continue
        for value in df[field].tolist():
            name = str(value or "").strip()
            if name and name not in seen:
                seen.add(name)
                players.append(name)

    players.sort()
    return players


def _matches_from_df(df: pd.DataFrame | None) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    registros: List[Dict[str, Any]] = []
    for row in df.reset_index().to_dict("records"):
        match: Dict[str, Any] = {
            "id": row.get("id"),
            "match_id": row.get("match_id"),
            "winner1": str(row.get("winner1") or "").strip(),
            "winner2": str(row.get("winner2") or "").strip(),
            "loser1": str(row.get("loser1") or "").strip(),
            "loser2": str(row.get("loser2") or "").strip(),
            "score": str(row.get("score") or "").strip(),
        }
        match["date"] = _normalize_admin_date(row.get("date"))
        identifier_value, identifier_field = _identifier_from_match_data(row)
        match["_identifier_value"] = identifier_value
        match["_identifier_field"] = identifier_field
        display_identifier = row.get("match_id") or row.get("id")
        match["identifier_display"] = str(display_identifier) if display_identifier not in (None, "") else ""
        formatted_date = match["date"].isoformat() if match["date"] else "Sem data"
        match["label"] = (
            f"{formatted_date} — {match['winner1']} & {match['winner2']} x "
            f"{match['loser1']} & {match['loser2']}"
        )
        if match["identifier_display"]:
            match["label"] += f" (ID: {match['identifier_display']})"
        registros.append(match)

    registros.sort(key=lambda item: item.get("date") or date.min, reverse=True)
    return registros


def _serialize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    data_value = payload.get("date")
    serialized: Dict[str, Any] = {field: payload.get(field, "").strip() for field in _TEAM_FIELDS}
    if isinstance(data_value, date):
        serialized["date"] = data_value.isoformat()
    else:
        serialized["date"] = str(data_value) if data_value is not None else None
    return serialized


def _validate_match_data(match_id: Any, action: str, payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    if action in {"Atualizar", "Excluir"} and match_id is None:
        errors.append("Selecione uma partida para continuar.")

    if action != "Excluir":
        players = [payload.get(field, "") for field in _TEAM_FIELDS]
        if any(not player for player in players):
            errors.append("Informe os quatro jogadores da partida.")

        if payload.get("winner1") == payload.get("winner2"):
            errors.append("Os vencedores devem ser jogadores diferentes.")

        if payload.get("loser1") == payload.get("loser2"):
            errors.append("Os perdedores devem ser jogadores diferentes.")

        if len({p for p in players if p}) < 4:
            errors.append("Cada jogador só pode aparecer uma vez na partida.")

        if not isinstance(payload.get("date"), date):
            errors.append("Informe uma data válida.")

    return errors


def _reset_cache() -> None:
    _fetch_base_dataframe.cache_clear()

# -------------------------------- Rotas -----------------------------------
@app.route("/")
def home():
    ui_config = _current_ui_config()
    periodos_disponiveis = list(ui_config.ranking_periods)

    modo = request.args.get("modo", "Dias")
    periodo = request.args.get("periodo", ui_config.default_ranking_period)
    inicio = request.args.get("inicio")
    fim = request.args.get("fim")

    base_df = _fetch_base_dataframe()
    datas_index = pd.to_datetime(base_df.index, errors="coerce")
    anos_disponiveis = sorted({str(int(dt.year)) for dt in datas_index if pd.notna(dt)})
    meses_disponiveis = sorted({dt.strftime("%Y-%m") for dt in datas_index if pd.notna(dt)})

    ano = request.args.get("ano", anos_disponiveis[-1] if anos_disponiveis else None)
    mes = request.args.get("mes", meses_disponiveis[-1] if meses_disponiveis else None)

    if modo == "Dias" and periodo not in periodos_disponiveis:
        periodo = "1 dia"
    if modo == "Ano" and ano not in anos_disponiveis:
        ano = anos_disponiveis[-1] if anos_disponiveis else None
    if modo == "Mês/Ano" and mes not in meses_disponiveis:
        mes = meses_disponiveis[-1] if meses_disponiveis else None

    df, jogadores, duplas = _filter_rankings(modo, periodo, ano, mes, inicio, fim)
    periodo_legenda = _descricao_periodo(modo, periodo, ano, mes, inicio, fim)

    jogadores_fmt = _format_ranking(jogadores, "jogadores")
    duplas_fmt = _format_ranking(duplas, "duplas")

    destaques = _build_highlights(jogadores_fmt)

    return render_template(
        "ranking.html",
        active_page="ranking",
        periodo_legenda=periodo_legenda,
        periodo_escolhido=periodo,
        modo=modo,
        periodos=periodos_disponiveis,
        anos=anos_disponiveis,
        ano_selecionado=ano,
        meses=meses_disponiveis,
        mes_selecionado=mes,
        inicio=inicio,
        fim=fim,
        jogos_total=len(df),
        jogos_filtrados=len(df),
        jogadores=dados_with_index(jogadores_fmt),
        duplas=dados_with_index(duplas_fmt),
        destaques=destaques,
    )


@app.route("/infos")
def infos():
    df = _fetch_base_dataframe()
    infos_payload = _resumo_infos(df)

    return render_template(
        "infos.html",
        active_page="infos",
        **infos_payload,
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
    app_config = get_config()
    mensagem = None

    if request.method == "POST":
        update_config(
            min_participation_ratio=_safe_float(
                request.form.get("min_participation_ratio"), app_config.min_participation_ratio
            ),
            min_duo_matches=_safe_int(
                request.form.get("min_duo_matches"), app_config.min_duo_matches
            ),
        )

        app_config = get_config()
        mensagem = "Configurações atualizadas com sucesso!"

    return render_template(
        "config.html",
        active_page="config",
        min_participation_ratio=app_config.min_participation_ratio,
        min_duo_matches=app_config.min_duo_matches,
        mensagem=mensagem,
    )


def _descricao_jogos(
    modo: str, periodo: str | None, ano: str | None, mes: str | None, data: str | None
) -> str:
    if modo == "Mês/Ano" and mes:
        return mes
    if modo == "Ano" and ano:
        return f"Ano {ano}"
    if modo == "Dias" and periodo == "Data" and data:
        return f"Dia {data}"
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
    ui_config = _current_ui_config()
    periodos_disponiveis = list(ui_config.games_periods)

    modo = request.args.get("modo", "Dias")
    periodo = request.args.get("periodo", ui_config.default_games_period)
    data_escolhida = request.args.get("data")

    base_df = _fetch_base_dataframe()
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

    if modo == "Dias":
        if periodo not in periodos_disponiveis:
            periodo = "Todos"
        if periodo == "Data":
            if data_escolhida not in datas_disponiveis:
                data_escolhida = datas_disponiveis[0] if datas_disponiveis else None
            filtro_modo = "Data"
            filtro_valor = data_escolhida
        else:
            filtro_valor = periodo
    elif modo == "Mês/Ano":
        if mes not in meses_disponiveis:
            mes = meses_disponiveis[-1] if meses_disponiveis else None
        filtro_valor = mes
    else:
        if ano not in anos_disponiveis:
            ano = anos_disponiveis[-1] if anos_disponiveis else None
        filtro_valor = ano

    if filtro_valor is None:
        df_filtrado = base_df.iloc[0:0]
    else:
        df_filtrado = filtrar_dados(base_df, filtro_modo, filtro_valor)

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
    partidas_fmt = _formatar_partidas(df_ordenado)

    periodo_legenda = _descricao_jogos(modo, periodo, ano, mes, data_escolhida)

    return render_template(
        "jogos.html",
        active_page="jogos",
        modo=modo,
        periodos=periodos_disponiveis,
        periodo_escolhido=periodo,
        datas=datas_disponiveis,
        data_selecionada=data_escolhida,
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


@app.route("/detalhamento")
def detalhamento():
    df = _fetch_base_dataframe()

    jogadores_disponiveis = sorted(
        {
            j
            for j in df[["winner1", "winner2", "loser1", "loser2"]].values.ravel()
            if isinstance(j, str) and "Outro" not in j
        }
    )

    tipo = request.args.get("tipo", "Jogador")
    if tipo not in {"Jogador", "Dupla"}:
        tipo = "Jogador"

    detalhes = None
    jogador_escolhido = request.args.get("jogador") if tipo == "Jogador" else None
    jogador1 = request.args.get("j1") if tipo == "Dupla" else None
    jogador2 = request.args.get("j2") if tipo == "Dupla" else None
    parceiros_validos = []

    if tipo == "Jogador":
        if jogador_escolhido not in jogadores_disponiveis:
            jogador_escolhido = None

        if jogador_escolhido:
            detalhes = calcular_metricas_jogador(df, jogador_escolhido)
    else:
        parceiros_por_jogador: dict[str, set[str]] = {j: set() for j in jogadores_disponiveis}
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

        if jogador1 not in jogadores_disponiveis:
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
        jogadores=jogadores_disponiveis,
        jogador_escolhido=jogador_escolhido,
        jogador1=jogador1,
        jogador2=jogador2,
        parceiros_validos=parceiros_validos,
        detalhes=detalhes,
    )


def _parse_form_date(value: str | None) -> date | None:
    if not value:
        return None

    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _set_admin_feedback(level: str, message: str) -> None:
    session["admin_feedback"] = {"status": level, "message": message}


@app.route("/admin", methods=["GET", "POST"])
def admin():
    full_password = ADMIN_PASSWORD
    entry_password = MATCH_ENTRY_PASSWORD
    requires_auth = bool(full_password or entry_password)

    feedback = session.pop("admin_feedback", None)
    authenticated = session.get("admin_authenticated") or not requires_auth
    role = session.get("admin_role", "full") if authenticated else None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "login":
            password = request.form.get("password", "")
            if full_password and password == full_password:
                session["admin_authenticated"] = True
                session["admin_role"] = "full"
                _set_admin_feedback("success", "Acesso liberado como administrador completo.")
            elif entry_password and password == entry_password:
                session["admin_authenticated"] = True
                session["admin_role"] = "limited"
                _set_admin_feedback("success", "Acesso liberado para lançamentos.")
            else:
                _set_admin_feedback("error", "Senha incorreta. Tente novamente.")

            return redirect(url_for("admin"))

        if action == "logout":
            session.pop("admin_authenticated", None)
            session.pop("admin_role", None)
            _set_admin_feedback("success", "Sessão encerrada.")
            return redirect(url_for("admin"))

        if not authenticated:
            _set_admin_feedback("error", "Acesso não autorizado. Informe a senha para continuar.")
            return redirect(url_for("admin"))

        if action == "refresh":
            try:
                _reset_cache()
                _fetch_base_dataframe()
                _set_admin_feedback("success", "Cache atualizado com sucesso!")
            except Exception as exc:  # pragma: no cover - feedback exibido na interface
                _set_admin_feedback("error", f"Erro ao atualizar cache: {exc}")

            return redirect(url_for("admin"))

        if role == "limited" and action in {"update", "delete"}:
            _set_admin_feedback(
                "error", "Somente administradores completos podem editar ou excluir partidas."
            )
            return redirect(url_for("admin"))

        if action in {"create", "update"}:
            payload = {
                "winner1": request.form.get("winner1", "").strip(),
                "winner2": request.form.get("winner2", "").strip(),
                "loser1": request.form.get("loser1", "").strip(),
                "loser2": request.form.get("loser2", "").strip(),
                "date": _parse_form_date(request.form.get("date")),
            }

            match_id = request.form.get("match_id") if action == "update" else None
            id_field = request.form.get("id_field", "id")
            errors = _validate_match_data(
                match_id, "Atualizar" if action == "update" else "Adicionar", payload
            )

            if errors:
                _set_admin_feedback("error", " ".join(errors))
                return redirect(url_for("admin"))

            try:
                if action == "create":
                    insert_match(_serialize_payload(payload))
                    _set_admin_feedback("success", "Partida cadastrada com sucesso!")
                else:
                    update_match(match_id, _serialize_payload(payload), id_field=id_field)
                    _set_admin_feedback("success", "Partida atualizada com sucesso!")
                _reset_cache()
            except Exception as exc:  # pragma: no cover - feedback exibido na interface
                _set_admin_feedback("error", f"Erro ao salvar partida: {exc}")

            return redirect(url_for("admin"))

        if action == "delete":
            match_id = request.form.get("match_id")
            id_field = request.form.get("id_field", "id")
            if not _is_valid_identifier(match_id):
                _set_admin_feedback("error", "Selecione uma partida para excluir.")
                return redirect(url_for("admin"))

            try:
                delete_match(match_id, id_field=id_field)
                _reset_cache()
                _set_admin_feedback("success", "Partida removida com sucesso!")
            except Exception as exc:  # pragma: no cover - feedback exibido na interface
                _set_admin_feedback("error", f"Erro ao excluir partida: {exc}")

            return redirect(url_for("admin"))

    df = _fetch_base_dataframe()
    matches = _matches_from_df(df)
    players = _players_from_df(df)

    return render_template(
        "admin.html",
        active_page="admin",
        authenticated=authenticated,
        role=role or "full",
        feedback=feedback,
        requires_auth=requires_auth,
        players=players,
        matches=matches,
        today=date.today().isoformat(),
    )


@app.route("/api/ranking")
def api_ranking():
    periodo = request.args.get("periodo", "90 dias")
    modo = request.args.get("modo", "Dias")
    ano = request.args.get("ano")
    mes = request.args.get("mes")
    inicio = request.args.get("inicio")
    fim = request.args.get("fim")

    df, jogadores, duplas = _filter_rankings(modo, periodo, ano, mes, inicio, fim)
    periodo_legenda = _descricao_periodo(modo, periodo, ano, mes, inicio, fim)

    return jsonify(
        {
            "periodo": periodo_legenda,
            "periodo_param": periodo,
            "modo": modo,
            "intervalo": {"inicio": inicio, "fim": fim} if modo == "Intervalo" else None,
            "total_partidas": len(df),
            "jogadores": _format_ranking(jogadores, "jogadores"),
            "duplas": _format_ranking(duplas, "duplas"),
        }
    )


def dados_with_index(linhas: List[Dict[str, str]]) -> List[Dict[str, str]]:
    for idx, linha in enumerate(linhas, start=1):
        linha["index"] = idx
    return linhas


def _resumo_infos(df: pd.DataFrame) -> Dict[str, object]:
    """Reproduz a lógica da antiga aba de infos em formato não-Streamlit."""

    config = get_config()
    excluidos = _excluded_players()

    df_jog = preparar_dados_individuais(df)
    df_jog = df_jog[~df_jog["jogadores"].str.contains("Outro", na=False)].copy()
    df_jog["vitórias"] = df_jog["vitórias"].astype(int)
    df_jog["derrotas"] = df_jog["derrotas"].astype(int)
    df_jog["jogos"] = df_jog["vitórias"] + df_jog["derrotas"]

    df_duplas = preparar_dados_duplas(df)
    df_duplas = df_duplas[~df_duplas["duplas"].str.contains("Outro", na=False)].copy()

    media_top_10 = df_jog["jogos"].nlargest(10).mean()
    if media_top_10 > 0:
        limiar = media_top_10 * config.min_participation_ratio
        df_jog = df_jog[df_jog["jogos"] >= limiar]

    df_jog["saldo"] = df_jog["vitórias"] - df_jog["derrotas"]
    df_jog = df_jog.set_index("jogadores")
    jogadores_validos = set(df_jog.index)

    dias_jogados = pd.to_datetime(df.index, errors="coerce").normalize().nunique()

    def _melhor_aproveitamento(label: str, pior: bool = False) -> Dict[str, str]:
        v = df_jog["vitórias"]
        d = df_jog["derrotas"]
        jogos = v + d
        stats = df_jog.copy()
        stats["jogos"] = jogos
        stats = stats.drop(index=list(excluidos), errors="ignore")

        if stats.empty:
            return {"title": label, "value": "-", "detail": "-"}

        n = len(stats)
        total_jogos = stats["jogos"].sum()
        media_dos_demais = (total_jogos - stats["jogos"]) / (n - 1) if n > 1 else 0
        limiar = 0.20 * media_dos_demais
        candidatos = stats[stats["jogos"] >= limiar].copy()

        if candidatos.empty:
            return {"title": label, "value": "-", "detail": "-"}

        candidatos["aprov"] = candidatos["vitórias"] / candidatos["jogos"]
        if pior:
            cand_ord = candidatos.sort_values(["aprov", "jogos"], ascending=[True, False])
        else:
            cand_ord = candidatos.sort_values(
                ["aprov", "jogos", "vitórias"], ascending=[False, False, False]
            )

        nome = cand_ord.index[0]
        row = cand_ord.iloc[0]
        return {
            "title": label,
            "value": nome,
            "detail": f"{row['aprov']:.0%} de aproveitamento",
        }

    def _mais_fominha() -> Dict[str, str]:
        if df_jog.empty:
            return {"title": "O mais fominha", "value": "-", "detail": "-"}

        jogostotal = df_jog["vitórias"] + df_jog["derrotas"]
        jogador = jogostotal.idxmax()
        return {
            "title": "O mais fominha",
            "value": jogador,
            "detail": f"Jogos: {jogostotal.max():.0f}",
        }

    def _maior_vexame() -> Dict[str, str]:
        registros: List[Dict[str, object]] = []
        df_dias = df.copy()
        df_dias.index = pd.to_datetime(df_dias.index, errors="coerce")
        df_dias = df_dias[pd.notna(df_dias.index)]

        for dia, df_dia in df_dias.groupby(df_dias.index.normalize()):
            vitorias_dia = pd.Series(df_dia.iloc[:, 0:2].values.ravel()).value_counts()
            derrotas_dia = pd.Series(df_dia.iloc[:, 2:4].values.ravel()).value_counts()

            vitorias_dia = vitorias_dia.drop(list(excluidos), errors="ignore")
            derrotas_dia = derrotas_dia.drop(list(excluidos), errors="ignore")

            jogadores_dia = sorted(set(vitorias_dia.index) | set(derrotas_dia.index))
            jogadores_dia = [j for j in jogadores_dia if j in jogadores_validos]
            if not jogadores_dia:
                continue

            vitorias_dia = vitorias_dia.reindex(jogadores_dia, fill_value=0)
            derrotas_dia = derrotas_dia.reindex(jogadores_dia, fill_value=0)

            saldo_dia = derrotas_dia - vitorias_dia
            if saldo_dia.empty:
                continue

            pior_jogador = saldo_dia.idxmax()
            registros.append(
                {
                    "dia": dia,
                    "jogador": pior_jogador,
                    "saldo": int(saldo_dia.loc[pior_jogador]),
                    "v": int(vitorias_dia.loc[pior_jogador]),
                    "d": int(derrotas_dia.loc[pior_jogador]),
                }
            )

        if not registros:
            return {"title": "O maior vexame na história", "value": "-", "detail": "-"}

        pior = max(registros, key=lambda r: r["saldo"])
        data_fmt = pd.to_datetime(pior["dia"]).strftime("%d/%m/%Y")
        return {
            "title": "O maior vexame na história",
            "value": pior["jogador"],
            "detail": f"{pior['v']}-{pior['d']}  ({data_fmt})",
        }

    def _mais_paneleiro() -> Dict[str, str]:
        EXCLUIR = {"Outro_1", "Outro_2"}

        partner_counts: Dict[str, Counter] = {}
        jogos_por_jogador = Counter()

        for _, row in df.iterrows():
            try:
                w1, w2, l1, l2 = row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3]
            except Exception:
                continue

            duplas = [(w1, w2), (l1, l2)]
            for a, b in duplas:
                if pd.isna(a) or pd.isna(b):
                    continue
                if a in EXCLUIR or b in EXCLUIR:
                    continue
                if a not in jogadores_validos or b not in jogadores_validos:
                    continue
                partner_counts.setdefault(a, Counter())
                partner_counts.setdefault(b, Counter())
                partner_counts[a][b] += 1
                partner_counts[b][a] += 1
                jogos_por_jogador[a] += 1
                jogos_por_jogador[b] += 1

        if not partner_counts:
            return {"title": "O mais paneleiro", "value": "-", "detail": "-"}

        registros = []
        for jog, cnts in partner_counts.items():
            total = sum(cnts.values())
            if total <= 0:
                continue
            parceiro, juntos = max(cnts.items(), key=lambda kv: (kv[1], kv[0]))
            share = juntos / total
            registros.append(
                {
                    "jogador": jog,
                    "parceiro": parceiro,
                    "juntos": int(juntos),
                    "jogos": int(total),
                    "share": float(share),
                }
            )

        if not registros:
            return {"title": "O mais paneleiro", "value": "-", "detail": "-"}

        stats = pd.DataFrame(registros).set_index("jogador")
        n = len(stats)
        media_dos_demais = (stats["jogos"].sum() - stats["jogos"]) / (n - 1) if n > 1 else pd.Series(0, index=stats.index)
        limiar = 0.20 * media_dos_demais
        cand = stats[stats["jogos"] >= limiar].copy()

        if cand.empty:
            return {"title": "O mais paneleiro", "value": "-", "detail": "-"}

        cand = cand.sort_values(
            ["share", "jogos", "juntos", "parceiro"], ascending=[False, False, False, True]
        )
        top = cand.iloc[0]
        return {
            "title": "O mais paneleiro",
            "value": top.name,
            "detail": f"com {top['parceiro']}: {top['share']:.0%} ({int(top['juntos'])}/{int(top['jogos'])} jogos)",
        }

    def _dupla_entrosada() -> Dict[str, str]:
        duplas_validas = df_duplas[df_duplas["jogos"] >= config.min_duo_matches]
        if duplas_validas.empty:
            return {
                "title": f"Dupla mais entrosada (mín de {config.min_duo_matches} jogos)",
                "value": "-",
                "detail": "-",
            }

        melhor_dupla = duplas_validas.iloc[0]

        def _formatar_nome_iniciais(nome_completo: str) -> str:
            partes = nome_completo.strip().split()
            if not partes:
                return nome_completo

            primeiro_nome = partes[0]
            ultimo_nome = partes[-1]

            inicial = primeiro_nome[0]
            return f"{inicial}. {ultimo_nome}"

        nomes_colapsados = [_formatar_nome_iniciais(nome) for nome in str(melhor_dupla["duplas"]).split(" e ")]
        nomes_ordenados = sorted(nomes_colapsados, key=lambda nome: (len(nome), nome))
        dupla_formatada = " e ".join(nomes_ordenados)

        return {
            "title": f"Dupla mais entrosada (mín de {config.min_duo_matches} jogos)",
            "value": dupla_formatada,
            "detail": f"{melhor_dupla['aproveitamento']:.0f}% de aproveitamento",
        }

    destaques_primarios = [
        _melhor_aproveitamento("O mais brabo"),
        _mais_fominha(),
        _dupla_entrosada(),
    ]

    destaques_secundarios = [
        _melhor_aproveitamento("Ninguém quer jogar com", pior=True),
        _maior_vexame(),
        _mais_paneleiro(),
    ]

    return {
        "resumo": {
            "total_partidas": len(df),
            "dias_jogados": dias_jogados,
            "total_minutos": len(df) * 20,
        },
        "destaques_primarios": destaques_primarios,
        "destaques_secundarios": destaques_secundarios,
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
