from __future__ import annotations
from batch_endpoints import bp as batch_bp

"""Aplicação Flask para exibir o ranking em HTML estático."""

from collections import Counter
import os
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from app_settings import get_config, update_config
from awards import available_awards_years, get_awards_for_year
from config import ADMIN_PASSWORD, MATCH_ENTRY_PASSWORD
from detalhamento import calcular_metricas_dupla, calcular_metricas_jogador
from extraction import get_matches
from preparation import preparar_dataframe
from processing import filtrar_dados, preparar_dados_duplas, preparar_dados_individuais
from data_access.supabase_repository import delete_match, insert_match, update_match
from data_access.player_registry import add_player, load_registered_players
from ui_config import get_ui_config

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")
app.register_blueprint(batch_bp)

def _current_ui_config():
    """Retorna a configuração de UI imutável."""

    return get_ui_config()


def _build_awards_data(year: int) -> List[Dict[str, Any]]:
    awards = get_awards_for_year(year)
    for award in awards:
        sorted_votes = sorted(
            award["votes"].items(), key=lambda item: item[1], reverse=True
        )
        award["top_three"] = sorted_votes[:3]
        winner_votes = sorted_votes[0][1]
        award["winners"] = [
            {"name": name, "votes": votes}
            for name, votes in sorted_votes
            if votes == winner_votes
        ]
        award["winner_votes"] = winner_votes
        award["total_votes"] = sum(award["votes"].values())

    return awards


def _format_champion_names(names: List[str]) -> str:
    if not names:
        return "-"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} e {names[1]}"
    return f"{', '.join(names[:-1])} e {names[-1]}"


def _build_awards_champions(awards: List[Dict[str, Any]]) -> Dict[str, Any]:
    trophies = {
        "positive": Counter(),
        "negative": Counter(),
    }

    for award in awards:
        category = award.get("category_type", "positive")
        for winner in award.get("winners", []):
            trophies[category][winner["name"]] += 1

    champions = {}
    for category, counts in trophies.items():
        if not counts:
            champions[category] = {"names": "-", "count": 0}
            continue
        max_count = max(counts.values())
        names = sorted([name for name, count in counts.items() if count == max_count])
        champions[category] = {
            "names": _format_champion_names(names),
            "count": max_count,
        }

    return champions


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

    aplicar_minimos = not (modo == "Dias" and periodo == "1 dia")

    if aplicar_minimos:
        media_top_10 = jogadores["jogos"].nlargest(10).mean()
        if media_top_10 > 0:
            limiar = media_top_10 * config.min_participation_ratio
            jogadores = jogadores[jogadores["jogos"] >= limiar]

        # Para janelas curtas, reduza dinamicamente o mínimo de jogos de duplas
        # para evitar que o ranking fique vazio quando houver poucas partidas.
        if config.min_duo_matches > 0 and not duplas.empty:
            max_jogos_duplas = int(duplas["jogos"].max())
            limiar_duplas = (
                config.min_duo_matches
                if max_jogos_duplas >= config.min_duo_matches
                else max_jogos_duplas
            )

            if limiar_duplas > 0:
                duplas = duplas[duplas["jogos"] >= limiar_duplas]

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


def _jogadores_disponiveis(df: pd.DataFrame) -> List[str]:
    jogadores = {
        j
        for j in df[["winner1", "winner2", "loser1", "loser2"]].values.ravel()
        if isinstance(j, str) and "Outro" not in j
    }

    excluidos = _excluded_players()
    return sorted(jogadores - excluidos)


def _duplas_disponiveis(df: pd.DataFrame) -> List[str]:
    duplas = set(df["dupla_winner"].tolist() + df["dupla_loser"].tolist())
    duplas = {d for d in duplas if isinstance(d, str) and "Outro" not in d}
    return sorted(duplas)


def _oponentes_por_jogador(df: pd.DataFrame) -> Dict[str, List[str]]:
    jogadores = set(_jogadores_disponiveis(df))
    oponentes: Dict[str, set[str]] = {j: set() for j in jogadores}

    def _limpar_nome(valor: Any) -> str | None:
        if not isinstance(valor, str):
            return None
        nome = valor.strip()
        if not nome or "Outro" in nome or nome not in jogadores:
            return None
        return nome

    for _, linha in df.iterrows():
        w1 = _limpar_nome(linha.get("winner1"))
        w2 = _limpar_nome(linha.get("winner2"))
        l1 = _limpar_nome(linha.get("loser1"))
        l2 = _limpar_nome(linha.get("loser2"))

        winners = [p for p in (w1, w2) if p]
        losers = [p for p in (l1, l2) if p]

        for vencedor in winners:
            oponentes[vencedor].update(losers)
        for perdedor in losers:
            oponentes[perdedor].update(winners)

    return {j: sorted(lista) for j, lista in oponentes.items()}


def _parceiros_por_jogador(df: pd.DataFrame) -> Dict[str, List[str]]:
    jogadores = set(_jogadores_disponiveis(df))
    parceiros: Dict[str, set[str]] = {j: set() for j in jogadores}

    for _, row in df.iterrows():
        duplas_partida = [
            [row.get("winner1"), row.get("winner2")],
            [row.get("loser1"), row.get("loser2")],
        ]
        for jogador_a, jogador_b in duplas_partida:
            if not jogador_a or not jogador_b:
                continue
            if jogador_a not in jogadores or jogador_b not in jogadores:
                continue
            parceiros[jogador_a].add(jogador_b)
            parceiros[jogador_b].add(jogador_a)

    return {j: sorted(lista) for j, lista in parceiros.items()}


def _oponentes_por_dupla_jogadores(df: pd.DataFrame) -> Dict[str, List[str]]:
    jogadores = set(_jogadores_disponiveis(df))
    duplas = set(_duplas_disponiveis(df))
    oponentes: Dict[str, set[str]] = {dupla: set() for dupla in duplas}

    for _, row in df.iterrows():
        dupla_winner = row.get("dupla_winner")
        dupla_loser = row.get("dupla_loser")
        winners = [row.get("winner1"), row.get("winner2")]
        losers = [row.get("loser1"), row.get("loser2")]

        if dupla_winner in duplas:
            oponentes[dupla_winner].update([j for j in losers if j in jogadores])
        if dupla_loser in duplas:
            oponentes[dupla_loser].update([j for j in winners if j in jogadores])

    return {dupla: sorted(lista) for dupla, lista in oponentes.items()}


def _oponentes_por_dupla(df: pd.DataFrame) -> Dict[str, List[str]]:
    duplas = set(_duplas_disponiveis(df))
    oponentes: Dict[str, set[str]] = {d: set() for d in duplas}

    for _, linha in df.iterrows():
        vencedor = linha.get("dupla_winner")
        perdedor = linha.get("dupla_loser")

        if vencedor not in duplas or perdedor not in duplas:
            continue

        oponentes[vencedor].add(perdedor)
        oponentes[perdedor].add(vencedor)

    return {dupla: sorted(lista) for dupla, lista in oponentes.items()}


def _estatisticas_jogador_individual(df: pd.DataFrame, jogador: str) -> Dict[str, int | float]:
    tabela = preparar_dados_individuais(df)
    tabela = tabela[~tabela["jogadores"].isin(_excluded_players())]
    linha = tabela.loc[tabela["jogadores"] == jogador]

    if linha.empty:
        return {
            "jogador": jogador,
            "jogos": 0,
            "vitorias": 0,
            "derrotas": 0,
            "saldo": 0,
            "aproveitamento": 0.0,
        }

    registro = linha.iloc[0]
    return {
        "jogador": jogador,
        "jogos": int(registro.get("jogos", 0)),
        "vitorias": int(registro.get("vitórias", 0)),
        "derrotas": int(registro.get("derrotas", 0)),
        "saldo": int(registro.get("saldo", 0)),
        "aproveitamento": float(registro.get("aproveitamento", 0.0)),
    }


def _estatisticas_dupla(df: pd.DataFrame, dupla: str) -> Dict[str, int | float]:
    tabela = preparar_dados_duplas(df[["winner1", "winner2", "loser1", "loser2"]])
    tabela = tabela[~tabela["duplas"].str.contains("Outro", na=False)]
    linha = tabela.loc[tabela["duplas"] == dupla]

    if linha.empty:
        return {
            "jogador": dupla,
            "jogos": 0,
            "vitorias": 0,
            "derrotas": 0,
            "saldo": 0,
            "aproveitamento": 0.0,
        }

    registro = linha.iloc[0]
    return {
        "jogador": dupla,
        "jogos": int(registro.get("jogos", 0)),
        "vitorias": int(registro.get("vitórias", 0)),
        "derrotas": int(registro.get("derrotas", 0)),
        "saldo": int(registro.get("saldo", 0)),
        "aproveitamento": float(registro.get("aproveitamento", 0.0)),
    }


def _confronto_direto(df: pd.DataFrame, jogador1: str, jogador2: str) -> Dict[str, Any]:
    if jogador1 == jogador2:
        return {
            "total": 0,
            "vitorias_j1": 0,
            "vitorias_j2": 0,
            "saldo": 0,
            "serie_mensal": [],
        }

    mask_j1_win = (
        ((df["winner1"] == jogador1) | (df["winner2"] == jogador1))
        & ((df["loser1"] == jogador2) | (df["loser2"] == jogador2))
    )
    mask_j2_win = (
        ((df["winner1"] == jogador2) | (df["winner2"] == jogador2))
        & ((df["loser1"] == jogador1) | (df["loser2"] == jogador1))
    )

    confrontos_df = df[mask_j1_win | mask_j2_win].copy()

    if confrontos_df.empty:
        return {
            "total": 0,
            "vitorias_j1": 0,
            "vitorias_j2": 0,
            "saldo": 0,
            "serie_mensal": [],
        }

    def _resultado_j1(row: pd.Series) -> int:
        if (row["winner1"] == jogador1) or (row["winner2"] == jogador1):
            return 1
        if (row["loser1"] == jogador1) or (row["loser2"] == jogador1):
            return -1
        return 0

    confrontos_df["resultado_j1"] = confrontos_df.apply(_resultado_j1, axis=1)

    if not isinstance(confrontos_df.index, pd.DatetimeIndex):
        confrontos_df.index = pd.to_datetime(confrontos_df.index, errors="coerce")

    if getattr(confrontos_df.index, "tz", None) is not None:
        confrontos_df.index = confrontos_df.index.tz_convert(None)

    confrontos_df = confrontos_df[~confrontos_df.index.isna()]

    if confrontos_df.empty:
        return {
            "total": 0,
            "vitorias_j1": int(mask_j1_win.sum()),
            "vitorias_j2": int(mask_j2_win.sum()),
            "saldo": int(mask_j1_win.sum() - mask_j2_win.sum()),
            "serie_mensal": [],
        }

    inicio = confrontos_df.index.min().normalize().replace(day=1)
    fim = pd.Timestamp.now().normalize().replace(day=1)
    meses_range = pd.date_range(start=inicio, end=fim, freq="MS")

    mensal = (
        confrontos_df["resultado_j1"]
        .resample("MS")
        .sum()
        .reindex(meses_range, fill_value=0)
        .sort_index()
    )
    saldo_acumulado = mensal.cumsum()

    serie_completa = [
        {
            "label": periodo.strftime("%b/%Y"),
            "saldo": int(mensal.loc[periodo]),
            "acumulado": int(saldo_acumulado.loc[periodo]),
        }
        for periodo in mensal.index
    ]

    serie_mensal = serie_completa[-12:]

    vitorias_j1 = int(mask_j1_win.sum())
    vitorias_j2 = int(mask_j2_win.sum())

    return {
        "total": len(confrontos_df),
        "vitorias_j1": vitorias_j1,
        "vitorias_j2": vitorias_j2,
        "saldo": vitorias_j1 - vitorias_j2,
        "serie_mensal": serie_mensal,
    }


def _confronto_direto_duplas(df: pd.DataFrame, dupla1: str, dupla2: str) -> Dict[str, Any]:
    if dupla1 == dupla2:
        return {
            "total": 0,
            "vitorias_j1": 0,
            "vitorias_j2": 0,
            "saldo": 0,
            "serie_mensal": [],
        }

    mask_dupla1_win = (df["dupla_winner"] == dupla1) & (df["dupla_loser"] == dupla2)
    mask_dupla2_win = (df["dupla_winner"] == dupla2) & (df["dupla_loser"] == dupla1)
    confrontos_df = df[mask_dupla1_win | mask_dupla2_win].copy()

    if confrontos_df.empty:
        return {
            "total": 0,
            "vitorias_j1": 0,
            "vitorias_j2": 0,
            "saldo": 0,
            "serie_mensal": [],
        }

    confrontos_df["resultado_j1"] = confrontos_df["dupla_winner"].apply(
        lambda valor: 1 if valor == dupla1 else -1
    )

    if not isinstance(confrontos_df.index, pd.DatetimeIndex):
        confrontos_df.index = pd.to_datetime(confrontos_df.index, errors="coerce")

    if getattr(confrontos_df.index, "tz", None) is not None:
        confrontos_df.index = confrontos_df.index.tz_convert(None)

    confrontos_df = confrontos_df[~confrontos_df.index.isna()]

    if confrontos_df.empty:
        return {
            "total": int(mask_dupla1_win.sum() + mask_dupla2_win.sum()),
            "vitorias_j1": int(mask_dupla1_win.sum()),
            "vitorias_j2": int(mask_dupla2_win.sum()),
            "saldo": int(mask_dupla1_win.sum() - mask_dupla2_win.sum()),
            "serie_mensal": [],
        }

    inicio = confrontos_df.index.min().normalize().replace(day=1)
    fim = pd.Timestamp.now().normalize().replace(day=1)
    meses_range = pd.date_range(start=inicio, end=fim, freq="MS")

    mensal = (
        confrontos_df["resultado_j1"]
        .resample("MS")
        .sum()
        .reindex(meses_range, fill_value=0)
        .sort_index()
    )
    saldo_acumulado = mensal.cumsum()

    serie_completa = [
        {
            "label": periodo.strftime("%b/%Y"),
            "saldo": int(mensal.loc[periodo]),
            "acumulado": int(saldo_acumulado.loc[periodo]),
        }
        for periodo in mensal.index
    ]

    serie_mensal = serie_completa[-12:]

    vitorias_dupla1 = int(mask_dupla1_win.sum())
    vitorias_dupla2 = int(mask_dupla2_win.sum())

    return {
        "total": len(confrontos_df),
        "vitorias_j1": vitorias_dupla1,
        "vitorias_j2": vitorias_dupla2,
        "saldo": vitorias_dupla1 - vitorias_dupla2,
        "serie_mensal": serie_mensal,
    }

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


def _registered_players(df: pd.DataFrame | None) -> List[str]:
    """Combina jogadores das partidas com os cadastrados manualmente."""

    base_players = set(_players_from_df(df))
    manual_players = {name for name in load_registered_players() if name}
    excluidos = _excluded_players()
    combined = sorted((base_players | manual_players) - excluidos)
    return combined


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


def _parse_bulk_line(line: str) -> Dict[str, str] | None:
    """Converte uma linha no formato "a e b x c e d" em um dicionário de campos."""

    import re

    pattern = re.compile(
        r"^(?P<w1>.+?)\s+e\s+(?P<w2>.+?)\s+x\s+(?P<l1>.+?)\s+e\s+(?P<l2>.+)$",
        flags=re.IGNORECASE,
    )
    match = pattern.match(line.strip())
    if not match:
        return None

    return {
        "winner1": match.group("w1").strip(),
        "winner2": match.group("w2").strip(),
        "loser1": match.group("l1").strip(),
        "loser2": match.group("l2").strip(),
    }


def _validate_registered_players(players: Iterable[str]) -> List[str]:
    registrados = set(_registered_players(_fetch_base_dataframe()))
    return [name for name in players if name not in registrados]


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


@app.route("/awards")
def awards():
    available_years = available_awards_years()
    default_year = available_years[0] if available_years else date.today().year
    selected_year = _safe_int(request.args.get("year"), default_year)
    if selected_year not in available_years:
        selected_year = default_year

    awards_data = _build_awards_data(selected_year)
    total_votes = sum(award["total_votes"] for award in awards_data)
    champions = _build_awards_champions(awards_data)
    positive_awards = [award for award in awards_data if award["category_type"] == "positive"]
    negative_awards = [award for award in awards_data if award["category_type"] == "negative"]
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

    jogadores_disponiveis = _jogadores_disponiveis(df)

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


@app.route("/versus")
def versus():
    df = _fetch_base_dataframe()
    tipo = request.args.get("tipo", "Jogador")
    if tipo not in ("Jogador", "Dupla"):
        tipo = "Jogador"

    parceiros_por_jogador = {}
    duplas_selecao = {
        "d1a": request.args.get("d1a"),
        "d1b": request.args.get("d1b"),
        "d2a": request.args.get("d2a"),
        "d2b": request.args.get("d2b"),
    }
    jogador1 = request.args.get("j1")
    jogador2 = request.args.get("j2")

    if tipo == "Dupla":
        jogadores_disponiveis = _jogadores_disponiveis(df)
        parceiros_por_jogador = _parceiros_por_jogador(df)
        oponentes_por_dupla_jogadores = _oponentes_por_dupla_jogadores(df)
        oponentes_por_jogador = {}

        for chave, valor in duplas_selecao.items():
            if valor not in jogadores_disponiveis:
                duplas_selecao[chave] = None

        def _dupla_valida(primeiro: str | None, segundo: str | None) -> bool:
            if not primeiro or not segundo or primeiro == segundo:
                return False
            return segundo in parceiros_por_jogador.get(primeiro, [])

        if not _dupla_valida(duplas_selecao["d1a"], duplas_selecao["d1b"]):
            duplas_selecao["d1b"] = None
        if not _dupla_valida(duplas_selecao["d2a"], duplas_selecao["d2b"]):
            duplas_selecao["d2b"] = None
    else:
        jogadores_disponiveis = _jogadores_disponiveis(df)
        oponentes_por_jogador = _oponentes_por_jogador(df)
        oponentes_por_dupla_jogadores = {}

        if jogador1 not in jogadores_disponiveis:
            jogador1 = None
        if jogador2 not in jogadores_disponiveis:
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
            oponentes_dupla1 = set(oponentes_por_dupla_jogadores.get(dupla1_nome, []))
            if duplas_selecao["d2a"] not in oponentes_dupla1:
                duplas_selecao["d2a"] = None
                duplas_selecao["d2b"] = None
            elif duplas_selecao["d2b"] and duplas_selecao["d2b"] not in oponentes_dupla1:
                duplas_selecao["d2b"] = None

        if duplas_selecao["d2a"] and duplas_selecao["d2b"]:
            parceiros_validos_d2a = set(parceiros_por_jogador.get(duplas_selecao["d2a"], []))
            if duplas_selecao["d2b"] not in parceiros_validos_d2a:
                duplas_selecao["d2b"] = None

        if duplas_selecao["d2a"] and duplas_selecao["d2b"]:
            dupla2_nome = " e ".join(sorted([duplas_selecao["d2a"], duplas_selecao["d2b"]]))

        if dupla1_nome and dupla2_nome and dupla1_nome != dupla2_nome:
            estatisticas = {
                "jogador1": _estatisticas_dupla(df, dupla1_nome),
                "jogador2": _estatisticas_dupla(df, dupla2_nome),
            }
            confronto = _confronto_direto_duplas(df, dupla1_nome, dupla2_nome)
    else:
        if jogador1 and jogador2 and jogador1 != jogador2:
            estatisticas = {
                "jogador1": _estatisticas_jogador_individual(df, jogador1),
                "jogador2": _estatisticas_jogador_individual(df, jogador2),
            }
            confronto = _confronto_direto(df, jogador1, jogador2)

    return render_template(
        "versus.html",
        active_page="versus",
        tipo=tipo,
        jogadores=jogadores_disponiveis,
        jogador1=jogador1,
        jogador2=jogador2,
        dupla1_nome=dupla1_nome,
        dupla2_nome=dupla2_nome,
        duplas_selecao=duplas_selecao,
        parceiros_por_jogador=parceiros_por_jogador,
        oponentes_por_dupla_jogadores=oponentes_por_dupla_jogadores,
        oponentes_por_jogador=oponentes_por_jogador,
        estatisticas=estatisticas,
        confronto=confronto,
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

        if action == "add_player":
            player_name = request.form.get("player_name", "").strip()
            if not player_name:
                _set_admin_feedback("error", "Informe o nome do jogador.")
                return redirect(url_for("admin"))

            if add_player(player_name):
                _set_admin_feedback("success", f"Jogador '{player_name}' cadastrado com sucesso!")
            else:
                _set_admin_feedback("error", "Jogador já cadastrado ou nome inválido.")

            return redirect(url_for("admin"))

        if role == "limited" and action in {"update", "delete"}:
            _set_admin_feedback(
                "error", "Somente administradores completos podem editar ou excluir partidas."
            )
            return redirect(url_for("admin"))

        if action == "bulk_create":
            bulk_matches = request.form.get("bulk_matches", "")
            match_date = _parse_form_date(request.form.get("bulk_date")) or date.today()

            linhas = [linha.strip() for linha in bulk_matches.splitlines() if linha.strip()]
            if not linhas:
                _set_admin_feedback("error", "Informe ao menos uma linha de partida.")
                return redirect(url_for("admin"))

            parsed_matches: List[Dict[str, Any]] = []
            errors: List[str] = []
            missing_players: set[str] = set()

            for idx, linha in enumerate(linhas, start=1):
                parsed = _parse_bulk_line(linha)
                if not parsed:
                    errors.append(f"Linha {idx} em formato inválido.")
                    continue

                players = [parsed.get(field, "") for field in _TEAM_FIELDS]
                missing_players.update(_validate_registered_players(players))
                payload = {**parsed, "date": match_date}
                validation_errors = _validate_match_data(None, "Adicionar", payload)
                if validation_errors:
                    errors.extend([f"Linha {idx}: {erro}" for erro in validation_errors])
                else:
                    parsed_matches.append(payload)

            if missing_players:
                errors.append(
                    "Jogadores não cadastrados: " + ", ".join(sorted(missing_players))
                )

            if errors:
                _set_admin_feedback("error", " ".join(errors))
                return redirect(url_for("admin"))

            try:
                for payload in parsed_matches:
                    insert_match(_serialize_payload(payload))
                _reset_cache()
                _set_admin_feedback(
                    "success", f"{len(parsed_matches)} partida(s) cadastrada(s) em bloco!"
                )
            except Exception as exc:  # pragma: no cover - feedback exibido na interface
                _set_admin_feedback("error", f"Erro ao salvar partidas em bloco: {exc}")

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
    players = _registered_players(df)
    players_text = "\n".join(players)

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
