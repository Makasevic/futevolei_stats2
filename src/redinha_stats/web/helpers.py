from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import pandas as pd


def normalize_filter_mode(value: str | None) -> str:
    """Normaliza nomes de filtros legados para valores internos estaveis."""

    normalized = (value or "Dia").strip()
    if normalized == "Dias":
        return "Dia"
    if normalized in {"Mes/Ano", "MÃªs/Ano", "MÃƒÂªs/Ano"}:
        return "Mes/Ano"
    return normalized


def describe_period(
    modo: str,
    periodo: str | None,
    ano: str | None,
    mes: str | None,
    inicio: str | None,
    fim: str | None,
    data: str | None,
) -> str:
    if modo == "Ano" and ano:
        return f"Ano {ano}"
    if modo == "Mes/Ano" and mes:
        return mes
    if modo == "Intervalo":
        if inicio or fim:
            return f"Intervalo {inicio or '...'} a {fim or '...'}"
        return "Intervalo personalizado"
    if modo == "Data" and data:
        return f"Dia {data}"
    return periodo or "Todos"


def format_ranking(df: pd.DataFrame, nome_col: str) -> List[Dict[str, str]]:
    medalhas = ["\U0001F947", "\U0001F948", "\U0001F949"]
    linhas: List[Dict[str, str]] = []
    total_linhas = len(df)

    for idx, linha in df.iterrows():
        if idx < len(medalhas):
            posicao = medalhas[idx]
        else:
            posicao = "\U0001F631" if idx == total_linhas - 1 else f"{idx + 1:02d}"
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


def build_highlights(linhas: List[Dict[str, str]]) -> List[Dict[str, str]]:
    medals = ["\U0001F947", "\U0001F948", "\U0001F949"]
    destaques = []

    for idx, linha in enumerate(linhas[:3]):
        destaque = {**linha}
        destaque["medal"] = medals[idx] if idx < len(medals) else ""
        destaques.append(destaque)

    return destaques


def safe_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def describe_games(
    modo: str,
    periodo: str | None,
    ano: str | None,
    mes: str | None,
    data: str | None,
    inicio: str | None,
    fim: str | None,
) -> str:
    if modo == "Mes/Ano" and mes:
        return mes
    if modo == "Ano" and ano:
        return f"Ano {ano}"
    if modo == "Data" and data:
        return f"Dia {data}"
    if modo == "Intervalo":
        if inicio or fim:
            return f"Intervalo {inicio or '...'} a {fim or '...'}"
        return "Intervalo personalizado"
    return periodo or "Todos"


def format_matches(df: pd.DataFrame) -> List[Dict[str, str]]:
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


def with_index(linhas: List[Dict[str, str]]) -> List[Dict[str, str]]:
    for idx, linha in enumerate(linhas, start=1):
        linha["index"] = idx
    return linhas


def serialize_payload(payload: Dict[str, Any], team_fields: List[str]) -> Dict[str, Any]:
    data_value = payload.get("date")
    serialized: Dict[str, Any] = {
        field: payload.get(field, "").strip() for field in team_fields
    }
    if isinstance(data_value, date):
        serialized["date"] = data_value.isoformat()
    else:
        serialized["date"] = str(data_value) if data_value is not None else None
    if "score" in payload:
        serialized["score"] = payload.get("score")
    return serialized


def parse_bulk_line(line: str) -> Dict[str, str] | None:
    """Converte uma linha no formato "a e b x c e d" em um dicionario de campos."""

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
