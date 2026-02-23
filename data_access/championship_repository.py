"""Persistencia de placares de campeonato no Supabase."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from postgrest.exceptions import APIError
from supabase import Client, create_client

from config import SUPABASE_SERVICE_KEY, SUPABASE_URL


CHAMPIONSHIP_SCORES_TABLE = "championship_scores"


@lru_cache(maxsize=1)
def _get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def fetch_scores_for_championship(championship_key: str) -> Dict[str, Dict[str, int]]:
    try:
        response = (
            _get_client()
            .table(CHAMPIONSHIP_SCORES_TABLE)
            .select("match_id,score_a,score_b")
            .eq("championship_key", championship_key)
            .execute()
        )
    except APIError as exc:
        code = getattr(exc, "code", None)
        if code == "PGRST205":
            return {}
        raise
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao buscar placares de campeonato: {response.error}")

    scores: Dict[str, Dict[str, int]] = {}
    for row in (response.data or []):
        match_id = str(row.get("match_id") or "").strip()
        score_a = row.get("score_a")
        score_b = row.get("score_b")
        if match_id and isinstance(score_a, int) and isinstance(score_b, int):
            scores[match_id] = {"score_a": score_a, "score_b": score_b}
    return scores


def upsert_score(championship_key: str, match_id: str, score_a: int, score_b: int) -> None:
    payload = {
        "championship_key": championship_key,
        "match_id": match_id,
        "score_a": score_a,
        "score_b": score_b,
    }
    try:
        response = (
            _get_client()
            .table(CHAMPIONSHIP_SCORES_TABLE)
            .upsert(payload, on_conflict="championship_key,match_id")
            .execute()
        )
    except APIError as exc:
        code = getattr(exc, "code", None)
        if code == "PGRST205":
            raise RuntimeError(
                "Tabela championship_scores nao encontrada no Supabase. Rode o SQL de criacao/migracao."
            ) from exc
        raise
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao salvar placar de campeonato: {response.error}")


def delete_score(championship_key: str, match_id: str) -> None:
    try:
        response = (
            _get_client()
            .table(CHAMPIONSHIP_SCORES_TABLE)
            .delete()
            .eq("championship_key", championship_key)
            .eq("match_id", match_id)
            .execute()
        )
    except APIError as exc:
        code = getattr(exc, "code", None)
        if code == "PGRST205":
            raise RuntimeError(
                "Tabela championship_scores nao encontrada no Supabase. Rode o SQL de criacao/migracao."
            ) from exc
        raise
    if getattr(response, "error", None):
        raise RuntimeError(f"Erro ao remover placar de campeonato: {response.error}")
