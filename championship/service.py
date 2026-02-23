from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from functools import cmp_to_key
from typing import Any, Dict, Iterable, List

from data_access.championship_repository import (
    delete_score,
    fetch_scores_for_championship,
    upsert_score,
)
from .championship_2026_02 import CHAMPIONSHIP_2026_02


_CHAMPIONSHIPS = {
    CHAMPIONSHIP_2026_02["key"]: CHAMPIONSHIP_2026_02,
}


def available_championship_keys() -> List[str]:
    return sorted(_CHAMPIONSHIPS.keys(), reverse=True)


def get_championship_edit_password(championship_key: str) -> str | None:
    config = _CHAMPIONSHIPS.get(championship_key, {})
    password = config.get("edit_password")
    if password is None:
        return None
    normalized = str(password).strip()
    return normalized or None


def _empty_stats() -> Dict[str, int]:
    return {
        "jogos": 0,
        "vitorias": 0,
        "derrotas": 0,
        "pontos_pro": 0,
        "pontos_contra": 0,
        "saldo": 0,
    }


def _state_for_championship(championship_key: str) -> Dict[str, Any]:
    return {"matches": fetch_scores_for_championship(championship_key)}


def _list_group_matches(group_id: str, team_ids: List[str]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    idx = 1
    for i in range(len(team_ids)):
        for j in range(i + 1, len(team_ids)):
            matches.append(
                {
                    "id": f"{group_id}_M{idx}",
                    "phase": "groups",
                    "group_id": group_id,
                    "team_a": team_ids[i],
                    "team_b": team_ids[j],
                    "score_a": None,
                    "score_b": None,
                }
            )
            idx += 1
    return matches


def _build_matches(template: Dict[str, Any]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for group in template["groups"]:
        matches.extend(_list_group_matches(group["id"], group["team_ids"]))

    matches.extend(
        [
            {"id": "QF1", "phase": "quarterfinals", "team_a": None, "team_b": None, "score_a": None, "score_b": None},
            {"id": "QF2", "phase": "quarterfinals", "team_a": None, "team_b": None, "score_a": None, "score_b": None},
            {"id": "QF3", "phase": "quarterfinals", "team_a": None, "team_b": None, "score_a": None, "score_b": None},
            {"id": "QF4", "phase": "quarterfinals", "team_a": None, "team_b": None, "score_a": None, "score_b": None},
            {"id": "SF1", "phase": "semifinals", "team_a": None, "team_b": None, "score_a": None, "score_b": None},
            {"id": "SF2", "phase": "semifinals", "team_a": None, "team_b": None, "score_a": None, "score_b": None},
            {"id": "FINAL", "phase": "final", "team_a": None, "team_b": None, "score_a": None, "score_b": None},
        ]
    )
    return matches


def _apply_saved_scores(matches: List[Dict[str, Any]], saved: Dict[str, Any]) -> None:
    by_id = {match["id"]: match for match in matches}
    for match_id, payload in saved.items():
        match = by_id.get(match_id)
        if not match:
            continue
        score_a = payload.get("score_a")
        score_b = payload.get("score_b")
        if isinstance(score_a, int) and isinstance(score_b, int):
            match["score_a"] = score_a
            match["score_b"] = score_b


def _is_played(match: Dict[str, Any]) -> bool:
    return isinstance(match.get("score_a"), int) and isinstance(match.get("score_b"), int)


def _winner(match: Dict[str, Any]) -> str | None:
    if not _is_played(match):
        return None
    if match["score_a"] == match["score_b"]:
        return None
    return match["team_a"] if match["score_a"] > match["score_b"] else match["team_b"]


def _head_to_head(a: str, b: str, matches: Iterable[Dict[str, Any]]) -> int:
    wins_a = 0
    wins_b = 0
    saldo_a = 0
    for match in matches:
        if not _is_played(match):
            continue
        teams = {match.get("team_a"), match.get("team_b")}
        if teams != {a, b}:
            continue
        score_a = int(match["score_a"])
        score_b = int(match["score_b"])
        if match["team_a"] == a:
            saldo_a += score_a - score_b
            if score_a > score_b:
                wins_a += 1
            elif score_b > score_a:
                wins_b += 1
        else:
            saldo_a += score_b - score_a
            if score_b > score_a:
                wins_a += 1
            elif score_a > score_b:
                wins_b += 1
    if wins_a != wins_b:
        return wins_a - wins_b
    if saldo_a != 0:
        return saldo_a
    return 0


def _sort_teams(
    team_ids: List[str],
    stats: Dict[str, Dict[str, int]],
    matches_for_head_to_head: List[Dict[str, Any]],
    team_names: Dict[str, str],
) -> List[str]:
    def _compare(a: str, b: str) -> int:
        a_stats = stats[a]
        b_stats = stats[b]
        if a_stats["vitorias"] != b_stats["vitorias"]:
            return b_stats["vitorias"] - a_stats["vitorias"]
        if a_stats["saldo"] != b_stats["saldo"]:
            return b_stats["saldo"] - a_stats["saldo"]

        h2h = _head_to_head(a, b, matches_for_head_to_head)
        if h2h != 0:
            return -1 if h2h > 0 else 1

        if a_stats["pontos_pro"] != b_stats["pontos_pro"]:
            return b_stats["pontos_pro"] - a_stats["pontos_pro"]

        a_name = team_names.get(a, a)
        b_name = team_names.get(b, b)
        if a_name.casefold() < b_name.casefold():
            return -1
        if a_name.casefold() > b_name.casefold():
            return 1
        return 0

    return sorted(team_ids, key=cmp_to_key(_compare))


def _accumulate_stats(matches: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(_empty_stats)
    for match in matches:
        team_a = match.get("team_a")
        team_b = match.get("team_b")
        if not team_a or not team_b or not _is_played(match):
            continue

        score_a = int(match["score_a"])
        score_b = int(match["score_b"])

        stats[team_a]["jogos"] += 1
        stats[team_b]["jogos"] += 1

        stats[team_a]["pontos_pro"] += score_a
        stats[team_a]["pontos_contra"] += score_b
        stats[team_b]["pontos_pro"] += score_b
        stats[team_b]["pontos_contra"] += score_a

        stats[team_a]["saldo"] = stats[team_a]["pontos_pro"] - stats[team_a]["pontos_contra"]
        stats[team_b]["saldo"] = stats[team_b]["pontos_pro"] - stats[team_b]["pontos_contra"]

        if score_a > score_b:
            stats[team_a]["vitorias"] += 1
            stats[team_b]["derrotas"] += 1
        elif score_b > score_a:
            stats[team_b]["vitorias"] += 1
            stats[team_a]["derrotas"] += 1
    return stats


def _set_knockout_teams(
    matches_by_id: Dict[str, Dict[str, Any]],
    seeds: List[str],
) -> None:
    def _clear_match(match: Dict[str, Any]) -> None:
        match["team_a"] = None
        match["team_b"] = None
        match["score_a"] = None
        match["score_b"] = None

    def _seed(idx: int) -> str | None:
        return seeds[idx] if idx < len(seeds) else None

    group_matches = [m for m in matches_by_id.values() if m["phase"] == "groups"]
    qf_matches = [matches_by_id[f"QF{i}"] for i in range(1, 5)]
    sf_matches = [matches_by_id[f"SF{i}"] for i in range(1, 3)]
    final_match = matches_by_id["FINAL"]

    groups_completed = bool(group_matches) and all(_is_played(m) for m in group_matches)

    if groups_completed:
        matches_by_id["QF1"]["team_a"] = _seed(0)
        matches_by_id["QF1"]["team_b"] = _seed(7)
        matches_by_id["QF2"]["team_a"] = _seed(3)
        matches_by_id["QF2"]["team_b"] = _seed(4)
        matches_by_id["QF3"]["team_a"] = _seed(1)
        matches_by_id["QF3"]["team_b"] = _seed(6)
        matches_by_id["QF4"]["team_a"] = _seed(2)
        matches_by_id["QF4"]["team_b"] = _seed(5)
    else:
        for match in qf_matches:
            _clear_match(match)

    quarterfinals_completed = groups_completed and all(_is_played(m) for m in qf_matches)

    if quarterfinals_completed:
        matches_by_id["SF1"]["team_a"] = _winner(matches_by_id["QF1"])
        matches_by_id["SF1"]["team_b"] = _winner(matches_by_id["QF2"])
        matches_by_id["SF2"]["team_a"] = _winner(matches_by_id["QF3"])
        matches_by_id["SF2"]["team_b"] = _winner(matches_by_id["QF4"])
    else:
        for match in sf_matches:
            _clear_match(match)

    semifinals_completed = quarterfinals_completed and all(_is_played(m) for m in sf_matches)

    if semifinals_completed:
        final_match["team_a"] = _winner(matches_by_id["SF1"])
        final_match["team_b"] = _winner(matches_by_id["SF2"])
    else:
        _clear_match(final_match)


def _public_match(match: Dict[str, Any], team_names: Dict[str, str], label: str | None = None) -> Dict[str, Any]:
    return {
        "id": match["id"],
        "label": label or match["id"],
        "team_a_id": match.get("team_a"),
        "team_b_id": match.get("team_b"),
        "team_a": team_names.get(match.get("team_a"), "-") if match.get("team_a") else "-",
        "team_b": team_names.get(match.get("team_b"), "-") if match.get("team_b") else "-",
        "score_a": match.get("score_a"),
        "score_b": match.get("score_b"),
        "hint_a": "",
        "hint_b": "",
    }


def _build_phase_summary(
    matches: List[Dict[str, Any]],
    team_names: Dict[str, str],
) -> Dict[str, List[Dict[str, Any]]]:
    phase_labels = {
        "groups": "Grupos",
        "quarterfinals": "Quartas",
        "semifinals": "Semi Final",
        "final": "Final",
    }
    grouped: Dict[str, List[Dict[str, Any]]] = {v: [] for v in phase_labels.values()}
    for phase, label in phase_labels.items():
        stats = _accumulate_stats([m for m in matches if m["phase"] == phase])
        items = []
        for team_id, st in stats.items():
            items.append(
                {
                    "team": team_names.get(team_id, team_id),
                    **st,
                }
            )
        items.sort(key=lambda row: (-row["vitorias"], -row["saldo"], -row["pontos_pro"], row["team"].casefold()))
        grouped[label] = items
    return grouped


def get_championship_view(championship_key: str) -> Dict[str, Any]:
    if championship_key not in _CHAMPIONSHIPS:
        raise KeyError(f"Championship data for key {championship_key} not found")

    template = deepcopy(_CHAMPIONSHIPS[championship_key])
    teams = template["teams"]
    groups = template["groups"]
    def _display_name(name: str) -> str:
        parts = [p for p in str(name).strip().split() if p]
        if not parts:
            return str(name).strip()
        connectors = {"e", "de", "da", "do", "dos", "das"}
        normalized = []
        for idx, part in enumerate(parts):
            lower = part.lower()
            if idx > 0 and lower in connectors:
                normalized.append(lower)
            else:
                normalized.append(part[:1].upper() + part[1:].lower())
        return " ".join(normalized)

    team_names = {team["id"]: _display_name(team["name"]) for team in teams}
    team_groups: Dict[str, str] = {}
    team_group_colors: Dict[str, str] = {}
    unique_palette = [
        "#E8F3FF", "#EAF8EE", "#FFF5E8", "#F4ECFF", "#FFEFF5",
        "#E8FFF8", "#FFF3E0", "#EAF4FF", "#F0F9E8", "#FFF0E8",
        "#F1ECFF", "#E8FBFF", "#FFF8E8", "#ECF8FF", "#FDEBFF",
        "#E9FFE8", "#FFEDE8", "#EAF0FF", "#F2FFE8", "#FFF0F8",
    ]
    team_index_map = {team["id"]: idx for idx, team in enumerate(teams)}
    for group in groups:
        for idx, team_id in enumerate(group["team_ids"]):
            team_groups[team_id] = group["name"]
            palette_idx = team_index_map.get(team_id, idx) % len(unique_palette)
            team_group_colors[team_id] = unique_palette[palette_idx]

    matches = _build_matches(template)
    state = _state_for_championship(championship_key)
    _apply_saved_scores(matches, state.get("matches", {}))
    matches_by_id = {match["id"]: match for match in matches}

    group_tables = []
    qualified: List[str] = []
    qualified_by_group_completed: set[str] = set()
    eliminated_by_group_completed: set[str] = set()
    group_completed_by_group_id: Dict[str, bool] = {}
    team_group_id: Dict[str, str] = {}
    group_matches = [match for match in matches if match["phase"] == "groups"]
    for group in groups:
        g_matches = [m for m in group_matches if m.get("group_id") == group["id"]]
        group_is_completed = all(_is_played(match) for match in g_matches) if g_matches else False
        group_completed_by_group_id[group["id"]] = group_is_completed

        stats = _accumulate_stats(g_matches)
        for team_id in group["team_ids"]:
            stats[team_id]
            team_group_id[team_id] = group["id"]

        ranking_ids = _sort_teams(group["team_ids"], stats, g_matches, team_names)
        qualified.extend(ranking_ids[:2])
        if group_is_completed:
            qualified_by_group_completed.update(ranking_ids[:2])
            eliminated_by_group_completed.update(ranking_ids[2:])
        ranking_rows = []
        for position, team_id in enumerate(ranking_ids, start=1):
            row_stats = stats[team_id]
            row_status = "pending"
            if group_is_completed and position <= 2:
                row_status = "qualified"
            elif group_is_completed and position > 2:
                row_status = "eliminated"
            ranking_rows.append(
                {
                    "posicao": position,
                    "team_id": team_id,
                    "team": team_names[team_id],
                    "group_stage_status": row_status,
                    **row_stats,
                }
            )

        group_tables.append(
            {
                "id": group["id"],
                "name": group["name"],
                "ranking": ranking_rows,
                "matches": [
                    _public_match(match, team_names, label=f"Jogo {idx}")
                    for idx, match in enumerate(g_matches, start=1)
                ],
            }
        )

    qualified_unique = []
    for team_id in qualified:
        if team_id and team_id not in qualified_unique:
            qualified_unique.append(team_id)

    qualified_stats = _accumulate_stats(group_matches)
    for team_id in qualified_unique:
        qualified_stats[team_id]

    general_seeding_ids = _sort_teams(
        qualified_unique,
        qualified_stats,
        group_matches,
        team_names,
    )

    _set_knockout_teams(matches_by_id, general_seeding_ids)

    quarterfinals = [_public_match(matches_by_id[f"QF{i}"], team_names, label=f"QF{i}") for i in range(1, 5)]
    semifinals = [_public_match(matches_by_id[f"SF{i}"], team_names, label=f"SF{i}") for i in range(1, 3)]
    final = _public_match(matches_by_id["FINAL"], team_names, label="Final")

    qf_hints = [
        ("1º geral", "8º geral"),
        ("4º geral", "5º geral"),
        ("2º geral", "7º geral"),
        ("3º geral", "6º geral"),
    ]
    for idx, match in enumerate(quarterfinals):
        if not match.get("team_a_id"):
            match["team_a"] = qf_hints[idx][0]
            match["hint_a"] = "Regra de chaveamento"
        if not match.get("team_b_id"):
            match["team_b"] = qf_hints[idx][1]
            match["hint_b"] = "Regra de chaveamento"

    sf_hints = [
        ("1º quartas", "2º quartas"),
        ("3º quartas", "4º quartas"),
    ]
    for idx, match in enumerate(semifinals):
        if not match.get("team_a_id"):
            match["team_a"] = sf_hints[idx][0]
            match["hint_a"] = "Regra de chaveamento"
        if not match.get("team_b_id"):
            match["team_b"] = sf_hints[idx][1]
            match["hint_b"] = "Regra de chaveamento"

    if not final.get("team_a_id"):
        final["team_a"] = "1º semi final"
        final["hint_a"] = "Regra de chaveamento"
    if not final.get("team_b_id"):
        final["team_b"] = "2º semi final"
        final["hint_b"] = "Regra de chaveamento"

    overall_stats = _accumulate_stats(matches)
    overall_rows = []
    for team in teams:
        team_id = team["id"]
        row_stats = overall_stats.get(team_id, _empty_stats())
        overall_rows.append({"team_id": team_id, "team": team["name"], **row_stats})
    overall_rows.sort(
        key=lambda row: (-row["vitorias"], -row["saldo"], -row["pontos_pro"], row["team"].casefold())
    )

    all_group_team_ids = [team["id"] for team in teams]
    all_group_stats = _accumulate_stats(group_matches)
    for team_id in all_group_team_ids:
        all_group_stats[team_id]

    all_general_ids = _sort_teams(
        all_group_team_ids,
        all_group_stats,
        group_matches,
        team_names,
    )
    qualified_set = set(general_seeding_ids[:8])

    general_classification = []
    for idx, team_id in enumerate(all_general_ids, start=1):
        row_stats = all_group_stats[team_id]
        group_id = team_group_id.get(team_id)
        group_completed = bool(group_completed_by_group_id.get(group_id, False))
        group_stage_status = "pending"
        if group_completed and team_id in qualified_by_group_completed:
            group_stage_status = "qualified"
        elif group_completed and team_id in eliminated_by_group_completed:
            group_stage_status = "eliminated"
        general_classification.append(
            {
                "posicao": idx,
                "team_id": team_id,
                "team": team_names[team_id],
                "classificado": team_id in qualified_set,
                "group_stage_status": group_stage_status,
                **row_stats,
            }
        )

    phase_summary = _build_phase_summary(matches, team_names)

    return {
        "key": championship_key,
        "title": template["title"],
        "teams": teams,
        "groups": group_tables,
        "general_classification": general_classification,
        "quarterfinals": quarterfinals,
        "semifinals": semifinals,
        "final": final,
        "team_groups": team_groups,
        "team_group_colors": team_group_colors,
        "overall_summary": overall_rows,
        "phase_summary": phase_summary,
        "editable_matches": [
            _public_match(match, team_names)
            | {"phase": match["phase"], "group_id": match.get("group_id")}
            for match in matches
        ],
    }


def _parse_score(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("Placar invalido: use numeros inteiros.") from exc
    if parsed < 0:
        raise ValueError("Placar invalido: nao pode ser negativo.")
    return parsed


def save_match_score(championship_key: str, match_id: str, score_a_raw: str | None, score_b_raw: str | None) -> None:
    if championship_key not in _CHAMPIONSHIPS:
        raise ValueError("Campeonato nao encontrado.")

    view = get_championship_view(championship_key)
    match_map = {m["id"]: m for m in view["editable_matches"]}
    target = match_map.get(match_id)
    if not target:
        raise ValueError("Partida nao encontrada.")

    score_a = _parse_score(score_a_raw)
    score_b = _parse_score(score_b_raw)

    if (score_a is None) != (score_b is None):
        raise ValueError("Preencha os dois lados do placar ou deixe ambos vazios.")

    # Limpeza de placar deve ser sempre permitida, mesmo se o confronto
    # de mata-mata ainda nao tiver adversarios definidos.
    if score_a is None and score_b is None:
        delete_score(championship_key, match_id)
        return

    if target["phase"] in {"quarterfinals", "semifinals", "final"}:
        if not target.get("team_a_id") or not target.get("team_b_id"):
            raise ValueError("A partida ainda nao tem adversarios definidos.")
        if score_a is not None and score_b is not None and score_a == score_b:
            raise ValueError("Mata-mata nao pode terminar empatado.")

    upsert_score(championship_key, match_id, score_a, score_b)
