from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from .awards_2025 import AWARDS_2025

AWARDS_BY_YEAR = {
    2025: AWARDS_2025,
}


def available_awards_years() -> List[int]:
    return sorted(AWARDS_BY_YEAR.keys(), reverse=True)


def get_awards_for_year(year: int) -> List[Dict[str, Any]]:
    if year not in AWARDS_BY_YEAR:
        raise KeyError(f"Awards data for year {year} not found")
    return deepcopy(AWARDS_BY_YEAR[year])
