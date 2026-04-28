"""
Classify parsed bank statement CSV rows using historical expense data
already stored in Supabase.

This script is intentionally conservative:
- it only updates rows when it finds a strong historical match;
- it uses the existing `expenses` table as the source of truth;
- it leaves rows untouched when the match is ambiguous or weak.

Default input:
    C:/Users/diego/Downloads/extratos/despesas_import.csv

Default output:
    C:/Users/diego/Downloads/extratos/despesas_import_classificado.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


DEFAULT_INPUT = Path(r"C:\Users\diego\Downloads\extratos\despesas_import.csv")
DEFAULT_OUTPUT = Path(r"C:\Users\diego\Downloads\extratos\despesas_import_classificado.csv")
DEFAULT_BATCH_SIZE = 1000
MIN_FUZZY_SIMILARITY = 0.91
MIN_MARGIN = 0.05

_INSTALLMENT_RE = re.compile(r"\s*\(\d{1,2}/\d{1,2}\)\s*$")
_LEADING_CODE_RE = re.compile(r"^\d{2,}(?:\s+\d{2,})+\s+")
_MERCHANT_PREFIX_RE = re.compile(
    r"^(?:mp|ec|ifd|pag|pagseguro|iugu|pix|ted|doc|mercadopago|merc pago)\s*\*?\s*",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Classification:
    description: str
    category: str
    tag: str
    expense_scope: str


@dataclass(frozen=True)
class Candidate:
    key: str
    classification: Classification
    count: int


@dataclass(frozen=True)
class MappingRule:
    description: str
    category: str
    tag: str
    expense_scope: str


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    text = strip_accents(text or "").lower().strip()
    text = _INSTALLMENT_RE.sub("", text)
    text = _LEADING_CODE_RE.sub("", text)
    text = _MERCHANT_PREFIX_RE.sub("", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def dominant_classification(counter: Counter[Classification]) -> tuple[Classification | None, int]:
    if not counter:
        return None, 0
    (best_cls, best_count), *rest = counter.most_common(2)
    total = sum(counter.values())
    second_count = rest[0][1] if rest else 0
    if best_count == 1 and total > 1:
        return None, total
    if second_count and best_count / total < 0.70:
        return None, total
    return best_cls, total


def build_history_indexes(expenses: list[dict]) -> tuple[dict[str, Candidate], dict[str, Candidate]]:
    by_bank_desc: dict[str, Counter[Classification]] = defaultdict(Counter)
    by_desc: dict[str, Counter[Classification]] = defaultdict(Counter)

    for expense in expenses:
        classification = Classification(
            description=(expense.get("description") or "").strip(),
            category=(expense.get("category") or "").strip(),
            tag=(expense.get("tag") or "").strip(),
            expense_scope=((expense.get("expense_scope") or "shared").strip() or "shared"),
        )
        if not classification.description or not classification.category:
            continue

        bank_key = normalize_text(expense.get("bank_description") or "")
        desc_key = normalize_text(expense.get("description") or "")

        if bank_key:
            by_bank_desc[bank_key][classification] += 1
        if desc_key:
            by_desc[desc_key][classification] += 1

    bank_index: dict[str, Candidate] = {}
    desc_index: dict[str, Candidate] = {}

    for key, counter in by_bank_desc.items():
        classification, total = dominant_classification(counter)
        if classification:
            bank_index[key] = Candidate(key=key, classification=classification, count=total)

    for key, counter in by_desc.items():
        classification, total = dominant_classification(counter)
        if classification:
            desc_index[key] = Candidate(key=key, classification=classification, count=total)

    return bank_index, desc_index


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        token_score = 0.0
    else:
        overlap = len(tokens_a & tokens_b)
        token_score = overlap / max(len(tokens_a), len(tokens_b))

    seq_score = SequenceMatcher(None, a, b).ratio()
    return max(seq_score, token_score)


def find_fuzzy_candidate(key: str, index: dict[str, Candidate]) -> Candidate | None:
    if not key:
        return None

    scored: list[tuple[float, Candidate]] = []
    key_tokens = set(key.split())

    for candidate in index.values():
        candidate_tokens = set(candidate.key.split())
        if key_tokens and candidate_tokens and not (key_tokens & candidate_tokens):
            continue
        score = similarity(key, candidate.key)
        if score >= MIN_FUZZY_SIMILARITY:
            scored.append((score, candidate))

    if not scored:
        return None

    scored.sort(key=lambda item: (item[0], item[1].count), reverse=True)
    best_score, best_candidate = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0

    if best_score < MIN_FUZZY_SIMILARITY:
        return None
    if second_score and best_score - second_score < MIN_MARGIN:
        return None
    return best_candidate


def choose_candidate(
    bank_key: str,
    desc_key: str,
    bank_index: dict[str, Candidate],
    desc_index: dict[str, Candidate],
) -> tuple[Candidate | None, str]:
    if bank_key and bank_key in bank_index:
        return bank_index[bank_key], "exact_bank_description"
    if desc_key and desc_key in desc_index:
        return desc_index[desc_key], "exact_description"

    candidate = find_fuzzy_candidate(bank_key, bank_index)
    if candidate:
        return candidate, "fuzzy_bank_description"

    candidate = find_fuzzy_candidate(desc_key, desc_index)
    if candidate:
        return candidate, "fuzzy_description"

    return None, "unmatched"


def load_supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
    )
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / key not configured in environment")

    from supabase import create_client  # noqa: PLC0415

    return create_client(url, key)


def load_expenses_history(batch_size: int = DEFAULT_BATCH_SIZE) -> list[dict]:
    client = load_supabase_client()
    rows: list[dict] = []
    start = 0

    while True:
        response = (
            client.table("expenses")
            .select("description, category, tag, expense_scope, bank_description")
            .eq("is_deleted", False)
            .order("id")
            .range(start, start + batch_size - 1)
            .execute()
        )
        batch = response.data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < batch_size:
            break
        start += batch_size

    return rows


def load_expense_mappings() -> dict[str, MappingRule]:
    client = load_supabase_client()

    try:
        response = (
            client.table("expense_mappings")
            .select("expense_name, category, tag, expense_scope")
            .eq("is_active", True)
            .execute()
        )
    except Exception:
        response = (
            client.table("expense_mappings")
            .select("expense_name, category, tag")
            .eq("is_active", True)
            .execute()
        )

    mappings: dict[str, MappingRule] = {}
    for row in response.data or []:
        expense_name = (row.get("expense_name") or "").strip()
        if not expense_name:
            continue
        key = normalize_text(expense_name)
        mappings[key] = MappingRule(
            description=expense_name,
            category=(row.get("category") or "").strip(),
            tag=(row.get("tag") or "").strip(),
            expense_scope=((row.get("expense_scope") or "shared").strip() or "shared"),
        )
    return mappings


def classify_rows(
    rows: list[dict],
    bank_index: dict[str, Candidate],
    desc_index: dict[str, Candidate],
    mappings: dict[str, MappingRule],
) -> dict[str, int]:
    stats = Counter()

    for row in rows:
        bank_description = (row.get("bank_description") or "").strip()
        description = (row.get("description") or "").strip()
        bank_key = normalize_text(bank_description)
        desc_key = normalize_text(description)

        candidate, reason = choose_candidate(bank_key, desc_key, bank_index, desc_index)
        if not candidate:
            stats["unmatched"] += 1
            continue

        resolved = candidate.classification
        mapping = mappings.get(normalize_text(resolved.description))
        if mapping:
            resolved = Classification(
                description=mapping.description or resolved.description,
                category=mapping.category or resolved.category,
                tag=mapping.tag,
                expense_scope=mapping.expense_scope or resolved.expense_scope,
            )
            stats["mapping_defaults_applied"] += 1

        row["description"] = resolved.description
        row["category"] = resolved.category
        row["tag"] = resolved.tag
        row["expense_scope"] = resolved.expense_scope

        stats["matched"] += 1
        stats[reason] += 1

    return dict(stats)


def read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not reader.fieldnames:
            raise RuntimeError("CSV has no header row")
        return rows, list(reader.fieldnames)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify parsed expense CSV rows using historical data from Supabase."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise RuntimeError(f"Input CSV not found: {input_path}")

    rows, fieldnames = read_csv(input_path)
    history = load_expenses_history()
    if not history:
        raise RuntimeError("No expense history found in Supabase")
    mappings = load_expense_mappings()

    bank_index, desc_index = build_history_indexes(history)
    stats = classify_rows(rows, bank_index, desc_index, mappings)
    write_csv(output_path, rows, fieldnames)

    print(f"Input CSV: {input_path}")
    print(f"Output CSV: {output_path}")
    print(f"Rows processed: {len(rows)}")
    print(f"Rows classified: {stats.get('matched', 0)}")
    print(f"Rows left unchanged: {stats.get('unmatched', 0)}")
    print(f"Exact bank_description matches: {stats.get('exact_bank_description', 0)}")
    print(f"Exact description matches: {stats.get('exact_description', 0)}")
    print(f"Fuzzy bank_description matches: {stats.get('fuzzy_bank_description', 0)}")
    print(f"Fuzzy description matches: {stats.get('fuzzy_description', 0)}")
    print(f"Mapping defaults applied: {stats.get('mapping_defaults_applied', 0)}")
    print(f"Historical examples loaded: {len(history)}")
    print(f"Expense mappings loaded: {len(mappings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
