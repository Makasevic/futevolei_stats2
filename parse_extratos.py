"""
Parse bank statements (extratos/faturas) from BTG and Itaú into CSV format
compatible with the Expenses app import feature.

CSV columns (full import format):
    transaction_date, amount, description, category, expense_scope, tag,
    bank_description, obs, schedule_mode, schedule_count, occurrence_index,
    occurrence_label

`description`/`category`/`tag` are inferred via fuzzy lookup on the
Supabase `expense_mappings` table. When no mapping match is found, a
clean name is heuristically derived from the bank description (title-cased
first meaningful token), so `description` is never empty.

`import_fingerprint` is intentionally omitted — the app computes it on
the fly during import (see `apps/my_life/app.py::parse_csv_import`).
"""

import csv
import io
import os
import re
import sys
import unicodedata
from datetime import datetime, date
from difflib import SequenceMatcher

import msoffcrypto
import openpyxl
import pdfplumber
import xlrd
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Supabase-backed mapping lookup (expense_mappings → description/category/tag)
# ---------------------------------------------------------------------------

def load_mappings_from_supabase() -> list[dict]:
    """Load active expense_mappings from Supabase. Returns [] on any failure."""
    url = os.environ.get("SUPABASE_URL")
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
    )
    if not url or not key:
        print("  WARNING: SUPABASE_URL / chave ausente — sem inferência de despesa/tag.")
        return []
    try:
        from supabase import create_client
        client = create_client(url, key)
        resp = (
            client.table("expense_mappings")
            .select("expense_name, category, tag")
            .eq("is_active", True)
            .order("expense_name")
            .execute()
        )
        rows = resp.data or []
        return [
            {"expense_name": r["expense_name"], "category": r["category"], "tag": r.get("tag") or ""}
            for r in rows
        ]
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: falha ao carregar expense_mappings do Supabase: {e}")
        return []


# ---------------------------------------------------------------------------
# Fuzzy expense matching + heuristic fallback (never leaves description empty)
# ---------------------------------------------------------------------------

_MERCHANT_PREFIX_RE = re.compile(
    r"^(?:mp|ec|ifd|pag|pagseguro|iug|pix|ted|doc)\s*\*\s*", re.IGNORECASE
)
_INSTALLMENT_RE = re.compile(r"\s*\(\d{1,2}/\d{1,2}\)\s*$")


def _normalize(s: str) -> str:
    """Lowercase, strip accents and non-alphanumeric noise."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _clean_for_match(raw_desc: str) -> str:
    """Remove installment suffix and merchant prefixes before matching."""
    s = _INSTALLMENT_RE.sub("", raw_desc or "")
    s = _MERCHANT_PREFIX_RE.sub("", s.strip())
    return s


def match_expense(raw_desc: str, mappings: list[dict]) -> tuple[str, str, str]:
    """Fuzzy-match a bank description against expense_mappings.

    Cascade:
      1. Normalized substring (mapping name appears inside description, or vice-versa).
      2. Token overlap: a significant token (>= 4 chars) of the mapping name
         appears inside the description.
      3. Best SequenceMatcher ratio >= 0.72.
    Returns ("", "", "") only when nothing clears any bar.
    """
    if not mappings:
        return "", "", ""

    cleaned = _clean_for_match(raw_desc)
    desc_norm = _normalize(cleaned)
    if not desc_norm:
        return "", "", ""

    best: tuple[float, dict] | None = None

    for m in mappings:
        name_norm = _normalize(m["expense_name"])
        if not name_norm:
            continue

        score = 0.0
        if name_norm == desc_norm:
            score = 1.0
        elif name_norm in desc_norm or desc_norm in name_norm:
            # Longer mapping names winning avoids "Luz" beating "Luzia Fono".
            score = 0.9 + (len(name_norm) / 200.0)
        else:
            tokens = [t for t in name_norm.split() if len(t) >= 4]
            if any(t in desc_norm for t in tokens):
                score = 0.8
            else:
                ratio = SequenceMatcher(None, name_norm, desc_norm).ratio()
                if ratio >= 0.72:
                    score = ratio

        if score and (best is None or score > best[0]):
            best = (score, m)

    if best:
        m = best[1]
        return m["expense_name"], m["category"], m.get("tag") or ""
    return "", "", ""


def heuristic_description(raw_desc: str) -> str:
    """Derive a clean display name from a bank description.

    Strips installment suffix and merchant prefixes, picks up to three
    meaningful alphabetic tokens and Title-Cases them.
    """
    cleaned = _clean_for_match(raw_desc)
    words = re.findall(r"[A-Za-zÀ-ÿ]+", cleaned)
    words = [w for w in words if len(w) >= 3]
    if not words:
        fallback = (raw_desc or "").strip() or "Outros"
        return fallback[:40]
    return " ".join(words[:3]).title()[:40]


# ---------------------------------------------------------------------------
# BTG Credit Card (xlsx, password-protected)
# ---------------------------------------------------------------------------

def parse_btg_credit_card(filepath: str, password: str) -> list[dict]:
    """Parse BTG fatura xlsx (password-protected)."""
    rows = []
    with open(filepath, "rb") as fh:
        decrypted = io.BytesIO()
        ms = msoffcrypto.OfficeFile(fh)
        ms.load_key(password=password)
        ms.decrypt(decrypted)
        decrypted.seek(0)

    wb = openpyxl.load_workbook(decrypted)
    ws = wb["Titular"]

    for row in ws.iter_rows(values_only=True):
        # Transaction rows have a datetime in col 1, description in col 2, amount in col 4
        dt = row[1]
        desc = row[2]
        amount = row[4]
        card = row[7] if len(row) > 7 else row[6] if len(row) > 6 else "0000"

        if not isinstance(dt, datetime):
            continue
        if not isinstance(amount, (int, float)):
            continue
        if amount <= 0:
            continue
        if not desc or not str(desc).strip():
            continue

        # Skip payment/credit rows
        desc_str = str(desc).strip()
        if "pagamento de fatura" in desc_str.lower():
            continue
        if "cancelamento" in desc_str.lower():
            continue

        rows.append({
            "data": dt.strftime("%m/%d/%Y"),
            "descricao_original": desc_str,
            "valor": f"{amount:.2f}",
            "despesa": desc_str,
            "banco": "BTG",
            "cartao": str(card or "0000").strip(),
        })

    return rows


# ---------------------------------------------------------------------------
# BTG Checking Account (xls)
# ---------------------------------------------------------------------------

def parse_btg_extrato(filepath: str) -> list[dict]:
    """Parse BTG extrato de conta corrente xls."""
    rows = []
    wb = xlrd.open_workbook(filepath)

    for sn in wb.sheet_names():
        ws = wb.sheet_by_name(sn)
        for r in range(ws.nrows):
            date_raw = str(ws.cell_value(r, 1)).strip()
            category = str(ws.cell_value(r, 2)).strip()
            transaction = str(ws.cell_value(r, 3)).strip()
            description = str(ws.cell_value(r, 6)).strip()
            amount_raw = ws.cell_value(r, 10)

            # Skip non-transaction rows
            if not date_raw or "Data e hora" in date_raw:
                continue
            if "Saldo" in description:
                continue
            if not isinstance(amount_raw, (int, float)):
                continue
            if amount_raw >= 0:
                # Only expenses (negative values), skip income
                continue

            # Skip card payments and self-transfers
            desc_lower = description.lower()
            if "fatura do cart" in desc_lower:
                continue
            if description.lower().startswith("diego"):
                continue

            # Parse date (format: DD/MM/YYYY HH:MM)
            date_match = re.match(r"(\d{2}/\d{2}/\d{4})", date_raw)
            if not date_match:
                continue

            amount = abs(amount_raw)
            desc = f"{transaction} - {description}".strip(" -")
            if not desc:
                desc = description or category

            rows.append({
                "data": datetime.strptime(date_match.group(1), "%d/%m/%Y").strftime("%m/%d/%Y"),
                "descricao_original": desc,
                "valor": f"{amount:.2f}",
                "despesa": desc,
                "banco": "BTG",
                "cartao": "0000",
            })

    return rows


# ---------------------------------------------------------------------------
# Itaú Credit Card (PDF) - complex two-column layout
# ---------------------------------------------------------------------------

def _infer_year(month: int, fatura_year: int, fatura_month: int) -> int:
    """Infer the year for a transaction given the fatura date.
    Transactions can be from the current year or previous year if installments."""
    if month > fatura_month + 2:
        return fatura_year - 1
    return fatura_year


def _extract_itau_transactions_from_page(page, col_boundary: float) -> list[dict]:
    """Extract transactions from a single Itau fatura page using word positions.

    Returns list of dicts with keys: date_str, description, amount_str
    for each column (left and right).
    """
    words = page.extract_words(keep_blank_chars=True, x_tolerance=1, y_tolerance=2)

    # Group words by line (y-position)
    line_map = {}
    for w in words:
        y = round(w["top"], 0)
        if y not in line_map:
            line_map[y] = []
        line_map[y].append(w)

    results = []
    date_re = re.compile(r"^\d{2}/\d{2}$")
    amount_re = re.compile(r"^-?[\d.]+,\d{2}$")

    for y in sorted(line_map.keys()):
        line_words = sorted(line_map[y], key=lambda w: w["x0"])

        # Split into left and right column words
        left_words = [w for w in line_words if w["x0"] < col_boundary]
        right_words = [w for w in line_words if w["x0"] >= col_boundary]

        for col_words in [left_words, right_words]:
            if not col_words:
                continue

            # Look for pattern: DATE_WORD DESCRIPTION_WORDS... AMOUNT_WORD
            first_text = col_words[0]["text"]
            if not date_re.match(first_text):
                continue
            if len(col_words) < 2:
                continue

            last_text = col_words[-1]["text"]
            if not amount_re.match(last_text):
                continue

            # Description is everything between date and amount
            desc_parts = [w["text"] for w in col_words[1:-1]]
            desc = " ".join(desc_parts).strip()
            if not desc:
                continue

            results.append({
                "date_str": first_text,
                "description": desc,
                "amount_str": last_text,
            })

    return results


def parse_itau_fatura(filepath: str) -> list[dict]:
    """Parse Itaú credit card fatura PDF using word-level extraction."""
    rows = []

    with pdfplumber.open(filepath) as pdf:
        # Detect card number from first page
        first_text = pdf.pages[0].extract_text() or ""
        card_match = re.search(r"(\d{4})\.XXXX\.XXXX\.(\d{4})", first_text)
        card_last4 = card_match.group(2) if card_match else "0000"

        # Detect fatura date (vencimento)
        venc_match = re.search(
            r"[Vv]encimento[:\s]*(\d{2}/\d{2}/\d{4})", first_text
        ) or re.search(
            r"[Vv]encimentoem:\s*(\d{2}/\d{2}/\d{4})", first_text
        )
        if venc_match:
            venc_date = datetime.strptime(venc_match.group(1), "%d/%m/%Y")
            fatura_year = venc_date.year
            fatura_month = venc_date.month
        else:
            fatura_year = 2026
            fatura_month = 4

        current_card = card_last4

        for page in pdf.pages:
            page_text = page.extract_text() or ""

            # Detect card section changes on this page
            for final_m in re.finditer(r"\(final\s*(\d{4})\)", page_text):
                # Use the last card seen on the page
                current_card = final_m.group(1)

            # Column boundary: typically around x=365-370
            col_boundary = page.width * 0.62

            txns = _extract_itau_transactions_from_page(page, col_boundary)

            for tx in txns:
                desc = tx["description"]

                # Skip headers and non-transaction lines
                if any(s in desc.upper() for s in [
                    "ESTABELECIMENTO", "VALOR EM", "VALOREMR",
                    "PROGRAMA", "CASHBACK",
                ]):
                    continue

                # Keep installment suffix for differentiation but clean it up
                # e.g., "Volarecomerciode 12/12" -> "Volarecomerciode (12/12)"
                installment_match = re.search(r"\s*(\d{2}/\d{2})\s*$", desc)
                installment_tag = ""
                if installment_match:
                    installment_tag = f" ({installment_match.group(1)})"
                    desc_clean = desc[:installment_match.start()].strip()
                else:
                    # Check for NN/NN glued at end
                    installment_match = re.search(r"(\d{2}/\d{2})$", desc)
                    if installment_match:
                        installment_tag = f" ({installment_match.group(1)})"
                        desc_clean = desc[:installment_match.start()].strip()
                    else:
                        desc_clean = desc.strip()
                if not desc_clean:
                    continue

                # Remove IFD* prefix
                desc_clean = re.sub(r"^IFD\*", "", desc_clean)
                desc_final = desc_clean + installment_tag

                # Parse amount
                amount_str = tx["amount_str"].replace(".", "").replace(",", ".")
                try:
                    amount = float(amount_str)
                except ValueError:
                    continue
                if amount <= 0:
                    continue

                # Parse date
                day = int(tx["date_str"][:2])
                month = int(tx["date_str"][3:5])
                year = _infer_year(month, fatura_year, fatura_month)
                try:
                    tx_date = date(year, month, day)
                except ValueError:
                    continue

                rows.append({
                    "data": tx_date.strftime("%m/%d/%Y"),
                    "descricao_original": desc_final,
                    "valor": f"{amount:.2f}",
                    "despesa": desc_clean,
                    "banco": "Itau",
                    "cartao": current_card,
                })

    return rows


# ---------------------------------------------------------------------------
# XP Credit Card (CSV, semicolon-delimited)
# ---------------------------------------------------------------------------

def parse_xp_fatura(filepath: str) -> list[dict]:
    """Parse XP fatura CSV (semicolon-delimited, Brazilian format)."""
    rows = []
    with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            date_str = (r.get("Data") or "").strip()
            desc = (r.get("Estabelecimento") or "").strip()
            amount_str = (r.get("Valor") or "").strip()
            parcela = (r.get("Parcela") or "").strip()

            if not date_str or not desc:
                continue

            # Parse amount (Brazilian format: "R$ 19,90" or "R$ -1.490,14")
            amt_clean = amount_str.replace("R$", "").replace(".", "").replace(",", ".").strip()
            try:
                amount = float(amt_clean)
            except ValueError:
                continue

            # Skip payments/credits (negative values)
            if amount <= 0:
                continue

            # Skip payment entries regardless of sign
            if "pagamentos validos" in desc.lower():
                continue

            # Format installment tag: "7 de 9" -> " (07/09)"
            installment_tag = ""
            m = re.match(r"(\d+)\s*de\s*(\d+)", parcela)
            if m:
                installment_tag = f" ({int(m.group(1)):02d}/{int(m.group(2)):02d})"

            rows.append({
                "data": datetime.strptime(date_str, "%d/%m/%Y").strftime("%m/%d/%Y"),
                "descricao_original": desc + installment_tag,
                "valor": f"{amount:.2f}",
                "despesa": desc,
                "banco": "XP",
                "cartao": "0000",
            })

    return rows


# ---------------------------------------------------------------------------
# Itaú Checking Account (PDF)
# ---------------------------------------------------------------------------

def parse_itau_extrato(filepath: str) -> list[dict]:
    """Parse Itaú extrato de conta corrente PDF."""
    rows = []

    # Amount is always last, in Brazilian format: -507,02 or -3.389,06
    tx_re = re.compile(
        r"(\d{2}/\d{2}/\d{4})\s+"       # date
        r"(.+?)\s+"                       # description (non-greedy)
        r"(-?[\d.]+,\d{2})\s*$"          # amount in Brazilian format
    )

    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.split("\n"):
                line = line.strip()

                # Skip non-transaction lines
                if "SALDO DO DIA" in line:
                    continue
                if "REND PAGO" in line:
                    continue
                if not line or line.startswith("*"):
                    continue

                parts = tx_re.match(line)
                if not parts:
                    continue

                date_str = parts.group(1)
                desc = parts.group(2).strip()
                amount_str = parts.group(3).strip()

                # Parse amount (Brazilian format)
                amt_clean = amount_str.replace(".", "").replace(",", ".")
                try:
                    amount = float(amt_clean)
                except ValueError:
                    continue

                # Only expenses (negative)
                if amount >= 0:
                    continue

                amount = abs(amount)

                # Skip internal transfers, investments, card payments, interest
                skip_descs = [
                    "RESGATE CDB", "JUROS LIMITE", "IOF", "SEGURO CARTAO",
                    "PERS BLACK", "CARTAO PERSONNALITE",
                ]
                if any(s in desc.upper() for s in skip_descs):
                    continue

                rows.append({
                    "data": datetime.strptime(date_str, "%d/%m/%Y").strftime("%m/%d/%Y"),
                    "descricao_original": desc,
                    "valor": f"{amount:.2f}",
                    "despesa": desc,
                    "banco": "Itau",
                    "cartao": "0000",
                })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

IMPORT_FIELDS = [
    "transaction_date",
    "amount",
    "description",
    "category",
    "expense_scope",
    "tag",
    "bank_description",
    "obs",
    "schedule_mode",
    "schedule_count",
    "occurrence_index",
    "occurrence_label",
]


def _to_import_row(raw: dict, mappings: list[dict]) -> dict:
    """Convert a parser dict into the full import-format row with inference."""
    tx_iso = datetime.strptime(raw["data"], "%m/%d/%Y").strftime("%Y-%m-%d")
    amount = float(raw["valor"])
    bank_description = raw["descricao_original"]
    raw_despesa = raw["despesa"]

    matched_name, matched_category, matched_tag = match_expense(raw_despesa, mappings)
    description = matched_name or heuristic_description(raw_despesa)

    return {
        "transaction_date": tx_iso,
        "amount": f"{amount:.2f}",
        "description": description,
        "category": matched_category,
        "expense_scope": "individual",
        "tag": matched_tag,
        "bank_description": bank_description,
        "obs": "",
        "schedule_mode": "none",
        "schedule_count": "1",
        "occurrence_index": "1",
        "occurrence_label": "Lançamento único",
    }


def parse_all(extratos_dir: str, password: str, output_csv: str) -> None:
    """Parse all bank statements and write a single CSV in the full import format."""
    all_rows = []

    for fname in sorted(os.listdir(extratos_dir)):
        filepath = os.path.join(extratos_dir, fname)
        print(f"Parsing: {fname}")

        try:
            if fname.endswith(".xlsx"):
                all_rows.extend(parse_btg_credit_card(filepath, password))
            elif fname.endswith(".xls"):
                all_rows.extend(parse_btg_extrato(filepath))
            elif fname.startswith("Fatura_Itau") and fname.endswith(".pdf"):
                all_rows.extend(parse_itau_fatura(filepath))
            elif fname.startswith("itau_extrato") and fname.endswith(".pdf"):
                all_rows.extend(parse_itau_extrato(filepath))
            elif fname.startswith("Fatura") and fname.endswith(".csv"):
                all_rows.extend(parse_xp_fatura(filepath))
            elif fname == os.path.basename(output_csv):
                # Skip our own output file
                continue
            else:
                print(f"  Skipping unknown format: {fname}")
        except Exception as e:
            print(f"  ERROR parsing {fname}: {e}")

    # Deduplicate by (data, despesa, valor, banco) — ignores cartao and installment
    # differences so the same purchase appearing across consecutive faturas
    # (different installment numbers, sometimes different card sections) collapses
    # into a single entry. Keep first occurrence.
    seen = set()
    unique_rows = []
    for r in all_rows:
        key = (r["data"], r["despesa"], r["valor"], r["banco"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    # Sort by date
    unique_rows.sort(key=lambda r: datetime.strptime(r["data"], "%m/%d/%Y"))

    # Load mappings from DB for inference
    mappings = load_mappings_from_supabase()
    print(f"  Mappings carregados: {len(mappings)}")

    import_rows = [_to_import_row(r, mappings) for r in unique_rows]
    matched = sum(1 for r in import_rows if r["category"])
    print(f"  Inferência: {matched}/{len(import_rows)} com despesa reconhecida.")

    # Write CSV in the full import format expected by apps/my_life/app.py::parse_csv_import
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=IMPORT_FIELDS)
        writer.writeheader()
        writer.writerows(import_rows)

    print(f"\nTotal: {len(import_rows)} transactions written to {output_csv}")


if __name__ == "__main__":
    extratos_dir = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/diego/Downloads/extratos"
    password = sys.argv[2] if len(sys.argv) > 2 else "11599639750"
    output_csv = sys.argv[3] if len(sys.argv) > 3 else "C:/Users/diego/Downloads/extratos/despesas_import.csv"

    parse_all(extratos_dir, password, output_csv)
