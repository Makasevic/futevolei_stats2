Classify the parsed CSV at `C:\Users\diego\Downloads\extratos\despesas_import.csv` using expense history already stored in Supabase, and generate a classified CSV for import into the Expenses app.

## What to do

1. Run the classifier script:
   - Command: `C:/Users/diego/miniconda3/envs/basico/python.exe classify_extratos_csv.py`
   - Input: `C:/Users/diego/Downloads/extratos/despesas_import.csv`
   - Output: `C:/Users/diego/Downloads/extratos/despesas_import_classificado.csv`

2. Show a short summary:
   - Total rows processed
   - How many rows were classified
   - How many rows were left unchanged
   - Breakdown of exact vs fuzzy matches

3. Be conservative:
   - Use the `expenses` table in Supabase as the source of truth for inferring the expense
   - If available, use `expense_mappings` as the source of defaults for category, tag, and expense scope
   - Only classify rows when there is a strong historical match
   - If a row is ambiguous or weakly matched, leave it untouched

## Matching behavior

- The script uses `bank_description` first, then `description`
- It normalizes text before matching:
  - lowercases
  - removes accents
  - removes punctuation noise
  - removes installment suffixes like `(03/10)`
  - removes common prefixes like `MP*`, `IFD*`, `PIX`
- It first infers the canonical `description` from historical rows in `expenses`
- Then, when present in `expense_mappings`, it applies the defaults configured for that expense:
  - `category`
  - `tag`
  - `expense_scope`
- If no mapping default exists, it falls back to the dominant tuple observed in `expenses`

## Output behavior

- The original parser CSV is preserved
- The classifier writes a new file:
  - `C:/Users/diego/Downloads/extratos/despesas_import_classificado.csv`
- Rows without a strong match stay exactly as they were

## After running

Tell the user:
- The output CSV file location
- How many rows were classified
- That unmatched rows were intentionally left untouched
- That they can import the classified CSV in the Expenses app
