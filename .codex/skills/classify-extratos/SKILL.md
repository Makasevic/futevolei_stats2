---
name: classify-extratos
description: Use when a parsed expenses CSV needs conservative classification from Supabase history. Reads `despesas_import.csv`, reuses previously confirmed `description`, `category`, `tag`, and `expense_scope`, and leaves unmatched rows unchanged.
---

# Classify Extratos

Use this skill after `parse_extratos.py` has already generated a structured CSV.

## Default workflow

1. Run `classify_extratos_csv.py`.
2. Use `C:/Users/diego/Downloads/extratos/despesas_import.csv` as input unless the user gives another path.
3. Write the result to `C:/Users/diego/Downloads/extratos/despesas_import_classificado.csv` unless the user asks for a different output file.

## Behavior

- Query the Supabase `expenses` table to infer the expense.
- Query `expense_mappings` to apply defaults when available.
- Normalize `bank_description` and `description`.
- Reuse historical classifications only when the match is strong.
- Fill:
  - `description`
  - `category`
  - `tag`
  - `expense_scope`
- Leave ambiguous rows untouched.

## Command

```powershell
C:/Users/diego/miniconda3/envs/basico/python.exe classify_extratos_csv.py
```

## Report back

Always report:
- output CSV path
- rows processed
- rows classified
- rows left unchanged
- exact vs fuzzy match counts
