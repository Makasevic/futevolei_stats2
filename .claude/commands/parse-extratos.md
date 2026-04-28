Parse bank statements (extratos/faturas) from C:\Users\diego\Downloads\extratos and generate a CSV file for import into the Expenses app.

## What to do

1. Run the parser script: `python parse_extratos.py`
   - Source directory: `C:/Users/diego/Downloads/extratos`
   - Password for protected files: `11599639750`
   - Output: `C:/Users/diego/Downloads/extratos/despesas_import.csv`

2. Show a summary of results:
   - Total number of transactions
   - Breakdown by bank and card
   - Date range covered
   - Total amount per source

3. If there are new file formats or parsing errors, update `parse_extratos.py` accordingly.

## Supported file formats

- **BTG Credit Card** (`.xlsx`, password-protected): Decrypts and parses fatura transactions
- **BTG Checking Account** (`.xls`): Parses PIX, boletos, and other checking account transactions
- **Itau Credit Card** (`.pdf`, `Fatura_Itau_*.pdf`): Parses two-column PDF layout using word-position extraction
- **Itau Checking Account** (`.pdf`, `itau_extrato_*.pdf`): Parses extrato with auto-debit, PIX, boleto entries
- **XP Credit Card** (`.csv`, `Fatura*.csv`): Semicolon-delimited CSV with columns `Data;Estabelecimento;Portador;Valor;Parcela`. Skips negative values (payments) and `Pagamentos Validos` rows.

## Deduplication

Rows are deduplicated by `(data, despesa, valor, banco)` — intentionally ignoring `cartao` and installment number. Rationale: the same original purchase often appears across consecutive monthly faturas with different installment numbers (e.g. `(09/10)` and `(10/10)`) and, for Itaú faturas containing multiple cards, may be attributed to different cartao last-4s by page-based detection. Collapsing on this key yields one row per unique purchase.

## CSV output format

The CSV follows the import format expected by `apps/my_life/app.py::parse_csv_import`:

| Coluna | Descrição |
|---|---|
| `transaction_date` | Data ISO `YYYY-MM-DD` |
| `amount` | Valor positivo decimal, ex.: `123.45` |
| `description` | Nome da despesa (nunca vazio — matched do banco ou derivado da descrição bruta) |
| `category` | Categoria do `expense_mappings` (vazio quando não houver match no banco) |
| `expense_scope` | `individual` por padrão |
| `tag` | Tag do `expense_mappings` (vazio quando não houver) |
| `bank_description` | Descrição original do banco (preserva sufixo de parcela, ex.: `(09/12)`) |
| `obs` | Vazio |
| `schedule_mode` | `none` |
| `schedule_count` | `1` |
| `occurrence_index` | `1` |
| `occurrence_label` | `Lançamento único` |

> `import_fingerprint` não é incluído no CSV. O backend calcula automaticamente (ver `compute_import_fingerprint` em `app.py`), e o Supabase não gera esse campo — ele existe só pra dedup entre re-imports.

## Inferência de despesa / categoria / tag

Cascata aplicada para cada transação:

1. **Fuzzy match no Supabase** (`expense_mappings` onde `is_active=true`). Compara a descrição bruta (normalizada: minúsculas, sem acentos, sem pontuação) contra cada `expense_name`:
   - substring bidirecional (mapping dentro da descrição ou vice-versa),
   - token overlap (qualquer token do mapping com ≥ 4 caracteres aparece na descrição),
   - similaridade via `difflib.SequenceMatcher` com corte em 0.72.

   Isso pega variações tipo `NETFLIX.COM BRAZIL` → `Netflix`, `IFD*MERCADO BOM DIA` → `Mercado`, etc.

2. **Fallback heurístico local** (sem banco): se nenhum mapping bateu, gera um nome limpo da descrição bruta — remove sufixo de parcela `(NN/NN)` e prefixos de adquirente (`MP*`, `EC*`, `IFD*`, `PAG*`, `PIX`, etc.), pega os 3 primeiros tokens alfabéticos com 3+ letras e aplica Title Case (limite de 40 caracteres). Com isso `description` nunca sai vazio.

3. Se o Supabase não estiver acessível, o script continua com apenas o passo 2.

## After running

Tell the user:
- The CSV file location
- How many transactions were parsed
- Remind them to open the Expenses app and use the Import menu to upload the CSV
- Note that parceled transactions include installment info like "(03/10)" in the description
