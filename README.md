# Futevolei Stats

Aplicacao Flask que expoe rankings e resumos de partidas de futevolei. A
entrada local e `app.py`, o entrypoint de deploy e `wsgi.py`, e a criacao da
app passa por `src/redinha_stats/web/app_factory.py`.

## Configuracao

Defina as credenciais exigidas pelo Supabase via variaveis de ambiente antes de
iniciar o servidor:

```bash
export SUPABASE_URL="https://<sua-instancia>.supabase.co"
export SUPABASE_ANON_KEY="chave-anon-publica"
# Senhas opcionais para proteger rotas administrativas
export ADMIN_PASSWORD="senha-admin"
export MATCH_ENTRY_PASSWORD="senha-lancamento"
```

Instale as dependencias e suba o servidor localmente:

```bash
pip install -r requirements.txt
python app.py
```

O deploy usa `gunicorn wsgi:app`. O modulo `main_api.py` ainda existe como
compatibilidade temporaria durante a refatoracao, mas a estrutura nova ja esta
concentrada em `src/redinha_stats/`.
