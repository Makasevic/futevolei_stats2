# Futevôlei Stats

Aplicação Flask que expõe rankings e resumos de partidas de futevôlei. O fluxo principal está em
`main_api.py`, que consome o Supabase para carregar e atualizar partidas.

## Configuração

Defina as credenciais exigidas pelo Supabase via variáveis de ambiente antes de iniciar o servidor:

```bash
export SUPABASE_URL="https://<sua-instancia>.supabase.co"
export SUPABASE_ANON_KEY="chave-anon-publica"
# Senhas opcionais para proteger rotas administrativas
export ADMIN_PASSWORD="senha-admin"
export MATCH_ENTRY_PASSWORD="senha-lancamento"
```

Instale as dependências e suba o servidor localmente:

```bash
pip install -r requirements.txt
python main_api.py  # inicia o Flask em 0.0.0.0:8000
```

O arquivo `main_api.py` define as rotas de listagem, ranking e administração, além de renderizar os
templates HTML presentes em `templates/` usando os dados processados em `processing.py` e
`detalhamento.py`.
