from flask import Blueprint, request, jsonify, Response
from supabase import create_client
from uuid import uuid4

# Importa secrets do seu sistema
from src.redinha_stats.config.settings import SUPABASE_SERVICE_KEY, SUPABASE_URL
from src.redinha_stats.infrastructure.supabase.groups_repository import fetch_group_by_slug

# Grupo fixo que lança partidas via GPT
_GPT_GROUP_SLUG = "redinha"

bp = Blueprint("batch_endpoints", __name__)

# Cliente Supabase usando SERVICE KEY (permite inserir no banco)
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ------------------------------------------------------------
#  POST /api/prepare_matches
#  O GPT envia: { "date": "AAAA-MM-DD", "matches": [...] }
#  O backend salva o payload no Supabase e devolve um token.
# ------------------------------------------------------------
@bp.post("/api/prepare_matches")
def prepare_matches():
    payload = request.get_json()

    if not payload:
        return jsonify({"error": "Payload vazio"}), 400

    # Gera token curto
    token = uuid4().hex[:8]

    # Salva no Supabase
    supabase.table("pending_batches").insert({
        "token": token,
        "payload": payload,
        "status": "pending"
    }).execute()

    # Retorna o token pro GPT
    return jsonify({"token": token})


# ------------------------------------------------------------
#  GET /confirm/<token>
#  Quando o usuário clica no link, esse endpoint:
#   1. Recupera o payload do Supabase
#   2. Insere todas as partidas em "matches"
#   3. Marca o token como "used"
#   4. Retorna página de sucesso
# ------------------------------------------------------------
@bp.get("/confirm/<token>")
def confirm_batch(token):

    # Busca o token no Supabase
    result = supabase.table("pending_batches").select("*").eq("token", token).execute()

    if not result.data:
        return Response("<h2>❌ Token inválido</h2>", mimetype="text/html")

    row = result.data[0]

    # Já foi utilizado
    if row["status"] != "pending":
        return Response("<h2>❌ Token já usado</h2>", mimetype="text/html")

    payload = row["payload"]
    match_date = payload["date"]
    matches = payload["matches"]

    # Resolve o group_id do grupo fixo que usa o GPT
    group = fetch_group_by_slug(_GPT_GROUP_SLUG)
    if not group:
        return Response(f"<h2>❌ Grupo '{_GPT_GROUP_SLUG}' não encontrado</h2>", mimetype="text/html")
    group_id = group["id"]

    # Prepara as linhas para inserir
    entries = []
    for m in matches:
        entries.append({
            "date": match_date,
            "winner1": m["winner1"],
            "winner2": m["winner2"],
            "loser1": m["loser1"],
            "loser2": m["loser2"],
            "group_id": group_id,
        })

    try:
        # Insere de uma vez no Supabase
        supabase.table("matches").insert(entries).execute()

        # Marca token como utilizado
        supabase.table("pending_batches").update({"status": "used"}).eq("token", token).execute()

        html = f"""
            <h2>✅ {len(entries)} partidas lançadas com sucesso!</h2>
            <p>Data: {match_date}</p>
            <p>Pronto — pode fechar esta aba.</p>
        """
        return Response(html, mimetype="text/html")

    except Exception as e:
        return Response(f"<h2>❌ Erro ao lançar partidas</h2><p>{e}</p>", mimetype="text/html")
