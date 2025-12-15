from flask import Blueprint, request, jsonify, Response
from supabase import create_client
from uuid import uuid4

# Pega secrets do seu sistema
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY

bp = Blueprint("batch_endpoints", __name__)

# Criar cliente Supabase usando a SERVICE KEY (necessária para inserir no banco)
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# -----------------------------------
#  POST /api/prepare_matches
# -----------------------------------
@bp.post("/api/prepare_matches")
def prepare_matches():
    payload = request.get_json()

    if not payload:
        return jsonify({"error": "Payload vazio"}), 400

    token = uuid4().hex[:8]

    supabase.table("pending_batches").insert({
        "token": token,
        "payload": payload,
        "status": "pending"
    }).execute()

    return jsonify({"token": token})


# -----------------------------------
#  GET /confirm/<token>
# -----------------------------------
@bp.get("/confirm/<token>")
def confirm_batch(token):

    result = supabase.table("pending_batches").select("*").eq("token", token).execute()

    if not result.data:
        return Response("<h2>❌ Token inválido</h2>", mimetype="text/html")

    row = result.data[0]

    if row["status"] != "pending":
        return Response("<h2>❌ Token já usado</h2>", mimetype="text/html")

    payload = row["payload"]
    match_date = payload["date"]
    matches = payload["matches"]

    entries = []
    for m in matches:
        entries.append({
            "date": match_date,
            "winner1": m["winner1"],
            "winner2": m["winner2"],
            "loser1": m["loser1"],
            "loser2": m["loser2"],
        })

    try:
        supabase.table("matches").insert(entries).execute()

        supabase.table("pending_batches").update({"status": "used"}).eq("token", token).execute()

        html = f"""
            <h2>✅ {len(entries)} partidas lançadas com sucesso!</h2>
            <p>Data: {match_date}</p>
            <p>Pronto, pode fechar esta aba.</p>
        """
        return Response(html, mimetype="text/html")

    except Exception as e:
        return Response(f"<h2>❌ Erro ao lançar partidas</h2><p>{e}</p>", mimetype="text/html")
