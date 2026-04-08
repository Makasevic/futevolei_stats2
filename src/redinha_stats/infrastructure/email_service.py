"""Serviço de envio de email via Resend."""

from __future__ import annotations

from src.redinha_stats.config.settings import INVITE_FROM_EMAIL, RESEND_API_KEY


def send_invite_email(
    to_email: str,
    group_name: str,
    invite_url: str,
) -> None:
    """Envia email de convite para o jogador.

    Raises RuntimeError se RESEND_API_KEY não estiver configurada ou o envio falhar.
    """

    if not RESEND_API_KEY:
        raise RuntimeError("RESEND_API_KEY não configurada. Defina no secrets.toml ou variável de ambiente.")

    try:
        import resend  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("Pacote 'resend' não instalado. Execute: pip install resend") from exc

    resend.api_key = RESEND_API_KEY

    html_body = f"""
    <p>Olá!</p>
    <p>Você foi convidado para participar do grupo <strong>{group_name}</strong> no Redinha Stats.</p>
    <p>Clique no link abaixo para criar sua conta:</p>
    <p><a href="{invite_url}">{invite_url}</a></p>
    <p>O link expira em 7 dias.</p>
    """

    response = resend.Emails.send({
        "from": INVITE_FROM_EMAIL,
        "to": to_email,
        "subject": f"Convite para {group_name} — Redinha Stats",
        "html": html_body,
    })

    if isinstance(response, dict) and response.get("statusCode") and response["statusCode"] >= 400:
        raise RuntimeError(f"Erro ao enviar email: {response}")
