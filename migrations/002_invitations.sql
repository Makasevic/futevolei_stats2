-- =============================================================================
-- Migração 002 — Tabela de convites
-- =============================================================================

CREATE TABLE IF NOT EXISTS invitations (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token      TEXT UNIQUE NOT NULL DEFAULT gen_random_uuid()::text,
    group_id   UUID NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    user_id    UUID REFERENCES users(id) ON DELETE SET NULL,  -- preenchido quando vinculado a jogador existente
    email      TEXT,                                           -- opcional (para envio por email)
    name       TEXT,                                           -- nome pré-preenchido (jogador existente)
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (now() + INTERVAL '7 days'),
    used_at    TIMESTAMPTZ,                                    -- preenchido quando o convite é resgatado
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_invitations_token    ON invitations (token);
CREATE INDEX IF NOT EXISTS idx_invitations_group_id ON invitations (group_id);
