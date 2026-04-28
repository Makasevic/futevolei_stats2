-- =============================================================================
-- Migração 001 — Fundação multi-tenant
-- Aplicar no SQL Editor do Supabase (ou via psql).
-- Seguro para rodar em base existente: apenas ADD COLUMN e CREATE TABLE.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Grupos
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS groups (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug       TEXT UNIQUE NOT NULL,
    name       TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- 2. Usuários (identidade própria — login/convite são funcionalidades futuras)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email          TEXT UNIQUE NOT NULL,
    name           TEXT NOT NULL,
    password_hash  TEXT,            -- null até o usuário fazer cadastro
    email_verified BOOLEAN DEFAULT false,
    created_at     TIMESTAMPTZ DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- 3. Vínculo usuário ↔ grupo ("matrícula")
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS group_members (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id   UUID NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    user_id    UUID NOT NULL REFERENCES users(id)  ON DELETE CASCADE,
    role       TEXT DEFAULT 'player',  -- 'player' | 'admin' | 'limited'
    joined_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE (group_id, user_id)
);

-- -----------------------------------------------------------------------------
-- 4. Campeonatos configuráveis (substitui arquivos Python hardcoded)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS championships (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id     UUID NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    slug         TEXT NOT NULL,
    title        TEXT NOT NULL,
    description  TEXT,
    format       TEXT DEFAULT 'groups_knockout',
    config       JSONB DEFAULT '{}',   -- times, grupos, chaves — estrutura livre
    edit_password TEXT,
    created_at   TIMESTAMPTZ DEFAULT now(),
    UNIQUE (group_id, slug)
);

-- -----------------------------------------------------------------------------
-- 5. Adicionar group_id nas tabelas existentes
--    (valores NULL = dados legados da Redinha, serão preenchidos na Fase 2)
-- -----------------------------------------------------------------------------
ALTER TABLE matches
    ADD COLUMN IF NOT EXISTS group_id UUID REFERENCES groups(id);

ALTER TABLE championship_scores
    ADD COLUMN IF NOT EXISTS group_id UUID REFERENCES groups(id);

-- -----------------------------------------------------------------------------
-- 6. Índices para performance nas queries filtradas por grupo
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_matches_group_id
    ON matches (group_id);

CREATE INDEX IF NOT EXISTS idx_championship_scores_group_id
    ON championship_scores (group_id);

CREATE INDEX IF NOT EXISTS idx_group_members_group_id
    ON group_members (group_id);

CREATE INDEX IF NOT EXISTS idx_championships_group_id
    ON championships (group_id);
