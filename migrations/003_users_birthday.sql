-- =============================================================================
-- Migração 003 — Adicionar data de aniversário na tabela users
-- =============================================================================

ALTER TABLE users ADD COLUMN IF NOT EXISTS birthday DATE;
