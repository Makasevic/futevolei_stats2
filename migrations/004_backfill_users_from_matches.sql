-- =============================================================================
-- Migração 004 — Backfill de users e group_members a partir das matches
--
-- Cria um registro em `users` para cada nome que aparece em
-- matches.{winner1,winner2,loser1,loser2} e ainda nao existe em `users`.
-- Em seguida cria o vinculo em `group_members` ligando esse user ao grupo
-- onde ele tem partidas (default: redinha — ajuste o slug se for outro).
--
-- E IDEMPOTENTE: pode rodar mais de uma vez sem efeito colateral.
--
-- ATENCAO sobre nomes inconsistentes:
--   Se houver duas grafias do mesmo jogador (ex: "Jorge Tadeu" vs
--   "Jorge Thadeu"), serao criados DOIS registros em users — um para cada.
--   Use a tela de admin (Renomear / Fundir jogadores) depois deste backfill
--   para consolidar.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- (Opcional) Diagnostico: quantos nomes em matches ainda nao tem registro em users?
-- Rode antes para ver o impacto previsto.
-- -----------------------------------------------------------------------------
-- SELECT DISTINCT TRIM(name) AS missing_name
-- FROM (
--     SELECT winner1 AS name FROM matches
--     UNION ALL SELECT winner2 FROM matches
--     UNION ALL SELECT loser1  FROM matches
--     UNION ALL SELECT loser2  FROM matches
-- ) m
-- WHERE m.name IS NOT NULL
--   AND TRIM(m.name) <> ''
--   AND TRIM(m.name) NOT LIKE '%Outro%'
--   AND NOT EXISTS (SELECT 1 FROM users u WHERE u.name = TRIM(m.name))
-- ORDER BY missing_name;


-- -----------------------------------------------------------------------------
-- 1. Cria registros em `users` para cada nome novo das matches.
--    `email` e UNIQUE NOT NULL — geramos um placeholder unico por user.
-- -----------------------------------------------------------------------------
INSERT INTO users (name, email)
SELECT DISTINCT
    TRIM(m.name),
    gen_random_uuid()::text || '@placeholder.local'
FROM (
    SELECT winner1 AS name FROM matches
    UNION ALL SELECT winner2 FROM matches
    UNION ALL SELECT loser1  FROM matches
    UNION ALL SELECT loser2  FROM matches
) m
WHERE m.name IS NOT NULL
  AND TRIM(m.name) <> ''
  AND TRIM(m.name) NOT LIKE '%Outro%'
  AND NOT EXISTS (
      SELECT 1 FROM users u WHERE u.name = TRIM(m.name)
  );


-- -----------------------------------------------------------------------------
-- 2. Cria vinculos em `group_members` ligando cada user ao grupo onde ele
--    tem partidas. Vincula:
--      - matches com group_id explicito → ao proprio grupo
--      - matches com group_id NULL (legado) → ao grupo 'redinha'
--    Ajuste o slug abaixo se quiser outro destino para os legados.
-- -----------------------------------------------------------------------------
WITH user_groups AS (
    SELECT DISTINCT
        u.id AS user_id,
        COALESCE(m.group_id, redinha.id) AS group_id
    FROM users u
    JOIN (
        SELECT winner1 AS name, group_id FROM matches
        UNION ALL SELECT winner2, group_id FROM matches
        UNION ALL SELECT loser1,  group_id FROM matches
        UNION ALL SELECT loser2,  group_id FROM matches
    ) m ON TRIM(m.name) = u.name
    CROSS JOIN (SELECT id FROM groups WHERE slug = 'redinha') redinha
    WHERE m.name IS NOT NULL
      AND TRIM(m.name) <> ''
      AND TRIM(m.name) NOT LIKE '%Outro%'
)
INSERT INTO group_members (group_id, user_id, role)
SELECT ug.group_id, ug.user_id, 'player'
FROM user_groups ug
WHERE NOT EXISTS (
    SELECT 1 FROM group_members gm
    WHERE gm.group_id = ug.group_id
      AND gm.user_id = ug.user_id
);


-- -----------------------------------------------------------------------------
-- (Opcional) Verificacao pos-execucao: quantos users e memberships existem?
-- -----------------------------------------------------------------------------
-- SELECT COUNT(*) AS total_users FROM users;
-- SELECT g.slug, COUNT(gm.id) AS membros
-- FROM groups g LEFT JOIN group_members gm ON gm.group_id = g.id
-- GROUP BY g.slug;
