from __future__ import annotations

from collections import Counter
from typing import Dict, List

import pandas as pd

from src.redinha_stats.config.app_settings import get_config
from src.redinha_stats.domain.matches.processing import (
    preparar_dados_duplas,
    preparar_dados_individuais,
)
from src.redinha_stats.web.data_helpers import excluded_players


def build_infos_summary(df: pd.DataFrame) -> Dict[str, object]:
    config = get_config()
    excluidos = excluded_players()

    df_jog = preparar_dados_individuais(df)
    df_jog = df_jog[~df_jog["jogadores"].str.contains("Outro", na=False)].copy()
    df_jog["vitórias"] = df_jog["vitórias"].astype(int)
    df_jog["derrotas"] = df_jog["derrotas"].astype(int)
    df_jog["jogos"] = df_jog["vitórias"] + df_jog["derrotas"]

    df_duplas = preparar_dados_duplas(df)
    df_duplas = df_duplas[~df_duplas["duplas"].str.contains("Outro", na=False)].copy()

    media_top_10 = df_jog["jogos"].nlargest(10).mean()
    if media_top_10 > 0:
        limiar = media_top_10 * config.min_participation_ratio
        df_jog = df_jog[df_jog["jogos"] >= limiar]

    df_jog["saldo"] = df_jog["vitórias"] - df_jog["derrotas"]
    df_jog = df_jog.set_index("jogadores")
    jogadores_validos = set(df_jog.index)
    dias_jogados = pd.to_datetime(df.index, errors="coerce").normalize().nunique()

    def melhor_aproveitamento(label: str, pior: bool = False) -> Dict[str, str]:
        stats = df_jog.copy()
        stats["jogos"] = stats["vitórias"] + stats["derrotas"]
        stats = stats.drop(index=list(excluidos), errors="ignore")
        if stats.empty:
            return {"title": label, "value": "-", "detail": "-"}

        n = len(stats)
        total_jogos = stats["jogos"].sum()
        media_dos_demais = (total_jogos - stats["jogos"]) / (n - 1) if n > 1 else 0
        limiar = 0.20 * media_dos_demais
        candidatos = stats[stats["jogos"] >= limiar].copy()
        if candidatos.empty:
            return {"title": label, "value": "-", "detail": "-"}

        candidatos["aprov"] = candidatos["vitórias"] / candidatos["jogos"]
        if pior:
            cand_ord = candidatos.sort_values(["aprov", "jogos"], ascending=[True, False])
        else:
            cand_ord = candidatos.sort_values(
                ["aprov", "jogos", "vitórias"], ascending=[False, False, False]
            )

        nome = cand_ord.index[0]
        row = cand_ord.iloc[0]
        return {
            "title": label,
            "value": nome,
            "detail": f"{row['aprov']:.0%} de aproveitamento",
        }

    def mais_fominha() -> Dict[str, str]:
        if df_jog.empty:
            return {"title": "O mais fominha", "value": "-", "detail": "-"}
        jogostotal = df_jog["vitórias"] + df_jog["derrotas"]
        jogador = jogostotal.idxmax()
        return {
            "title": "O mais fominha",
            "value": jogador,
            "detail": f"Jogos: {jogostotal.max():.0f}",
        }

    def maior_vexame() -> Dict[str, str]:
        registros: List[Dict[str, object]] = []
        df_dias = df.copy()
        df_dias.index = pd.to_datetime(df_dias.index, errors="coerce")
        df_dias = df_dias[pd.notna(df_dias.index)]

        for dia, df_dia in df_dias.groupby(df_dias.index.normalize()):
            vitorias_dia = pd.Series(df_dia.iloc[:, 0:2].values.ravel()).value_counts()
            derrotas_dia = pd.Series(df_dia.iloc[:, 2:4].values.ravel()).value_counts()
            vitorias_dia = vitorias_dia.drop(list(excluidos), errors="ignore")
            derrotas_dia = derrotas_dia.drop(list(excluidos), errors="ignore")

            jogadores_dia = sorted(set(vitorias_dia.index) | set(derrotas_dia.index))
            jogadores_dia = [j for j in jogadores_dia if j in jogadores_validos]
            if not jogadores_dia:
                continue

            vitorias_dia = vitorias_dia.reindex(jogadores_dia, fill_value=0)
            derrotas_dia = derrotas_dia.reindex(jogadores_dia, fill_value=0)
            saldo_dia = derrotas_dia - vitorias_dia
            if saldo_dia.empty:
                continue

            pior_jogador = saldo_dia.idxmax()
            registros.append(
                {
                    "dia": dia,
                    "jogador": pior_jogador,
                    "saldo": int(saldo_dia.loc[pior_jogador]),
                    "v": int(vitorias_dia.loc[pior_jogador]),
                    "d": int(derrotas_dia.loc[pior_jogador]),
                }
            )

        if not registros:
            return {"title": "O maior vexame na historia", "value": "-", "detail": "-"}

        pior = max(registros, key=lambda r: r["saldo"])
        data_fmt = pd.to_datetime(pior["dia"]).strftime("%d/%m/%Y")
        return {
            "title": "O maior vexame na historia",
            "value": pior["jogador"],
            "detail": f"{pior['v']}-{pior['d']}  ({data_fmt})",
        }

    def mais_paneleiro() -> Dict[str, str]:
        excluir = {"Outro_1", "Outro_2"}
        partner_counts: Dict[str, Counter] = {}

        for _, row in df.iterrows():
            try:
                w1, w2, l1, l2 = row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3]
            except Exception:
                continue

            for a, b in [(w1, w2), (l1, l2)]:
                if pd.isna(a) or pd.isna(b):
                    continue
                if a in excluir or b in excluir:
                    continue
                if a not in jogadores_validos or b not in jogadores_validos:
                    continue
                partner_counts.setdefault(a, Counter())
                partner_counts.setdefault(b, Counter())
                partner_counts[a][b] += 1
                partner_counts[b][a] += 1

        if not partner_counts:
            return {"title": "O mais paneleiro", "value": "-", "detail": "-"}

        registros = []
        for jog, cnts in partner_counts.items():
            total = sum(cnts.values())
            if total <= 0:
                continue
            parceiro, juntos = max(cnts.items(), key=lambda kv: (kv[1], kv[0]))
            registros.append(
                {
                    "jogador": jog,
                    "parceiro": parceiro,
                    "juntos": int(juntos),
                    "jogos": int(total),
                    "share": float(juntos / total),
                }
            )

        if not registros:
            return {"title": "O mais paneleiro", "value": "-", "detail": "-"}

        stats = pd.DataFrame(registros).set_index("jogador")
        n = len(stats)
        media_dos_demais = (
            (stats["jogos"].sum() - stats["jogos"]) / (n - 1)
            if n > 1
            else pd.Series(0, index=stats.index)
        )
        limiar = 0.20 * media_dos_demais
        cand = stats[stats["jogos"] >= limiar].copy()
        if cand.empty:
            return {"title": "O mais paneleiro", "value": "-", "detail": "-"}

        cand = cand.sort_values(
            ["share", "jogos", "juntos", "parceiro"],
            ascending=[False, False, False, True],
        )
        top = cand.iloc[0]
        return {
            "title": "O mais paneleiro",
            "value": top.name,
            "detail": f"com {top['parceiro']}: {top['share']:.0%} ({int(top['juntos'])}/{int(top['jogos'])} jogos)",
        }

    def dupla_entrosada() -> Dict[str, str]:
        duplas_validas = df_duplas[df_duplas["jogos"] >= config.min_duo_matches]
        if duplas_validas.empty:
            return {
                "title": f"Dupla mais entrosada (min de {config.min_duo_matches} jogos)",
                "value": "-",
                "detail": "-",
            }

        melhor_dupla = duplas_validas.iloc[0]

        def formatar_nome_iniciais(nome_completo: str) -> str:
            partes = nome_completo.strip().split()
            if not partes:
                return nome_completo
            return f"{partes[0][0]}. {partes[-1]}"

        nomes_colapsados = [
            formatar_nome_iniciais(nome)
            for nome in str(melhor_dupla["duplas"]).split(" e ")
        ]
        nomes_ordenados = sorted(nomes_colapsados, key=lambda nome: (len(nome), nome))
        dupla_formatada = " e ".join(nomes_ordenados)

        return {
            "title": f"Dupla mais entrosada (min de {config.min_duo_matches} jogos)",
            "value": dupla_formatada,
            "detail": f"{melhor_dupla['aproveitamento']:.0f}% de aproveitamento",
        }

    return {
        "resumo": {
            "total_partidas": len(df),
            "dias_jogados": dias_jogados,
            "total_minutos": len(df) * 20,
        },
        "destaques_primarios": [
            melhor_aproveitamento("O mais brabo"),
            mais_fominha(),
            dupla_entrosada(),
        ],
        "destaques_secundarios": [
            melhor_aproveitamento("Ninguem quer jogar com", pior=True),
            maior_vexame(),
            mais_paneleiro(),
        ],
    }
