#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R6 -- a razao dp entre sorteios / dp entre as medias das celulas depende do nivel do preditor
constante? Criterio: _v3_2026-09-25/criterios/criterio_R6_nivel_do_sentinela.json (fixado em
2026-10-01T22:47:16-03:00, antes de existir este script).

Consome os parciais do laco comum (scripts/v3_13_laco_comum.py -> fase5/_parcial_laco/<celula>.json),
que trazem, por celula e sorteio, o MAE nos nos VALIDOS do teste de cada constante:
c in {-100, -110, -120, -130} dBm e a "mediana dos validos do treino" (recalculada por sorteio).
Referencia suplementar (nao e um dos cinco do criterio): a mediana do treino inteiro, que e a
constante do teste 2.1 (vale -110,0 dBm em todos os sorteios das 16 celulas).

Por preditor: dp (ddof=1) entre sorteios do MAE-validos por celula (sorteios sem no valido no
teste excluidos e contados); media das 16 celulas; dp (ddof=1) entre as 16 medias das celulas
(media entre sorteios incluidos); razao = media dos dp / dp entre medias. Suplementar: mesma
coisa com dp ponderado (peso = n de nos validos do teste, orig._dp_ponderado), como na folha v3-12.
Conferencia obrigatoria: c = -110 dBm reproduz 8,0166 (media dos dp) e 2,1796 (dp entre celulas).

Nao interpreta resultado: so numeros. Sem GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
SCRIPT_SHA256 = hashlib.sha256(HERE.read_bytes()).hexdigest()
LACO = HERE.parent / "v3_13_laco_comum.py"
LACO_SHA256 = hashlib.sha256(LACO.read_bytes()).hexdigest()
BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
V3 = BASE / "_v3_2026-09-25"
OUT = V3 / "fase5"
PARCIAL_DIR = OUT / "_parcial_laco"
VALID = OUT / "R5R6_validacao_laco.json"
CRITERIO = V3 / "criterios" / "criterio_R6_nivel_do_sentinela.json"
F2_JSON = V3 / "fase2" / "2.1_deriva_erro_baselines_16x60rnd.json"
OUT_SORTEIO = OUT / "R6_por_sorteio.json"
OUT_RESUMO = OUT / "R6_resumo.json"
CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QUADRANTES = ["Q1", "Q2", "Q3", "Q4"]
PREDITORES = ("c-100", "c-110", "c-120", "c-130", "mediana_validos_treino", "mediana_treino_inteiro_ref")
CRITERIO_5 = PREDITORES[:5]

_spec = importlib.util.spec_from_file_location("laco_v3_13", LACO)
laco = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(laco)


def agora() -> str:
    return datetime.now(timezone.utc).isoformat()


def dp1(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else None


def razao(a, b):
    return None if (a is None or b in (None, 0)) else float(a / b)


def consolidar():
    val = json.loads(VALID.read_text(encoding="utf-8"))
    if not val.get("reproduz"):
        raise SystemExit("ABORTA: validacao do laco nao reproduz o parcial gravado")
    f2 = json.loads(F2_JSON.read_text(encoding="utf-8"))
    seeds = [int(s) for s in f2["nota_divergencia_seeds"]["seeds_usados_nesta_rodada"]]
    celulas = [f"{c}_{q}" for c in CIDADES for q in QUADRANTES]
    crit = json.loads(CRITERIO.read_text(encoding="utf-8"))

    dados = {}
    sha_parciais = {}
    for ch in celulas:
        arq = PARCIAL_DIR / f"{ch}.json"
        r = json.loads(arq.read_text(encoding="utf-8"))
        if r.get("status") != "ok":
            raise SystemExit(f"parcial {ch} status {r.get('status')}")
        assert [s["split_seed"] for s in r["por_sorteio"]] == seeds
        sha_parciais[ch] = hashlib.sha256(arq.read_bytes()).hexdigest()
        dados[ch] = r

    por_sorteio = {ch: [{"split_seed": s["split_seed"], "n_validos_teste": s["n_pop_teste"]["validos"],
                          "constantes": {k: {"c": s["R6"][k]["c"], "mae_validos": s["R6"][k]["mae_validos"]}
                                         for k in PREDITORES}}
                         for s in dados[ch]["por_sorteio"]] for ch in celulas}

    por_pred = {}
    for k in PREDITORES:
        cel = {}
        for ch in celulas:
            inc = [(s["R6"][k]["mae_validos"], s["n_pop_teste"]["validos"]) for s in dados[ch]["por_sorteio"]
                   if s["R6"][k]["mae_validos"] is not None]
            vals = [x for x, _ in inc]
            pesos = [w for _, w in inc]
            cs = [s["R6"][k]["c"] for s in dados[ch]["por_sorteio"] if s["R6"][k]["mae_validos"] is not None]
            cel[ch] = {"n_sorteios_incluidos": len(vals), "n_sorteios_sem_no_valido_excluidos": len(dados[ch]["por_sorteio"]) - len(vals),
                       "dp_entre_sorteios": dp1(vals), "dp_ponderado_por_n_validos": laco.orig._dp_ponderado(vals, pesos),
                       "media_mae_validos": float(np.mean(vals)) if vals else None,
                       "c_min": float(min(cs)) if cs else None, "c_max": float(max(cs)) if cs else None}
        dps = [cel[c]["dp_entre_sorteios"] for c in celulas]
        dpw = [cel[c]["dp_ponderado_por_n_validos"] for c in celulas]
        meds = [cel[c]["media_mae_validos"] for c in celulas]
        m_dp = float(np.mean(dps))
        dp_cel = dp1(meds)
        m_dpw = float(np.mean(dpw))
        por_pred[k] = {
            "no_criterio": k in CRITERIO_5,
            "media_16_celulas_dp_entre_sorteios": m_dp,
            "dp_entre_medias_das_celulas": dp_cel,
            "razao": razao(m_dp, dp_cel),
            "suplementar_ponderado": {"media_16_celulas_dp_ponderado": m_dpw,
                                      "razao_ponderado_sobre_dp_entre_celulas": razao(m_dpw, dp_cel)},
            "media_das_16_medias_das_celulas": float(np.mean(meds)),
            "dp_entre_sorteios_por_celula_min_max": [float(min(dps)), float(max(dps))],
            "n_sorteios_excluidos_total": int(sum(cel[c]["n_sorteios_sem_no_valido_excluidos"] for c in celulas)),
            "por_celula": cel}

    # ---- conferencia c = -110 contra 8,0166 e 2,1796 (e contra o artefato 2.1)
    p110 = por_pred["c-110"]
    art = f2["resumo"]
    conf = {"alvo_enunciado": {"media_dp_entre_sorteios": 8.0166, "dp_entre_celulas": 2.1796},
            "obtido": {"media_dp_entre_sorteios": p110["media_16_celulas_dp_entre_sorteios"],
                       "dp_entre_celulas": p110["dp_entre_medias_das_celulas"]},
            "arredondado_4_casas": {"media_dp_entre_sorteios": round(p110["media_16_celulas_dp_entre_sorteios"], 4),
                                    "dp_entre_celulas": round(p110["dp_entre_medias_das_celulas"], 4)},
            "artefato_2.1": {"dp_medio_entre_sorteios_validos_dB_simples": art["dp_medio_entre_sorteios_validos_dB_simples"],
                              "dp_entre_celulas_das_medias_validos_dB": art["dp_entre_celulas_das_medias_validos_dB"],
                              "dp_medio_ponderado": art["dp_medio_entre_sorteios_validos_dB_ponderado"]},
            "absdiff_vs_artefato_2.1": {
                "media_dp": abs(p110["media_16_celulas_dp_entre_sorteios"] - art["dp_medio_entre_sorteios_validos_dB_simples"]),
                "dp_celulas": abs(p110["dp_entre_medias_das_celulas"] - art["dp_entre_celulas_das_medias_validos_dB"]),
                "media_dp_ponderado": abs(p110["suplementar_ponderado"]["media_16_celulas_dp_ponderado"]
                                          - art["dp_medio_entre_sorteios_validos_dB_ponderado"])}}
    conf["reproduz_8.0166_e_2.1796_em_4_casas"] = (conf["arredondado_4_casas"]["media_dp_entre_sorteios"] == 8.0166
                                                   and conf["arredondado_4_casas"]["dp_entre_celulas"] == 2.1796)
    conf["reproduz_artefato_2.1_dentro_de_1e-9"] = (conf["absdiff_vs_artefato_2.1"]["media_dp"] < 1e-9
                                                    and conf["absdiff_vs_artefato_2.1"]["dp_celulas"] < 1e-9)
    # a mediana do treino inteiro (constante do 2.1) coincide com c=-110 em todos os sorteios?
    conf["mediana_treino_inteiro_igual_a_-110_em_todos_os_sorteios"] = bool(all(
        s["R6"]["mediana_treino_inteiro_ref"]["c"] == -110.0 for ch in celulas for s in dados[ch]["por_sorteio"]))

    razoes = {k: por_pred[k]["razao"] for k in CRITERIO_5}
    mecanico = {"razoes_cinco_preditores": razoes,
                "razao_minima": float(min(razoes.values())), "razao_maxima": float(max(razoes.values())),
                "todas_ge_2": all(r >= 2.0 for r in razoes.values()),
                "alguma_lt_1.5": any(r < 1.5 for r in razoes.values()),
                "mediana_validos_treino_lt_1": razoes["mediana_validos_treino"] < 1.0,
                "media_dp_por_preditor": {k: por_pred[k]["media_16_celulas_dp_entre_sorteios"] for k in CRITERIO_5},
                "dp_entre_celulas_por_preditor": {k: por_pred[k]["dp_entre_medias_das_celulas"] for k in CRITERIO_5}}

    carimbo = {"script": str(HERE), "script_sha256": SCRIPT_SHA256, "laco_comum": str(LACO),
               "laco_comum_sha256": LACO_SHA256, "script_original_reusado": "scripts/v3_2.1_3.1_deriva_calibracao.py",
               "script_original_sha256": val["script_original_sha256"], "sementes": seeds,
               "sementes_fonte": str(F2_JSON), "data_utc": agora(), "comando": " ".join(sys.argv),
               "criterio": str(CRITERIO), "venv": sys.executable, "buffer_km": 2.0, "grid_km": 10.0,
               "validacao_do_laco": {"arquivo": str(VALID), "reproduz": val["reproduz"],
                                      "sha256_arquivo": hashlib.sha256(VALID.read_bytes()).hexdigest()},
               "sha256_parciais_laco": sha_parciais}
    OUT_SORTEIO.write_text(json.dumps({**carimbo, "id": "R6_por_sorteio",
                                        "campos": "por celula e sorteio: n_validos_teste e, por preditor, c usado e MAE nos validos do teste",
                                        "celulas": por_sorteio}, indent=1, ensure_ascii=False), encoding="utf-8")
    OUT_RESUMO.write_text(json.dumps({
        **carimbo, "id": "R6_resumo",
        "preditores": {"c-100": "constante -100 dBm", "c-110": "constante -110 dBm", "c-120": "constante -120 dBm",
                       "c-130": "constante -130 dBm", "mediana_validos_treino": "mediana do RSSI dos validos do treino, por sorteio",
                       "mediana_treino_inteiro_ref": "REFERENCIA fora do criterio: mediana do treino inteiro (constante do 2.1)"},
        "estatisticas": "dp ddof=1 entre sorteios do MAE nos validos do teste, por celula; media das 16; dp ddof=1 entre as 16 "
                        "medias das celulas; razao = media dos dp / dp entre celulas; sorteios sem no valido no teste excluidos e contados",
        "conferencia_c_-110": conf, "contagem_mecanica_dos_limiares": mecanico,
        "leitura_do_criterio_texto": crit["leitura"], "se_inverter_texto": crit["se_inverter"],
        "por_preditor": por_pred, "interpretacao": "NAO feita neste artefato (fora do escopo do pipeline)"},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"conferencia": conf, "mecanico": mecanico}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consolidar", action="store_true", required=True)
    ap.parse_args()
    consolidar()


if __name__ == "__main__":
    main()
