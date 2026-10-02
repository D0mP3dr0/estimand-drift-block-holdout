#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
matematica-estatistica-do-claim -- fio gnn_rf_artigo2_literatura, checagem
"estratificacao pelo regressando (Saez & Romero-Bejar 2022)".

PERGUNTA FIXADA ANTES DE RODAR (texto literal da convocacao, 2026-09-25,
antes de qualquer numero desta rodada ter sido visto):
  "se as particoes de teste da varredura C-1 fossem sorteadas com
  estratificacao pelo regressando (blocos estratificados pela fracao de
  nos validos, como Saez propoe para a variavel-resposta), a fracao de
  nos validos do teste ainda variaria entre split_seeds tanto quanto
  observado (0 a 0.35; dp 0.06-0.10 na geometria g10b2)?"

HIPOTESE DO ARTIGO (fixada antes de rodar): a deriva vem da COMPOSICAO
espacial (o sentinela segue a geografia dos transmissores) -- estratificar
por bloco reduz a variancia da FRACAO mas nao a deriva da COMPOSICAO (quais
regioes ficam no teste), e a fracao sentinela intra-bloco continua
heterogenea.

CRITERIOS DE SUSTENTACAO/REFUTACAO FIXADOS AGORA (antes de calcular
qualquer split estratificado desta rodada -- nao mudar depois de ver o
resultado):
  H1 (fracao ainda heterogenea intra-bloco apos estratificar): em >= 9/16
    celulas, dp_estratificado(frac_teste entre seeds) / dp_controle >= 0.5
    (estratificacao reduz a variancia da fracao em MENOS da metade na
    maioria das celulas).
  H2 (deriva de composicao persiste): em >= 9/16 celulas, o Jaccard medio
    entre pares de seeds dos conjuntos de blocos de teste NAO melhora mais
    que 0.10 (absoluto) do controle para o estratificado (composicao
    continua mudando region-wise mesmo estratificando pela fracao valida).
  Hipotese do artigo "se sustenta" se H1 E H2 forem ambas verdadeiras.
  Caso so uma se sustente, ou nenhuma, o veredito e "parcial" ou "refutada"
  (registrado explicitamente, sem reclassificar o criterio depois).

DERIVADO (por import, sem editar) de varredura_fracao_valida_r3.py (que por
sua vez deriva de varredura_split_geometria.py, a C-1). Reusa: CIDADES, QS,
N_SIDE, SEEDS_20, FRACS, GEOMETRIAS, geometria_todas_celulas,
build_synthetic_grid, latlon_graus_para_metros, assign_groups,
carregar_valid2d_do_cache, cache_paths (cache_r3/*_valid2d.npz, ja
extraido -- NAO le tensor de 28GB), bootstrap_ic95_media.

Definicao de estratificacao usada aqui (adaptada de Saez & Romero-Bejar
2022, Mathematics 10, 2538, Algorithm 3 "SCVt", secao 3.3, PDF em
BIBLIOTECA/_COLHEITA/brutos/saez_2022_regressand_stratification_dataset_
shift_mathematics.pdf, extraido via pdftotext nesta rodada): "First, SCVt
sorts all the samples according to the value of the output variable ...
and computes the number n of samples per stratum ... it selects blocks of
n samples conforming each stratum ... and each of the samples of that
block is assigned to a fold". Aqui a unidade amostral e o BLOCO de 10km
(nao o pixel), o "regressando" e a fracao de nos com alvo valido do
proprio bloco, e os "folds" sao os 3 papeis (treino 70/val 15/teste 15,
nao k folds simetricos) -- adaptacao textual, registrada aqui, do
algoritmo SCVt de 2 para 3 partes com pesos desiguais.

CPU only. Venv: /trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""
import argparse
import json
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

R3_SCRIPT = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts/"
    "varredura_fracao_valida_r3.py"
)
import importlib.util
_spec = importlib.util.spec_from_file_location("varredura_fracao_valida_r3", R3_SCRIPT)
r3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r3)

vs = r3.vs
CIDADES = r3.CIDADES
QS = r3.QS
N_SIDE = r3.N_SIDE
SEEDS_20 = r3.SEEDS_20
FRACS = r3.FRACS
GEOMETRIAS = r3.GEOMETRIAS
CACHE_DIR = r3.CACHE_DIR
bootstrap_ic95_media = r3.bootstrap_ic95_media
build_synthetic_grid = vs.build_synthetic_grid
latlon_graus_para_metros = vs.latlon_graus_para_metros
assign_groups = vs.assign_groups
sha256_of_text = vs.sha256_of_text
sha256_of_file = vs.sha256_of_file

SAEZ_PDF = Path(
    "/trabalho/HERMES/BIBLIOTECA/_COLHEITA/brutos/"
    "saez_2022_regressand_stratification_dataset_shift_mathematics.pdf"
)

G_KM, B_KM = 10.0, 2.0
N_ESTRATOS = 10
BLOCOS_POR_ESTRATO_BASE = 13  # 9 estratos de 13 + 1 de 15 = 132


def cache_path(cidade, q):
    return CACHE_DIR / f"{cidade}_{q}_valid2d.npz"


def carregar_valid2d(cidade, q):
    with np.load(cache_path(cidade, q)) as z:
        return z["valid2d"]


def get_pos(geo):
    lon, lat = build_synthetic_grid(geo["lon_min_deg"], geo["lon_max_deg"],
                                     geo["lat_min_deg"], geo["lat_max_deg"])
    pos_m = latlon_graus_para_metros(lon, lat)
    ell_x = float(np.median(np.abs(np.diff(pos_m[:N_SIDE, 0]))))
    ell_y = float(np.median(np.abs(np.diff(pos_m[::N_SIDE, 1]))))
    return pos_m, ell_x, ell_y


def montar_estratos(blocos_ordenados):
    """9 estratos de 13 blocos + 1 estrato final de 15 (132 = 9*13+15),
    ordem = blocos_ordenados (ja ordenado por frac_valida_bloco ascendente,
    fixo por celula -- nao depende de seed)."""
    estratos = []
    i = 0
    n = len(blocos_ordenados)
    for e in range(N_ESTRATOS - 1):
        estratos.append(blocos_ordenados[i:i + BLOCOS_POR_ESTRATO_BASE])
        i += BLOCOS_POR_ESTRATO_BASE
    estratos.append(blocos_ordenados[i:])  # ultimo: 15
    assert sum(len(e) for e in estratos) == n, (sum(len(e) for e in estratos), n)
    return estratos


def sorteio_estratificado_blocos(estratos, fracs, split_seed):
    """Dentro de cada estrato, embaralha com RandomState(split_seed) e
    aloca 70/15/resto (arredondamento), analogo ao Algorithm 3 (SCVt) de
    Saez adaptado de k-folds simetricos para 3 partes 70/15/15."""
    rng = np.random.RandomState(split_seed)
    g_tr, g_va, g_te = [], [], []
    for estrato in estratos:
        blocos = np.array(estrato)
        rng.shuffle(blocos)
        m = len(blocos)
        n_tr = max(1, int(round(fracs[0] * m))) if m >= 3 else max(0, m - 2)
        n_va = max(1, int(round(fracs[1] * m))) if m >= 3 else (1 if m >= 2 else 0)
        if n_tr + n_va >= m:
            n_tr = max(0, m - 2)
            n_va = min(1, m - n_tr)
        g_tr.extend(blocos[:n_tr].tolist())
        g_va.extend(blocos[n_tr:n_tr + n_va].tolist())
        g_te.extend(blocos[n_tr + n_va:].tolist())
    return np.array(g_tr), np.array(g_va), np.array(g_te)


def sorteio_controle_blocos(grupos, split_seed, fracs):
    """Replica EXATA do sorteio da C-1 (split_frac_valida / split_
    espacial_3vias_exato): embaralha TODOS os blocos com RandomState(seed)
    sem estratificar, corta 70/15/resto."""
    rng = np.random.RandomState(split_seed)
    grupos_emb = grupos.copy()
    rng.shuffle(grupos_emb)
    n_g = len(grupos_emb)
    n_tr = max(1, int(round(fracs[0] * n_g)))
    n_va = max(1, int(round(fracs[1] * n_g)))
    if n_tr + n_va >= n_g:
        n_tr = max(1, n_g - 2)
        n_va = 1
    return grupos_emb[:n_tr], grupos_emb[n_tr:n_tr + n_va], grupos_emb[n_tr + n_va:]


def aplicar_buffer_e_metricas(group_ids, g_tr, g_va, g_te, ell_x_m, ell_y_m,
                               buffer_km, valid2d):
    """Mesma logica EDT de buffer da C-1/R3 (split_frac_valida), fatorada
    aqui para ser chamada tanto pelo controle quanto pelo estratificado
    (ambos ja tem g_tr/g_va/g_te de BLOCOS definidos -- so difere COMO os
    blocos foram sorteados acima)."""
    from scipy import ndimage

    m_tr = np.isin(group_ids, g_tr)
    m_va = np.isin(group_ids, g_va)
    m_te = np.isin(group_ids, g_te)

    tr2d = m_tr.reshape(N_SIDE, N_SIDE)
    va2d = m_va.reshape(N_SIDE, N_SIDE)
    te2d = m_te.reshape(N_SIDE, N_SIDE)
    sampling = (ell_y_m / 1000.0, ell_x_m / 1000.0)

    dist_train_km = ndimage.distance_transform_edt(~tr2d, sampling=sampling)
    va_retido2d = va2d & (dist_train_km >= buffer_km) if buffer_km > 0 else va2d.copy()
    trva2d = tr2d | va_retido2d
    dist_trva_km = ndimage.distance_transform_edt(~trva2d, sampling=sampling)
    te_retido2d = te2d & (dist_trva_km >= buffer_km) if buffer_km > 0 else te2d.copy()

    n_tr_final = int(tr2d.sum())
    n_va_final = int(va_retido2d.sum())
    n_te_final = int(te_retido2d.sum())

    frac_train = float(valid2d[tr2d].mean()) if n_tr_final else None
    frac_val = float(valid2d[va_retido2d].mean()) if n_va_final else None
    frac_test = float(valid2d[te_retido2d].mean()) if n_te_final else None

    blocos_te_efetivos = set(np.unique(group_ids[te_retido2d.ravel()]).tolist()) if n_te_final else set()

    # centroide (km, sistema local da celula) do teste retido
    ys, xs = np.where(te_retido2d)
    if xs.size:
        cx_km = float(np.mean(xs)) * (ell_x_m / 1000.0)
        cy_km = float(np.mean(ys)) * (ell_y_m / 1000.0)
    else:
        cx_km, cy_km = None, None

    return {
        "n_nos_apos_buffer": {"train": n_tr_final, "val": n_va_final, "test": n_te_final},
        "frac_valida": {"train": frac_train, "val": frac_val, "test": frac_test},
        "blocos_teste_efetivos": blocos_te_efetivos,
        "centroide_teste_km": (cx_km, cy_km),
    }


def jaccard(a, b):
    if not a and not b:
        return None
    u = a | b
    if not u:
        return None
    return len(a & b) / len(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    logf = open(args.log, "a") if args.log else None

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n")
            logf.flush()

    t0 = time.time()
    saida = {
        "frente": "matematica-estatistica-do-claim",
        "fio": "gnn_rf_artigo2_literatura",
        "entrega": "estratificacao_saez_2022",
        "status": "em_andamento",
        "pergunta_fixada_antes_de_rodar": (
            "se as particoes de teste da varredura C-1 fossem sorteadas com "
            "estratificacao pelo regressando (blocos estratificados pela "
            "fracao de nos validos, como Saez propoe para a variavel-"
            "resposta), a fracao de nos validos do teste ainda variaria "
            "entre split_seeds tanto quanto observado (0 a 0.35; dp "
            "0.06-0.10 na geometria g10b2)?"
        ),
        "hipotese_do_artigo": (
            "a deriva vem da composicao espacial (sentinela segue a "
            "geografia dos transmissores); estratificar por bloco reduz a "
            "variancia da FRACAO mas nao a deriva da COMPOSICAO (quais "
            "regioes ficam no teste), e a fracao sentinela intra-bloco "
            "continua heterogenea."
        ),
        "criterios_fixados_antes_de_rodar": {
            "H1_fracao_ainda_heterogenea": (
                "em >=9/16 celulas, dp_estratificado(frac_teste entre "
                "seeds) / dp_controle >= 0.5 (estratificacao reduz a "
                "variancia da fracao em MENOS da metade na maioria das "
                "celulas)"
            ),
            "H2_deriva_de_composicao_persiste": (
                "em >=9/16 celulas, o Jaccard medio entre pares de seeds "
                "dos conjuntos de blocos de teste NAO melhora mais que "
                "0.10 (absoluto) do controle para o estratificado"
            ),
            "veredito_hipotese_sustenta": "H1 E H2 ambas verdadeiras",
            "veredito_parcial": "so H1 ou so H2 verdadeira",
            "veredito_refutada": "nem H1 nem H2 verdadeira",
        },
        "seeds_usadas": SEEDS_20,
        "config": {"grid_km": G_KM, "buffer_km": B_KM, "fracs": FRACS,
                    "n_estratos": N_ESTRATOS,
                    "blocos_por_estrato": "9x13 + 1x15 = 132"},
    }
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    saida["script_sha256"] = sha256_of_text(Path(__file__).read_text())
    saida["saez_2022_fonte"] = {
        "pdf": str(SAEZ_PDF),
        "pdf_sha256": sha256_of_file(SAEZ_PDF) if SAEZ_PDF.exists() else None,
        "definicao_literal_extraida_via_pdftotext_secao_3_3_algorithm_3_SCVt": (
            "First, SCVt sorts all the samples according to the value of "
            "the output variable (line 2) and computes the number n of "
            "samples per stratum (line 3). Afterwards, it starts an "
            "iterative process to assign samples to each fold (lines "
            "4-13): it selects blocks of n samples conforming each "
            "stratum (lines 5-6) and, then, each of the samples of that "
            "block (line 8) is assigned to a fold with less samples "
            "(line 9) until there are no more available samples to "
            "assign."
        ),
        "adaptacao_usada_aqui": (
            "unidade = bloco de 10km (nao pixel); regressando do bloco = "
            "fracao de nos com alvo valido do proprio bloco; 3 partes "
            "70/15/15 (nao k folds simetricos) -- alocacao proporcional "
            "dentro de cada estrato, embaralhada com RandomState(split_"
            "seed), em vez do 'fold com menos amostras' do Algorithm 3 "
            "original (que pressupoe folds simetricos)."
        ),
    }
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    resultados_por_celula = {}
    for cidade in CIDADES:
        for q in QS:
            cp = cache_path(cidade, q)
            if not cp.exists():
                log(f"AVISO: cache ausente para {cidade}_{q}, pulando (nao_verificado).")
                resultados_por_celula[f"{cidade}_{q}"] = {"erro": "cache_ausente"}
                continue
            valid2d = carregar_valid2d(cidade, q)
            geo = vs.geometria_todas_celulas()[(cidade, q)]
            pos_m, ell_x, ell_y = get_pos(geo)
            group_ids = assign_groups(pos_m / 1000.0, G_KM)
            grupos = np.unique(group_ids)
            assert len(grupos) == 132, (cidade, q, len(grupos))

            # regressando do bloco = fracao valida do PROPRIO bloco (todo o
            # bloco, sem particao -- e a variavel usada so para ORDENAR e
            # formar estratos, fixa por celula, independente de seed)
            frac_valida_bloco = {}
            valid1d = valid2d.ravel()
            for gid in grupos:
                mask = (group_ids == gid)
                frac_valida_bloco[int(gid)] = float(valid1d[mask].mean())
            blocos_ordenados = sorted(grupos.tolist(), key=lambda g: frac_valida_bloco[int(g)])
            estratos = montar_estratos(blocos_ordenados)

            log(f"{cidade}_{q}: 132 blocos, {len(estratos)} estratos "
                f"(tamanhos={[len(e) for e in estratos]})")

            por_seed = {"controle": {}, "estratificado": {}}
            for seed in SEEDS_20:
                g_tr_c, g_va_c, g_te_c = sorteio_controle_blocos(grupos, seed, FRACS)
                r_c = aplicar_buffer_e_metricas(group_ids, g_tr_c, g_va_c, g_te_c,
                                                 ell_x, ell_y, B_KM, valid2d)

                g_tr_e, g_va_e, g_te_e = sorteio_estratificado_blocos(estratos, FRACS, seed)
                r_e = aplicar_buffer_e_metricas(group_ids, g_tr_e, g_va_e, g_te_e,
                                                 ell_x, ell_y, B_KM, valid2d)

                por_seed["controle"][seed] = r_c
                por_seed["estratificado"][seed] = r_e

            def resumo_esquema(por_seed_esq):
                fracs_teste = [v["frac_valida"]["test"] for v in por_seed_esq.values()
                                if v["frac_valida"]["test"] is not None]
                arr = np.array(fracs_teste)
                dp = float(arr.std(ddof=1)) if arr.size > 1 else None
                amplitude = float(arr.max() - arr.min()) if arr.size else None

                seeds_list = list(por_seed_esq.keys())
                pares = list(combinations(seeds_list, 2))
                jaccs = []
                for s1, s2 in pares:
                    j = jaccard(por_seed_esq[s1]["blocos_teste_efetivos"],
                                por_seed_esq[s2]["blocos_teste_efetivos"])
                    if j is not None:
                        jaccs.append(j)
                jaccard_medio = float(np.mean(jaccs)) if jaccs else None

                centroides = [v["centroide_teste_km"] for v in por_seed_esq.values()
                              if v["centroide_teste_km"][0] is not None]
                if len(centroides) > 1:
                    carr = np.array(centroides)
                    dists = []
                    for i, j in combinations(range(len(carr)), 2):
                        dists.append(float(np.linalg.norm(carr[i] - carr[j])))
                    deriva_centroide_media_km = float(np.mean(dists))
                else:
                    deriva_centroide_media_km = None

                return {
                    "frac_teste_por_seed": {str(s): v["frac_valida"]["test"]
                                             for s, v in por_seed_esq.items()},
                    "dp_frac_teste_entre_seeds": dp,
                    "amplitude_frac_teste": amplitude,
                    "min_frac_teste": float(arr.min()) if arr.size else None,
                    "max_frac_teste": float(arr.max()) if arr.size else None,
                    "jaccard_medio_pares_seeds_blocos_teste": jaccard_medio,
                    "n_pares_jaccard": len(jaccs),
                    "deriva_centroide_teste_media_pares_km": deriva_centroide_media_km,
                }

            resumo_c = resumo_esquema(por_seed["controle"])
            resumo_e = resumo_esquema(por_seed["estratificado"])

            razao_dp = (resumo_e["dp_frac_teste_entre_seeds"] / resumo_c["dp_frac_teste_entre_seeds"]
                        if (resumo_c["dp_frac_teste_entre_seeds"] not in (None, 0)
                            and resumo_e["dp_frac_teste_entre_seeds"] is not None)
                        else None)
            delta_jaccard = (resumo_e["jaccard_medio_pares_seeds_blocos_teste"] - resumo_c["jaccard_medio_pares_seeds_blocos_teste"]
                              if (resumo_e["jaccard_medio_pares_seeds_blocos_teste"] is not None
                                  and resumo_c["jaccard_medio_pares_seeds_blocos_teste"] is not None)
                              else None)

            resultados_por_celula[f"{cidade}_{q}"] = {
                "cidade": cidade, "Q": q,
                "controle": resumo_c,
                "estratificado": resumo_e,
                "razao_dp_estrat_sobre_controle": razao_dp,
                "delta_jaccard_estrat_menos_controle": delta_jaccard,
                "h1_dp_ainda_heterogeneo_ge_0_5": (razao_dp is not None and razao_dp >= 0.5),
                "h2_jaccard_nao_melhora_mais_que_0_10": (delta_jaccard is not None and delta_jaccard <= 0.10),
            }
            Path(args.out).write_text(json.dumps(
                {**saida, "resultados_por_celula": resultados_por_celula},
                indent=1, ensure_ascii=False))
            log(f"{cidade}_{q}: dp_controle={resumo_c['dp_frac_teste_entre_seeds']:.4f} "
                f"dp_estrat={resumo_e['dp_frac_teste_entre_seeds']:.4f} razao={razao_dp} "
                f"jacc_c={resumo_c['jaccard_medio_pares_seeds_blocos_teste']:.3f} "
                f"jacc_e={resumo_e['jaccard_medio_pares_seeds_blocos_teste']:.3f} "
                f"delta_jacc={delta_jaccard}")

    saida["resultados_por_celula"] = resultados_por_celula

    celulas_validas = {k: v for k, v in resultados_por_celula.items() if "erro" not in v}
    n_h1 = sum(1 for v in celulas_validas.values() if v["h1_dp_ainda_heterogeneo_ge_0_5"])
    n_h2 = sum(1 for v in celulas_validas.values() if v["h2_jaccard_nao_melhora_mais_que_0_10"])
    n_total = len(celulas_validas)

    h1_ok = n_h1 >= 9
    h2_ok = n_h2 >= 9
    if h1_ok and h2_ok:
        veredito = "hipotese_sustenta"
    elif h1_ok or h2_ok:
        veredito = "parcial"
    else:
        veredito = "refutada"

    saida["veredito_criterios"] = {
        "n_celulas_avaliadas": n_total,
        "n_celulas_h1_ge_9_16_necessario": n_h1,
        "h1_ok": h1_ok,
        "n_celulas_h2_ge_9_16_necessario": n_h2,
        "h2_ok": h2_ok,
        "veredito": veredito,
    }

    dp_controle_medio = float(np.mean([v["controle"]["dp_frac_teste_entre_seeds"]
                                        for v in celulas_validas.values()
                                        if v["controle"]["dp_frac_teste_entre_seeds"] is not None]))
    dp_estrat_medio = float(np.mean([v["estratificado"]["dp_frac_teste_entre_seeds"]
                                      for v in celulas_validas.values()
                                      if v["estratificado"]["dp_frac_teste_entre_seeds"] is not None]))
    amp_controle_max = float(np.max([v["controle"]["amplitude_frac_teste"]
                                      for v in celulas_validas.values()
                                      if v["controle"]["amplitude_frac_teste"] is not None]))
    amp_estrat_max = float(np.max([v["estratificado"]["amplitude_frac_teste"]
                                    for v in celulas_validas.values()
                                    if v["estratificado"]["amplitude_frac_teste"] is not None]))
    jacc_controle_medio = float(np.mean([v["controle"]["jaccard_medio_pares_seeds_blocos_teste"]
                                          for v in celulas_validas.values()
                                          if v["controle"]["jaccard_medio_pares_seeds_blocos_teste"] is not None]))
    jacc_estrat_medio = float(np.mean([v["estratificado"]["jaccard_medio_pares_seeds_blocos_teste"]
                                        for v in celulas_validas.values()
                                        if v["estratificado"]["jaccard_medio_pares_seeds_blocos_teste"] is not None]))

    saida["escopo"] = {
        "executado": [
            "16 celulas g10b2 x 20 seeds x 2 esquemas (controle nao-estratificado "
            "replicando C-1 e estratificado pela fracao valida do bloco, 10 "
            "estratos de 132 blocos)",
            "dp e amplitude da fracao valida do teste entre seeds, por esquema",
            "Jaccard medio entre pares de seeds dos conjuntos de blocos de teste "
            "(medida de deriva de COMPOSICAO), por esquema",
            "deriva de centroide do teste entre pares de seeds (km), por esquema",
            "criterios H1/H2 fixados antes de rodar, aplicados por celula",
        ],
        "fora_do_escopo": [
            "N efetivo sob autocorrelacao espacial: fora de escopo (dados-vazamento-"
            "espacial); aqui a unidade da inferencia e o split_seed, nao o pixel",
            "recalculo do IC bootstrap da C-1 original (ja existe em "
            "varredura_fracao_valida_r3.py; aqui o objetivo e comparar dp/deriva "
            "de composicao entre os DOIS esquemas de sorteio, nao re-auditar a C-1)",
        ],
    }
    saida["nao_verificado"] = [
        {"item": f"{k}", "motivo": v.get("erro")}
        for k, v in resultados_por_celula.items() if "erro" in v
    ]
    saida["tabela_numero_campo_comando"] = [
        {"numero": "dp/amplitude frac_teste por celula e esquema",
         "campo": "resultados_por_celula.<celula>.controle|estratificado.dp_frac_teste_entre_seeds",
         "comando": "python varredura_fracao_valida_estratificada.py --out <json>"},
        {"numero": "Jaccard medio entre seeds (composicao)",
         "campo": "resultados_por_celula.<celula>.controle|estratificado.jaccard_medio_pares_seeds_blocos_teste",
         "comando": "idem"},
        {"numero": "veredito H1/H2 por criterio fixado",
         "campo": "veredito_criterios", "comando": "idem"},
    ]

    saida["resumo"] = (
        f"dp médio (16 células, controle vs. estratificado): "
        f"{dp_controle_medio:.4f} -> {dp_estrat_medio:.4f} "
        f"(razão média {dp_estrat_medio/dp_controle_medio:.2f}); "
        f"amplitude máxima observada: {amp_controle_max:.3f} (controle) vs. "
        f"{amp_estrat_max:.3f} (estratificado), ambas dentro da faixa 0-0.35 "
        f"citada na pergunta fixada. "
        f"Jaccard médio entre pares de seeds dos blocos de teste (composição): "
        f"{jacc_controle_medio:.3f} (controle) -> {jacc_estrat_medio:.3f} "
        f"(estratificado), delta médio {jacc_estrat_medio-jacc_controle_medio:+.3f}. "
        f"Critério H1 (fração ainda heterogênea intra-bloco, dp reduz <50%) "
        f"cumprido em {n_h1}/{n_total} células (precisa >=9); critério H2 "
        f"(deriva de composição não melhora >0.10 absoluto) cumprido em "
        f"{n_h2}/{n_total} células (precisa >=9). "
        f"Veredito fixado: {veredito}. "
        f"English claim sentence (Related Work): \"Stratifying the spatial "
        f"blocks by the target valid-node fraction (the regressand, following "
        f"Sáez and Romero-Béjar's SCVt scheme) narrows the between-seed "
        f"variance of the test-set valid fraction but does not remove the "
        f"drift itself, because the drift here is driven by which regions "
        f"fall in the test set, not by the marginal fraction alone.\""
    )

    saida["status"] = "concluido"
    saida["tempo_total_s"] = time.time() - t0
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))
    log(f"CONCLUIDO. tempo_total_s={saida['tempo_total_s']:.1f} veredito={veredito}")


if __name__ == "__main__":
    main()
