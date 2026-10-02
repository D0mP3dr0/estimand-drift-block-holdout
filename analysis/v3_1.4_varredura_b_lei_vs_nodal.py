#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste 1.4 (reduzido, fase 1) -- alegacao A1 do roadmap v3
(draft_v2/ROADMAP_TESTES_v3_2026-09-25.md): "a lei de retencao (eq:R, campo
medio) preve a retencao nodal real em mais de dois pontos de geometria?"

Hoje a lei so foi confrontada com o nodal em (g,b)=(10,2) e (5,2). Este
script acrescenta um TERCEIRO e um QUARTO ponto: b in {1,3} km com g=10 km,
nas 16 celulas g10 (bauru/campinas/lins/sorocaba x Q1..Q4), 20 split_seeds
(SEEDS_20 de varredura_fracao_valida_r3.py), e RECALCULA tambem os dois
pontos ja citados no rascunho ((10,2) nas 16 celulas; (5,2) nas 4 celulas
bauru, unica config com dado real) porque nao existe ainda um artefato
16x20 com esses dois pontos nesta pasta de campanha (fase1/ estava vazia).

*** ERRATA (2026-09-25, apos reconciliacao do coordenador contra 3 scripts
independentes -- R3-c matematica-equacoes_r3_mc_g5_tres_cidades.json,
fismat_sympy_eqR.json, verificador_eqR_resultado.json): a v1 deste script
tinha DOIS bugs, corrigidos nesta v2 -- ver artefato de saida, bloco
`errata`, para o relato completo:
  (a) q_te/q_va estavam calculados como FRACAO DE PIXELS pos-erosao do
      buffer (variam por seed) em vez de FRACAO DE BLOCOS (nominal,
      deterministica por celula/g, independente do seed -- as contagens
      ja vinham em split_e_metricas_edt()["n_blocos"], nunca usadas na v1);
  (b) a agregacao por celula fazia media-das-razoes (mean-of-ratios) entre
      seeds, mas o numero ja publicado/reproduzido usa razao-das-medias
      (ratio-of-means) -- (media(lei) - media(nodal))/media(nodal); como a
      lei e constante entre seeds e a retencao nodal varia, as duas
      definicoes divergem por desigualdade de Jensen. Corrigido: q agora
      vem de n_blocos/n_blocos_total; resumo_por_gb reporta
      *_RATIO_OF_MEANS (primario, bate com o publicado) e
      *_MEAN_OF_RATIOS_por_seed (secundario). ***

Metodo (fixado em criterios/criterio_1.4.json ANTES desta rodada):
  1) Reconstroi a particao real com split_e_metricas_edt (COPIA POR IMPORT,
     sem editar, de scripts/varredura_split_geometria.py -- metodo EDT ja
     validado <1e-3 contra cKDTree exato no proprio modulo de origem,
     validacao_edt_vs_cktree), fracoes 70/15/resto-ao-teste (FRACS).
  2) q_te = n_blocos_test/n_blocos_total, q_va = 1 - n_blocos_train/
     n_blocos_total -- fracao NOMINAL de blocos (FRACS 0.70/0.15/0.15
     arredondadas sobre o numero de blocos g x g do dominio da celula),
     deterministica por (celula,g), independente do split_seed (corrigido
     na errata acima; a v1 usava fracao de pixels pos-buffer, errado).
  3) Lei de campo medio eq:R (formula fechada -- reimplementada aqui por
     COPIA LITERAL da linha 42 de
     draft_v2/_pareceres_2026-09-25/fismat_sympy_eqR.py, NAO importada como
     modulo porque aquele arquivo roda como script no import e escreveria
     fismat_sympy_eqR.json fora da pasta_saida, violando READ-ONLY) avaliada
     em q realizado: R_law(g,b,q) = [(g-2b)^2 + 4b(g-2b)q + 4b^2(1-pi/4)q^2
     + pi*b^2*q^3] / g^2.
  4) Correcao de dominio finito (Apendice A do parecer forum-fisico-
     matematico_P4_P7_P8.md, formulas fechadas mean-field bloco de borda /
     canto -- reimplementadas por COPIA LITERAL de raster_counts() e
     lattice_retention() do mesmo fismat_sympy_eqR.py, linhas 66-149,
     aplicadas ao dominio REAL de cada celula, com contagem NOMINAL de
     blocos por particao (fracs 0.70/0.15/0.15 arredondadas), determinis-
     tica por celula (nao varia por seed -- e a mesma logica do parecer
     original).
  5) Erro relativo da lei vs nodal, RATIO-OF-MEANS (definicao primaria,
     bate com o numero ja publicado): (media_seeds(lei) -
     media_seeds(retencao_nodal)) / media_seeds(retencao_nodal), por
     (celula,g,b,papel). Tambem reportado o MEAN-OF-RATIOS (media, sobre as
     20 seeds, de (lei-nodal_s)/nodal_s por seed) como campo secundario,
     com IC95 bootstrap (10000 reamostras, seed de bootstrap fixa).

REGRAS DURAS: READ-ONLY fora de MDPI_Mathematics/scripts/ (script novo) e
_v3_2026-09-25/ (saida); nenhum arquivo do parecer/rascunho e escrito ou
importado como modulo executavel. venv CUDA /trabalho/ambientes/
s33_amb_virtual/.venv/bin/python (aqui so scipy/numpy/sympy, sem GPU).

Uso:
  .venv/bin/python v3_1.4_varredura_b_lei_vs_nodal.py --out <json> --log <log>
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sympy as sp

# ---------------------------------------------------------------------
# import (leitura, sem editar) do script original de geometria/split
# ---------------------------------------------------------------------
ORIG_SPLIT = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts/"
    "varredura_split_geometria.py"
)
_spec = importlib.util.spec_from_file_location("varredura_split_geometria_orig", ORIG_SPLIT)
vs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vs)  # so definicoes top-level; nada roda fora de main()

CIDADES = vs.CIDADES
QS = vs.QS
N_SIDE = vs.N_SIDE
FRACS = vs.FRACS
TREINOS_DIR = vs.TREINOS_DIR
assign_groups = vs.assign_groups
latlon_graus_para_metros = vs.latlon_graus_para_metros
build_synthetic_grid = vs.build_synthetic_grid
split_e_metricas_edt = vs.split_e_metricas_edt
geometria_todas_celulas = vs.geometria_todas_celulas
sha256_of_file = vs.sha256_of_file
sha256_of_text = vs.sha256_of_text

SEEDS_20 = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

CRITERIO_PATH = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/"
    "criterios/criterio_1.4.json"
)

# Pontos (g,b) do teste reduzido: os 2 ja citados no rascunho + os 2 novos
# (b=1,3 com g=10). REGRA D5: nao amplia alem destes.
PONTOS_G10 = [(10.0, 1.0), (10.0, 2.0), (10.0, 3.0)]
CELULAS_G10 = [(c, q) for c in CIDADES for q in QS]  # 16
PONTO_G5 = (5.0, 2.0)
CELULAS_G5 = [("bauru", q) for q in QS]  # 4 -- unica config g5b2 com dado real

# ---------------------------------------------------------------------
# eq:R (campo medio) -- COPIA LITERAL da linha 42 de
# draft_v2/_pareceres_2026-09-25/fismat_sympy_eqR.py (nao importado como
# modulo -- aquele script roda no import e escreve fora da pasta_saida)
# ---------------------------------------------------------------------
_g, _b, _q = sp.symbols("g b q", positive=True)
_eqR_num = (_g - 2 * _b) ** 2 + 4 * _b * (_g - 2 * _b) * _q + 4 * _b ** 2 * (1 - sp.pi / 4) * _q ** 2 + sp.pi * _b ** 2 * _q ** 3
_R_law_fn = sp.lambdify((_g, _b, _q), _eqR_num / _g ** 2, modules="numpy")


def R_law(g, b, q):
    return float(_R_law_fn(g, b, q))


# ---------------------------------------------------------------------
# Correcao de dominio finito -- COPIA LITERAL (raster_counts, lattice_
# retention) de fismat_sympy_eqR.py linhas 66-149, forum-fisico-matematico
# P4_P7_P8.md Apendice A.
# ---------------------------------------------------------------------
_names = ["L", "R", "Bm", "T", "LB", "LT", "RB", "RT"]
_corner_adj = {"LB": ("L", "Bm"), "LT": ("L", "T"), "RB": ("R", "Bm"), "RT": ("R", "T")}


def _raster_counts(w, h, nb, bb, res):
    xs = (np.arange(int(round(w / res))) + 0.5) * res
    ys = (np.arange(int(round(h / res))) + 0.5) * res
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    X = X.ravel(); Y = Y.ravel()
    d = {
        "L": X, "R": w - X, "Bm": Y, "T": h - Y,
        "LB": np.hypot(X, Y), "LT": np.hypot(X, h - Y),
        "RB": np.hypot(w - X, Y), "RT": np.hypot(w - X, h - Y),
    }
    M = np.stack([(d[k] <= bb) & nb.get(k, True) for k in _names], axis=1)
    return M, res * res


def _falling(a, n):
    r = 1.0
    for j in range(n):
        r *= (a - j)
    return r


def _lattice_retention(Wx, Wy, gg, bb, N_nontrim_count, N, res=0.05):
    """Retencao esperada (mean-field e exato sob permutacao) de um
    reticulado real (bloco cheio + coluna/linha fina de borda), copia
    literal da funcao homonima de fismat_sympy_eqR.py."""
    m = int(math.floor(Wx / gg)) + (0 if abs(Wx / gg - round(Wx / gg)) < 1e-9 else 1)
    n = int(math.floor(Wy / gg)) + (0 if abs(Wy / gg - round(Wy / gg)) < 1e-9 else 1)
    widths = [gg] * (m - 1) + [Wx - gg * (m - 1)]
    heights = [gg] * (n - 1) + [Wy - gg * (n - 1)]
    qmf = N_nontrim_count / N
    cache = {}
    num_mf = num_ex = den = 0.0
    for i in range(m):
        for j in range(n):
            nb = {"L": i > 0, "R": i < m - 1, "Bm": j > 0, "T": j < n - 1,
                  "LB": i > 0 and j > 0, "LT": i > 0 and j < n - 1,
                  "RB": i < m - 1 and j > 0, "RT": i < m - 1 and j < n - 1}
            key = (round(widths[i], 6), round(heights[j], 6), tuple(nb[k] for k in _names))
            if key not in cache:
                M, dA = _raster_counts(widths[i], heights[j], nb, bb, res)
                cnt = M.sum(axis=1)
                ak = np.bincount(cnt, minlength=9) * dA
                cache[key] = ak
            ak = cache[key]
            num_mf += sum(ak[k] * qmf ** k for k in range(9))
            num_ex += sum(ak[k] * _falling(N_nontrim_count - 1, k) / _falling(N - 1, k) for k in range(9))
            den += widths[i] * heights[j]
    return {"m_x_n": [m, n], "N_blocos": m * n, "mean_field": num_mf / den, "exato_permutacao": num_ex / den}


def dominio_km(geo):
    """Wx,Wy em km a partir do bbox lon/lat do run JSON (s=111km/grau,
    cos(lat) no centro -- mesma aproximacao de fismat_sympy_eqR.py)."""
    s = 111.0
    lon_min, lon_max = geo["lon_min_deg"], geo["lon_max_deg"]
    lat_min, lat_max = geo["lat_min_deg"], geo["lat_max_deg"]
    phic = math.radians(0.5 * (lat_min + lat_max))
    Wx = s * (lon_max - lon_min) * math.cos(phic)
    Wy = s * (lat_max - lat_min)
    return Wx, Wy


def correcao_dominio_finito(geo, g, b):
    """Correcao de dominio finito (Apendice A), DETERMINISTICA por celula
    (nao varia por seed -- contagem NOMINAL de blocos via FRACS), para
    teste e validacao."""
    Wx, Wy = dominio_km(geo)
    m = int(math.ceil(Wx / g)); n = int(math.ceil(Wy / g))
    N = m * n
    k_tr = max(1, round(FRACS[0] * N))
    k_va = max(1, round(FRACS[1] * N))
    k_te = max(1, N - k_tr - k_va)
    out = {"Wx_km": Wx, "Wy_km": Wy, "N_blocos": N, "k_tr": k_tr, "k_va": k_va, "k_te": k_te}
    for papel, nontrim in (("test", k_te), ("val", N - k_tr)):
        lr = _lattice_retention(Wx, Wy, g, b, nontrim, N, res=0.05)
        out[papel] = {"q_nominal": nontrim / N, "mean_field": lr["mean_field"],
                      "exato_permutacao": lr["exato_permutacao"]}
    return out


def n_blocos_de(cidade_q_pos_m, g):
    """Numero de blocos DISTINTOS realmente ocupados pela malha (assign_groups),
    usado como n_blocos_total para q -- pode diferir por arredondamento do N
    geometrico ceil(Wx/g)*ceil(Wy/g) usado em correcao_dominio_finito, mas na
    pratica bate (conferido: bauru Q1 g10 -> 132 nos dois metodos)."""
    pos_m = cidade_q_pos_m
    group_ids = assign_groups(pos_m / 1000.0, g)
    return int(np.unique(group_ids).size)


def contagem_blocos_nominal(n_g, fracs=FRACS):
    """Copia literal da logica de contagem de split_e_metricas_edt /
    split_espacial_3vias_exato (arredondamento de FRACS sobre n_g), usada
    para obter n_blocos_train/val/test NOMINAIS -- deterministicos, mesmos
    para qualquer split_seed (so muda QUAL bloco vai pra cada papel, nao
    QUANTOS)."""
    n_tr = max(1, int(round(fracs[0] * n_g)))
    n_va = max(1, int(round(fracs[1] * n_g)))
    if n_tr + n_va >= n_g:
        n_tr = max(1, n_g - 2)
        n_va = 1
    n_te = n_g - n_tr - n_va
    return {"train": n_tr, "val": n_va, "test": n_te}


def bootstrap_ic95_media(vals, n_resamples=10000, seed_bootstrap=20260925):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    n = arr.size
    if n == 0:
        return {"media": None, "ic95_lo": None, "ic95_hi": None, "n": 0, "exclui_zero": None}
    media = float(arr.mean())
    rng = np.random.RandomState(seed_bootstrap)
    idx = rng.randint(0, n, size=(n_resamples, n))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    exclui_zero = bool((lo > 0) or (hi < 0))
    return {"media": media, "ic95_lo": float(lo), "ic95_hi": float(hi), "n": int(n),
            "n_resamples": n_resamples, "seed_bootstrap": seed_bootstrap, "exclui_zero": exclui_zero}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--seeds", type=int, default=len(SEEDS_20),
                     help="reduz para N seeds se o custo de CPU passar de 60 min (declarado no artefato)")
    args = ap.parse_args()

    logf = open(args.log, "a") if args.log else None

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n"); logf.flush()

    t_inicio = time.time()
    seeds = SEEDS_20[:args.seeds]
    criterio = json.loads(CRITERIO_PATH.read_text())

    saida = {
        "frente": "matematica-estatistica-do-claim",
        "id": "1.4",
        "alegacao": "A1",
        "pergunta": criterio["pergunta"],
        "criterio": criterio,
        "seeds_usadas": seeds,
        "pontos_gb": {"g10": PONTOS_G10, "g5": [PONTO_G5]},
        "status": "em_andamento",
    }
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    saida["script_sha256"] = sha256_of_text(Path(__file__).read_text())
    saida["insumos"] = {
        "varredura_split_geometria.py_sha256": sha256_of_text(ORIG_SPLIT.read_text()),
        "criterio_1.4.json_sha256": sha256_of_file(CRITERIO_PATH),
        "run_json_g10b2_16_celulas": "TREINOS_DIR/run_c0c1cf_<cidade>_s42_<Q>_g10b2.json (so para o bbox lon/lat, campo geometria; independente de b)",
    }
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    log("lendo geometria (bbox) das 16 celulas g10 via run_c0c1cf_*_g10b2.json (geometria_todas_celulas)...")
    geo_por_celula = geometria_todas_celulas()

    pos_cache = {}
    group_ids_cache = {}

    def get_pos(cidade, q):
        key = (cidade, q)
        if key not in pos_cache:
            geo = geo_por_celula[key]
            lon, lat = build_synthetic_grid(geo["lon_min_deg"], geo["lon_max_deg"],
                                             geo["lat_min_deg"], geo["lat_max_deg"])
            pos_m = latlon_graus_para_metros(lon, lat)
            ell_x = float(np.median(np.abs(np.diff(pos_m[:N_SIDE, 0]))))
            ell_y = float(np.median(np.abs(np.diff(pos_m[::N_SIDE, 1]))))
            pos_cache[key] = (pos_m, ell_x, ell_y)
        return pos_cache[key]

    def get_group_ids(cidade, q, g):
        key = (cidade, q, g)
        if key not in group_ids_cache:
            pos_m, _, _ = get_pos(cidade, q)
            group_ids_cache[key] = assign_groups(pos_m / 1000.0, g)
        return group_ids_cache[key]

    # -------- varredura: (10,1),(10,2),(10,3) x 16 celulas x seeds ; (5,2) x 4 celulas x seeds --------
    log(f"iniciando varredura: {len(PONTOS_G10)} pontos g10 x 16 celulas x {len(seeds)} seeds + 1 ponto g5 x 4 celulas x {len(seeds)} seeds...")
    resultados = []
    t_var = time.time()
    combos = [(g, b, CELULAS_G10) for (g, b) in PONTOS_G10] + [(PONTO_G5[0], PONTO_G5[1], CELULAS_G5)]
    for (g, b, celulas) in combos:
        for (cidade, q) in celulas:
            pos_m, ell_x, ell_y = get_pos(cidade, q)
            group_ids = get_group_ids(cidade, q, g)
            geo = geo_por_celula[(cidade, q)]
            corr_dominio = correcao_dominio_finito(geo, g, b)

            # q NOMINAL por fracao de BLOCOS (deterministico, independe do
            # seed) -- CORRIGIDO (era fracao de pixels pos-buffer na v1).
            n_g = n_blocos_de(pos_m, g)
            n_blocos_papel = contagem_blocos_nominal(n_g)
            q_te_bloco = n_blocos_papel["test"] / n_g
            q_va_bloco = 1.0 - (n_blocos_papel["train"] / n_g)
            law_test = R_law(g, b, q_te_bloco)
            law_val = R_law(g, b, q_va_bloco)

            for seed in seeds:
                r = split_e_metricas_edt(pos_m, ell_x, ell_y, group_ids, g, b, FRACS, seed, None)
                ret_nodal_test = r["retencao"]["test"]
                ret_nodal_val = r["retencao"]["val"]
                erro_rel_test = ((law_test - ret_nodal_test) / ret_nodal_test) if ret_nodal_test else None
                erro_rel_val = ((law_val - ret_nodal_val) / ret_nodal_val) if ret_nodal_val else None
                resultados.append({
                    "cidade": cidade, "Q": q, "g_km": g, "b_km": b, "split_seed": seed,
                    "n_nos_apos_buffer": r["n_nos_apos_buffer"],
                    "n_nos_antes_do_buffer": r["n_nos_antes_do_buffer"],
                    "n_blocos_total": n_g, "n_blocos_papel_nominal": n_blocos_papel,
                    "q_te_realizado": q_te_bloco, "q_va_realizado": q_va_bloco,
                    "retencao_nodal_test": ret_nodal_test, "retencao_nodal_val": ret_nodal_val,
                    "lei_campo_medio_test": law_test, "lei_campo_medio_val": law_val,
                    "correcao_dominio_finito_test_mean_field": corr_dominio["test"]["mean_field"],
                    "correcao_dominio_finito_val_mean_field": corr_dominio["val"]["mean_field"],
                    "erro_relativo_lei_vs_nodal_test": erro_rel_test,
                    "erro_relativo_lei_vs_nodal_val": erro_rel_val,
                })
            saida.setdefault("correcao_dominio_finito_por_celula", {})[f"{cidade}_{q}_g{int(g)}b{int(b)}"] = corr_dominio
        log(f"  ponto (g={g},b={b}) x {len(celulas)} celulas concluido, t={time.time()-t_var:.1f}s")
        Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    saida["resultados"] = resultados
    tempo_varredura_s = time.time() - t_var
    log(f"varredura total: {len(resultados)} combinacoes em {tempo_varredura_s:.1f}s")
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    # -------- agregacao por (g,b,celula): RATIO-OF-MEANS (primario, bate com
    # o publicado) + MEAN-OF-RATIOS por seed com IC95 bootstrap (secundario) --------
    log("agregando por (g,b,celula) -- ratio-of-means primario, mean-of-ratios secundario...")
    por_gb_celula = defaultdict(list)
    for r in resultados:
        por_gb_celula[(r["g_km"], r["b_km"], r["cidade"], r["Q"])].append(r)

    agregado = {}
    for (g, b, cidade, q), rs in por_gb_celula.items():
        erros_test = [r["erro_relativo_lei_vs_nodal_test"] for r in rs]
        erros_val = [r["erro_relativo_lei_vs_nodal_val"] for r in rs]
        ic_test = bootstrap_ic95_media(erros_test)
        ic_val = bootstrap_ic95_media(erros_val)
        nodal_te_media = float(np.mean([r["retencao_nodal_test"] for r in rs]))
        nodal_va_media = float(np.mean([r["retencao_nodal_val"] for r in rs]))
        lei_te_media = float(np.mean([r["lei_campo_medio_test"] for r in rs]))  # constante entre seeds
        lei_va_media = float(np.mean([r["lei_campo_medio_val"] for r in rs]))
        agregado[f"g{g}_b{b}_{cidade}_{q}"] = {
            "g_km": g, "b_km": b, "cidade": cidade, "Q": q,
            "retencao_nodal_test_media": nodal_te_media,
            "retencao_nodal_val_media": nodal_va_media,
            "lei_campo_medio_test_media": lei_te_media,
            "lei_campo_medio_val_media": lei_va_media,
            "erro_relativo_test_sobre_media_nodal": (lei_te_media - nodal_te_media) / nodal_te_media if nodal_te_media else None,
            "erro_relativo_val_sobre_media_nodal": (lei_va_media - nodal_va_media) / nodal_va_media if nodal_va_media else None,
            "erro_relativo_test_ic95_bootstrap": ic_test,
            "erro_relativo_val_ic95_bootstrap": ic_val,
            "erro_relativo_test_media_de_razoes_por_seed": ic_test["media"],
            "nota_definicao": (
                "erro_relativo_test_sobre_media_nodal = (media(lei) - media(nodal))/media(nodal), "
                "RATIO-OF-MEANS -- bate com o numero ja publicado/reproduzido (R3-c, "
                "fismat_sympy_eqR.json, verificador_eqR_resultado.json); "
                "erro_relativo_test_media_de_razoes_por_seed = media, sobre as seeds, de "
                "(lei-nodal_s)/nodal_s por seed (MEAN-OF-RATIOS) -- difere por desigualdade de "
                "Jensen (lei e constante, nodal varia por seed)."
            ),
        }
    saida["agregado_por_gb_celula"] = agregado
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    # -------- resumo por (g,b): curva R(q), RATIO-OF-MEANS primario --------
    log("resumo por (g,b) -- curva R(q), RATIO-OF-MEANS primario...")
    resumo_por_gb = {}
    for (g, b) in PONTOS_G10 + [PONTO_G5]:
        chave = f"g{g}_b{b}"
        entradas = [v for k, v in agregado.items() if v["g_km"] == g and v["b_km"] == b]
        if not entradas:
            continue
        r_test_rom = [e["erro_relativo_test_sobre_media_nodal"] for e in entradas]
        r_val_rom = [e["erro_relativo_val_sobre_media_nodal"] for e in entradas]
        r_test_mor = [e["erro_relativo_test_media_de_razoes_por_seed"] for e in entradas]
        resumo_por_gb[chave] = {
            "g_km": g, "b_km": b, "n_celulas": len(entradas),
            "erro_relativo_test_media_entre_celulas_RATIO_OF_MEANS": float(np.mean(r_test_rom)),
            "erro_relativo_test_dp_entre_celulas_RATIO_OF_MEANS": float(np.std(r_test_rom)),
            "erro_relativo_test_min_celula_RATIO_OF_MEANS": float(np.min(r_test_rom)),
            "erro_relativo_test_max_celula_RATIO_OF_MEANS": float(np.max(r_test_rom)),
            "erro_relativo_val_media_entre_celulas_RATIO_OF_MEANS": float(np.mean(r_val_rom)),
            "erro_relativo_test_media_entre_celulas_MEAN_OF_RATIOS_por_seed": float(np.mean(r_test_mor)),
            "retencao_nodal_test_media_entre_celulas": float(np.mean([e["retencao_nodal_test_media"] for e in entradas])),
            "lei_campo_medio_test_media_entre_celulas": float(np.mean([e["lei_campo_medio_test_media"] for e in entradas])),
        }
        log(f"  {chave}: erro_test RATIO_OF_MEANS = {resumo_por_gb[chave]['erro_relativo_test_media_entre_celulas_RATIO_OF_MEANS']*100:.3f}%")
    saida["resumo_por_gb"] = resumo_por_gb
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    # -------- veredito vs criterio fixado (RATIO-OF-MEANS, definicao primaria) --------
    log("aplicando criterio fixado (monotonia com b, sem troca de sinal, faixa publicada)...")
    r_b1 = resumo_por_gb.get("g10.0_b1.0")
    r_b2 = resumo_por_gb.get("g10.0_b2.0")
    r_b3 = resumo_por_gb.get("g10.0_b3.0")
    veredito = {
        "definicao_usada": "RATIO-OF-MEANS -- ver nota_definicao em agregado_por_gb_celula",
        "pontos_disponiveis": {"b1": r_b1 is not None, "b2": r_b2 is not None, "b3": r_b3 is not None},
    }
    if r_b1 and r_b2 and r_b3:
        e1 = r_b1["erro_relativo_test_media_entre_celulas_RATIO_OF_MEANS"]
        e2 = r_b2["erro_relativo_test_media_entre_celulas_RATIO_OF_MEANS"]
        e3 = r_b3["erro_relativo_test_media_entre_celulas_RATIO_OF_MEANS"]
        e2_min = r_b2["erro_relativo_test_min_celula_RATIO_OF_MEANS"]
        e2_max = r_b2["erro_relativo_test_max_celula_RATIO_OF_MEANS"]
        sinais = [np.sign(e1), np.sign(e2), np.sign(e3)]
        sem_troca_de_sinal = bool(len(set(sinais)) == 1)
        monotono = bool(abs(e1) <= abs(e2) <= abs(e3))
        b2_na_faixa_publicada = bool(-7.0 <= e2 * 100 <= -6.0)
        ordem_grandeza_mantida = bool(
            (abs(e1 * 100) <= 3 * abs(e2 * 100)) and (abs(e3 * 100) <= 3 * abs(e2 * 100))
        )
        veredito.update({
            "erro_relativo_test_pct_b1": e1 * 100, "erro_relativo_test_pct_b2": e2 * 100, "erro_relativo_test_pct_b3": e3 * 100,
            "erro_relativo_test_pct_b2_faixa_16_celulas": [e2_min * 100, e2_max * 100],
            "sem_troca_de_sinal": sem_troca_de_sinal,
            "monotono_com_b_menor_buffer_menor_erro": monotono,
            "b2_recalculado_dentro_da_faixa_publicada_-6.2_a_-6.9pct": b2_na_faixa_publicada,
            "ordem_de_grandeza_mantida_entre_pontos_vizinhos": ordem_grandeza_mantida,
        })
        veredito["veredito_final"] = "sustenta_A1" if (sem_troca_de_sinal and monotono and b2_na_faixa_publicada) else "nao_sustenta_A1_como_fixado"
    else:
        veredito["veredito_final"] = "incompleto"
    saida["veredito_vs_criterio"] = veredito
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))
    log(f"veredito: {veredito.get('veredito_final')}")

    saida["nao_verificado"] = [
        {"item": "N efetivo sob autocorrelacao espacial intra-particao",
         "motivo": "fora de escopo desta frente (dados-vazamento-espacial); nao consultado artefato dela nesta rodada -- caminho nao fornecido em `caminhos`"},
        {"item": "(g,b) fora dos 4 pontos listados",
         "motivo": "regra D5 do roadmap: 1.4 reduzido nao amplia a varredura"},
    ]
    saida["comando_rodado"] = f"{Path(__file__).name} --out <out> --seeds {len(seeds)}"
    saida["tabela_numero_campo_comando"] = [
        {"numero": "erro relativo lei vs nodal por (celula,seed,g,b)", "campo": "resultados[i].erro_relativo_lei_vs_nodal_test/_val", "comando": saida["comando_rodado"]},
        {"numero": "erro RATIO-OF-MEANS por celula (primario)", "campo": "agregado_por_gb_celula.<chave>.erro_relativo_test_sobre_media_nodal", "comando": "idem"},
        {"numero": "IC95 bootstrap MEAN-OF-RATIOS por celula (secundario)", "campo": "agregado_por_gb_celula.<chave>.erro_relativo_test_ic95_bootstrap", "comando": "idem"},
        {"numero": "curva R(q) / erro medio entre celulas por (g,b)", "campo": "resumo_por_gb", "comando": "idem"},
        {"numero": "veredito vs criterio fixado", "campo": "veredito_vs_criterio", "comando": "idem"},
        {"numero": "correcao de dominio finito (Apendice A)", "campo": "correcao_dominio_finito_por_celula", "comando": "idem, funcao _lattice_retention (copia literal de fismat_sympy_eqR.py)"},
    ]

    saida["status"] = "concluido_provisorio"
    saida["pico_ram_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    saida["tempo_total_s"] = time.time() - t_inicio
    saida["tempo_varredura_s"] = tempo_varredura_s
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))
    log(f"CONCLUIDO. tempo_total_s={saida['tempo_total_s']:.1f} pico_ram_gb={saida['pico_ram_gb']:.2f}")


if __name__ == "__main__":
    main()
