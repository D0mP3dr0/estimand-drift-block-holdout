# -*- coding: utf-8 -*-
"""
Parecer MDPI Mathematics - bloco B3 (nucleo formal do candidato #1).

(A) LEI DE RETENCAO do split em blocos com buffer de 3 vias.
    Area sobrevivente EXATA de um bloco [0,g]^2 aparado contra um conjunto S
    de vizinhos (4 lados + 4 cantos), para b <= g/2:

      Area(S) = (g - b*(1_L+1_R)) * (g - b*(1_B+1_T))
                - (pi b^2/4) * #{cantos c em S cujos DOIS lados adjacentes
                                 NAO estao em S}

    rho(S) = Area(S)/g^2.  Retencao esperada = media de rho sobre os blocos da
    classe, com a atribuicao aleatoria de blocos (mesmo sorteio do codigo:
    permutacao uniforme dos blocos, contagens fixas -> hipergeometrica).

    Aproximacao de campo medio (vizinhos i.i.d. com prob. q de NAO aparar):
      R(g,b,q) = [ (g-2b)^2 + 4b(g-2b)q + 4b^2(1-pi/4)q^2 + pi b^2 q^3 ] / g^2

    Regra do codigo (train_gnn_c0_spatial.py:452-465):
      val  aparado contra TRAIN            -> q_val  = 1 - p_train
      test aparado contra TRAIN u VAL      -> q_test = p_test

(B) DERIVA DO ESTIMANDO: fracao de alvo nao-sentinela por particao.

Carimbo: 2026-09-11_geometria_split_mdpi.json em ATAS.
"""
import json, math, os, glob, random
import statistics as st

TREINOS = r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO\treinos"
OUT = r"D:\HERMES_AGENTE_PESQUISA\AGENTES\forum-fisico-matematico\ATAS\2026-09-11_geometria_split_mdpi.json"

PI4 = math.pi / 4.0
SIDES = [(-1, 0), (1, 0), (0, -1), (0, 1)]           # L R B T
CORNERS = {(-1, -1): [(-1, 0), (0, -1)], (1, -1): [(1, 0), (0, -1)],
           (-1, 1): [(-1, 0), (0, 1)], (1, 1): [(1, 0), (0, 1)]}


def rho_bloco(g, b, trim):
    """trim: dict (dx,dy)->bool, True se aquele vizinho APARA este bloco."""
    nx = (1 if trim.get((-1, 0)) else 0) + (1 if trim.get((1, 0)) else 0)
    ny = (1 if trim.get((0, -1)) else 0) + (1 if trim.get((0, 1)) else 0)
    area = (g - b * nx) * (g - b * ny)
    for c, (s1, s2) in CORNERS.items():
        if trim.get(c) and not trim.get(s1) and not trim.get(s2):
            area -= PI4 * b * b
    return max(area, 0.0) / (g * g)


def R_campo_medio(g, b, q):
    if b > g / 2.0:
        return None
    A = (g - 2 * b) ** 2
    B = 4.0 * b * (g - 2 * b) * q
    C = 4.0 * (b * b - PI4 * b * b) * q * q
    D = 4.0 * PI4 * b * b * (q ** 3)
    return (A + B + C + D) / (g * g)


def R_lattice_mc(g, b, m, n, k_tr, k_va, k_te, n_mc=4000, seed=12345):
    """Expectativa exata (ate erro de MC) sob o MESMO sorteio do codigo:
    permutacao uniforme dos m*n blocos -> primeiros k_tr = train, etc.
    Trata a borda do dominio (vizinho inexistente nunca apara)."""
    rng = random.Random(seed)
    cells = [(i, j) for i in range(m) for j in range(n)]
    N = len(cells)
    idx = {c: t for t, c in enumerate(cells)}
    acc_va, acc_te = [], []
    for _ in range(n_mc):
        perm = list(range(N))
        rng.shuffle(perm)
        lab = [None] * N
        for t, p in enumerate(perm):
            lab[p] = 0 if t < k_tr else (1 if t < k_tr + k_va else 2)
        sv, ste = [], []
        for c in cells:
            l = lab[idx[c]]
            if l == 0:
                continue
            trim = {}
            for d in SIDES + list(CORNERS.keys()):
                nb = (c[0] + d[0], c[1] + d[1])
                if nb not in idx:
                    trim[d] = False           # fora do dominio: nao apara
                else:
                    ln = lab[idx[nb]]
                    if l == 1:                # val aparado so contra train
                        trim[d] = (ln == 0)
                    else:                     # test aparado contra train u val
                        trim[d] = (ln in (0, 1))
            r = rho_bloco(g, b, trim)
            (sv if l == 1 else ste).append(r)
        acc_va.append(st.mean(sv))
        acc_te.append(st.mean(ste))
    return st.mean(acc_va), st.mean(acc_te), st.pstdev(acc_va), st.pstdev(acc_te)


# ---------------- leitura dos runs ----------------
runs = []
for d in sorted(os.listdir(TREINOS)):
    p = os.path.join(TREINOS, d)
    if not os.path.isdir(p):
        continue
    for f in glob.glob(os.path.join(p, "run_*.json")):
        j = json.load(open(f, encoding="utf-8"))
        si = j.get("split") or {}
        if not si or j.get("smoke"):
            continue
        par = j.get("particoes") or {}
        geo = j.get("geometria") or {}
        nb = si.get("n_blocos") or {}
        toks = d.split("_")
        runs.append({
            "run": d, "cidade": toks[1],
            "quadrante": next((t for t in toks if t.startswith("Q")), None),
            "seed": j.get("seed"),
            "g": si.get("grid_km"), "b": si.get("buffer_km"),
            "nbt": si.get("n_blocos_total"),
            "k_tr": nb.get("train"), "k_va": nb.get("val"), "k_te": nb.get("test"),
            "ret_val": (si.get("retencao_apos_buffer") or {}).get("val"),
            "ret_test": (si.get("retencao_apos_buffer") or {}).get("test"),
            "dmin": {k: v["dist_min_km"] for k, v in
                     ((si.get("verificacao") or {}).get("pares") or {}).items()},
            "ext_x": geo.get("extensao_x_km"), "ext_y": geo.get("extensao_y_km"),
            "fv_tr": (par.get("train") or {}).get("frac_pl_alvo_valido"),
            "fv_va": (par.get("val") or {}).get("frac_pl_alvo_valido"),
            "fv_te": (par.get("test") or {}).get("frac_pl_alvo_valido"),
            "n_tr": (par.get("train") or {}).get("n"),
            "n_va": (par.get("val") or {}).get("n"),
            "n_te": (par.get("test") or {}).get("n"),
        })

print("runs (nao-smoke) com bloco 'split':", len(runs))

# celula unica = (cidade, quadrante, g, b); split_seed fixo -> identico entre seeds
cel = {}
for r in runs:
    cel.setdefault((r["cidade"], r["quadrante"], r["g"], r["b"]), r)
print("celulas unicas (cidade,Q,g,b):", len(cel))

tabela = []
for (ci, q, g, b), r in sorted(cel.items()):
    m = max(1, int(round(r["ext_x"] / g)))
    n = max(1, int(round(r["ext_y"] / g)))
    # ajusta m,n para bater com n_blocos_total observado quando possivel
    alvo = r["nbt"]
    melhor = (m, n, abs(m * n - alvo))
    for mm in range(max(1, m - 2), m + 3):
        for nn in range(max(1, n - 2), n + 3):
            if abs(mm * nn - alvo) < melhor[2]:
                melhor = (mm, nn, abs(mm * nn - alvo))
    m, n = melhor[0], melhor[1]
    p_tr = r["k_tr"] / r["nbt"]; p_va = r["k_va"] / r["nbt"]; p_te = r["k_te"] / r["nbt"]
    cm_val = R_campo_medio(g, b, 1.0 - p_tr)
    cm_te = R_campo_medio(g, b, p_te)
    mc_va, mc_te, sd_va, sd_te = R_lattice_mc(g, b, m, n, r["k_tr"], r["k_va"], r["k_te"],
                                              n_mc=1500)
    tabela.append({
        "celula": f"{ci}_{q}_g{int(g)}b{int(b)}", "g_km": g, "b_km": b,
        "ext_km": [r["ext_x"], r["ext_y"]], "lattice_mn": [m, n],
        "n_blocos_total": r["nbt"], "n_blocos": [r["k_tr"], r["k_va"], r["k_te"]],
        "p_blocos": [p_tr, p_va, p_te],
        "R_min_q0": R_campo_medio(g, b, 0.0),
        "R_campo_medio_val": cm_val, "R_campo_medio_test": cm_te,
        "R_lattice_mc_val": mc_va, "R_lattice_mc_test": mc_te,
        "R_obs_val": r["ret_val"], "R_obs_test": r["ret_test"],
        "err_rel_mc_val": (mc_va - r["ret_val"]) / r["ret_val"],
        "err_rel_mc_test": (mc_te - r["ret_test"]) / r["ret_test"],
        "err_rel_cm_val": (cm_val - r["ret_val"]) / r["ret_val"],
        "err_rel_cm_test": (cm_te - r["ret_test"]) / r["ret_test"],
        "dist_min_km_pares": r["dmin"],
        "frac_valid": {"train": r["fv_tr"], "val": r["fv_va"], "test": r["fv_te"]},
        "n_nos": {"train": r["n_tr"], "val": r["n_va"], "test": r["n_te"]},
    })

# resumo por configuracao
resumo = {}
for g, b in sorted({(t["g_km"], t["b_km"]) for t in tabela}):
    sub = [t for t in tabela if t["g_km"] == g and t["b_km"] == b]
    def agg(key):
        v = [t[key] for t in sub if t[key] is not None]
        return {"min": min(v), "med": st.median(v), "max": max(v), "media": st.mean(v)}
    resumo[f"g{int(g)}b{int(b)}"] = {
        "n_celulas": len(sub),
        "R_min_teorico_q0": sub[0]["R_min_q0"],
        "R_obs_val": agg("R_obs_val"), "R_obs_test": agg("R_obs_test"),
        "R_lattice_mc_val": agg("R_lattice_mc_val"), "R_lattice_mc_test": agg("R_lattice_mc_test"),
        "R_campo_medio_val": agg("R_campo_medio_val"), "R_campo_medio_test": agg("R_campo_medio_test"),
        "erro_rel_mc_val": agg("err_rel_mc_val"), "erro_rel_mc_test": agg("err_rel_mc_test"),
    }

# deriva do estimando (so g10b2, que e a linhagem publicavel)
deriva = {}
for t in tabela:
    if t["g_km"] != 10.0:
        continue
    fv, nn = t["frac_valid"], t["n_nos"]
    if fv["test"] is None:
        continue
    tot = nn["train"] + nn["val"] + nn["test"]
    dom = (fv["train"] * nn["train"] + fv["val"] * nn["val"] + fv["test"] * nn["test"]) / tot
    deriva[t["celula"]] = {
        "frac_valid_train": fv["train"], "frac_valid_val": fv["val"],
        "frac_valid_test": fv["test"], "frac_valid_dominio_retido": dom,
        "razao_test_sobre_dominio": fv["test"] / dom,
        "pi_sentinela_test": 1.0 - fv["test"],
        "cota_MAE_cob_sobre_MAE_all": 1.0 / fv["test"],
    }
raz = [v["razao_test_sobre_dominio"] for v in deriva.values()]

out = {
    "gerado_em": "2026-09-11",
    "autor": "forum-fisico-matematico",
    "script": os.path.abspath(__file__),
    "fonte": TREINOS + r"\<run>\run_*.json  (campos: split.*, particoes.*, geometria.*)",
    "lei_exata_bloco": "Area(S)=(g-b(1_L+1_R))(g-b(1_B+1_T)) - (pi b^2/4)*#{cantos em S com ambos os lados fora de S}; b<=g/2",
    "lei_campo_medio": "R(g,b,q)=[(g-2b)^2+4b(g-2b)q+4b^2(1-pi/4)q^2+pi b^2 q^3]/g^2",
    "regra_do_codigo": "val aparado so contra train (q=1-p_train); test aparado contra train u val (q=p_test) -- train_gnn_c0_spatial.py:452-465",
    "n_runs": len(runs), "n_celulas": len(tabela),
    "resumo_por_config": resumo,
    "tabela_por_celula": tabela,
    "deriva_do_estimando_g10b2": deriva,
    "resumo_deriva": {
        "razao_test_sobre_dominio": {"min": min(raz), "med": st.median(raz),
                                     "max": max(raz), "n": len(raz)},
        "frac_valid_test": {"min": min(v["frac_valid_test"] for v in deriva.values()),
                            "max": max(v["frac_valid_test"] for v in deriva.values())},
    },
}
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(resumo, indent=1, ensure_ascii=False))
print("---- deriva ----")
print(json.dumps(out["resumo_deriva"], indent=1, ensure_ascii=False))
print("gravado:", OUT)
