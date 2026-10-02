#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""fismat_razao_cov.py -- forum-fisico-matematico, 2026-09-25 (D6/F2, prop:ratio).
Ilustra a identidade exata E[Err|M>0] = E[S]/E[M] - Cov(Err,M|M>0)/E[M|M>0]
(S = soma dos erros na subpopulacao retida, M = seu tamanho) com os sorteios
por celula gravados em fase2/_v3_2.1_3.1_parcial.json (preditor constante, 16 celulas
x 20 sorteios, split_seed 42,...). Para cada celula e populacao: E[M], CV(M), dp(Err),
media de Err (esperanca da razao), Sum S / Sum M (razao de esperancas, amostral),
termo -cov/E[M] e a cota de Cauchy-Schwarz dp(Err)*CV(M). Tambem recomputa, a partir de
1.6/1.8/1.5, a decomposicao da retencao nodal por classe k (so aritmetica sobre JSON).
"""
import hashlib, json
from pathlib import Path
import numpy as np

R = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
F21 = R / "fase2/_v3_2.1_3.1_parcial.json"
F16 = R / "fase2/1.6_p_inclusao_por_no.json"
F18 = R / "fase1/1.8_referencia_nodal_200seeds.json"
F15 = R / "fase1/1.5_retencao_exata_vs_mc.json"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
out = {"script": __file__, "sha256_script": sha(Path(__file__)),
       "insumos": {str(p): sha(p) for p in (F21, F16, F18, F15)}}

# ---- (A) razao aleatoria: identidade e cota, preditor constante ----
d = json.load(open(F21))
tab = {}
for c, v in d["celulas"].items():
    row = {}
    for pop in ("validos", "todos"):
        M, E = [], []
        for s in v["por_sorteio"]:
            if s.get("status") != "ok":
                continue
            m = s["n_pop_teste"][pop]
            if m > 0:
                M.append(m); E.append(s["mae_constante_teste"][pop])
        M = np.array(M, float); E = np.array(E, float); S = E * M
        cov = float(np.mean((E - E.mean()) * (M - M.mean())))  # normalizacao 1/n: identidade amostral exata
        row[pop] = {"n_sorteios_M_pos": int(M.size), "M_min": int(M.min()), "M_max": int(M.max()),
                    "CV_M": float(M.std() / M.mean()), "dp_Err_dB": float(E.std()),
                    "esperanca_da_razao_dB": float(E.mean()), "razao_de_esperancas_dB": float(S.sum() / M.sum()),
                    "dif_b_menos_c_dB": float(E.mean() - S.sum() / M.sum()), "menos_cov_sobre_EM_dB": float(-cov / M.mean()),
                    "cota_CS_dpErr_x_CVM_dB": float(E.std() * M.std() / M.mean()),
                    "corr_Err_M": float(np.corrcoef(E, M)[0, 1])}
        assert abs(row[pop]["dif_b_menos_c_dB"] - row[pop]["menos_cov_sobre_EM_dB"]) < 1e-9
    tab[c] = row
out["A_razao_aleatoria_constante_16x20"] = tab
V = [tab[c]["validos"] for c in tab]; T = [tab[c]["todos"] for c in tab]
out["A_resumo"] = {k: {"validos": [min(x[k] for x in V), max(x[k] for x in V)], "todos": [min(x[k] for x in T), max(x[k] for x in T)]}
                   for k in ("CV_M", "dp_Err_dB", "dif_b_menos_c_dB", "cota_CS_dpErr_x_CVM_dB", "corr_Err_M")}
out["A_resumo"]["M_validos_min_max_global"] = [min(x["M_min"] for x in V), max(x["M_max"] for x in V)]

# ---- (B) decomposicao da retencao nodal por classe k (bauru..sorocaba Q1) ----
d16 = json.load(open(F16)); d18 = json.load(open(F18)); d15 = json.load(open(F15))
from math import perm
N, kte = 132, 20
ff = {k: perm(kte - 1, k) / perm(N - 1, k) for k in (0, 1, 2, 3)}
dec = {}
for c, v in d16["por_celula"].items():
    pk = v["perfil_por_classe_k"]
    f = {k: pk[str(k)]["frac_nos"] for k in (0, 1, 2)}
    emp = {k: pk[str(k)]["media_p_cond_te"] for k in (0, 1, 2)}
    lateral = sum(f[k] * ff[k] for k in (0, 1, 2))
    # canto: fracao pi/4 dos nos k=2 esta a <b do vertice diagonal -> precisa de 3 blocos de teste
    canto = f[0] * ff[0] + f[1] * ff[1] + f[2] * ((1 - np.pi / 4) * ff[2] + (np.pi / 4) * ff[3])
    nodal = v["media_p_cond_te_geral"]
    dec[c] = {"media_nodal_1.6": nodal, "forma_fechada_lateral": lateral, "forma_fechada_com_canto": float(canto),
              "excesso_total": nodal - float(canto), "contrib_k1_excesso_empirico_menos_ff": f[1] * (emp[1] - ff[1]),
              "contrib_k2_excesso_vs_canto": f[2] * (emp[2] - ((1 - np.pi / 4) * ff[2] + (np.pi / 4) * ff[3])),
              "fracao_do_excesso_explicada_por_k1": f[1] * (emp[1] - ff[1]) / (nodal - float(canto)),
              "k2_previsto_com_canto": float((1 - np.pi / 4) * ff[2] + (np.pi / 4) * ff[3]), "k2_empirico": emp[2]}
    r18 = d18["por_celula"][c]
    dec[c].update({"nodal_te_1.8_200seeds": r18["nodal_te_media"], "lei_te_1.8": r18["lei_te"],
                   "erro_lei_1.8": r18["erro_lei_vs_nodal_te"],
                   "excesso_k1_relativo_ao_nodal_1.8": f[1] * (emp[1] - ff[1]) / r18["nodal_te_media"]})
    k15 = [k for k in d15["por_celula"] if k.startswith(c) and k.endswith("g10b2")]
    if k15:
        e = d15["por_celula"][k15[0]]
        n18 = r18["nodal_te_media"]
        dec[c].update({"1.5_campo_medio_reticulado_te": e["campo_medio"]["teste"], "1.5_exato_reticulado_te": e["exato_sem_reposicao"]["teste"],
                       "1.5_nodal_referencia_20seeds": e["nodal_real_20seeds"]["retencao_test_media"],
                       "erro_rel_campo_medio_vs_1.8": (e["campo_medio"]["teste"] - n18) / n18,
                       "erro_rel_exato_vs_1.8": (e["exato_sem_reposicao"]["teste"] - n18) / n18})
out["B_decomposicao_retencao_por_k"] = dec
out["forma_fechada"] = ff
json.dump(out, open(R / "fase2/fismat_razao_cov.json", "w"), indent=1)
print(json.dumps(out["A_resumo"], indent=1))
for c, x in dec.items():
    print(c, {k: round(v, 5) for k, v in x.items()})
