#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""fismat_inclusao_decomp.py -- forum-fisico-matematico, 2026-09-25 (D6/F2, prop:inclusion).
Le fase2/fismat_k_corrigido_bauru.json (classe geometrica exata m, 100+100 seeds) e:
(1) decompoe o excesso k=1 do 1.6 em defeito de classe (vizinho fantasma), discretizacao
    da face (retidos com B treino/val) e flutuacao MC de P(B teste);
(2) avalia a formula de SEGUNDA ORDEM para m=2 lateral (canto fora do disco de raio b):
    p = [ (kte-1)(kte-2) + kva(kte-1)(I_L+I_B) q_D + kva(kva-1) I_L I_B q_D ] / ((N-1)(N-2)),
    q_D = k_tr/(N-3) (bloco diagonal D e treino), I_L = 1{dx^2+(dy-b)^2 > b^2},
    I_B = 1{(dx-b)^2+dy^2 > b^2} (todo no val do vizinho a <b do no esta a <b de D),
    media uniforme sobre a regiao {0<=dx,dy<b, dx^2+dy^2>=b^2} (Monte Carlo 4e6 pontos, seed fixa);
(3) cadeia lei (1.8) -> forma fechada nodal corrigida -> media nodal empirica -> retencao 1.8.
"""
import hashlib, json
from pathlib import Path
import numpy as np
R = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
FK = R / "fase2/fismat_k_corrigido_bauru.json"; F16 = R / "fase2/1.6_p_inclusao_por_no.json"; F18 = R / "fase1/1.8_referencia_nodal_200seeds.json"
dk, d16, d18 = json.load(open(FK)), json.load(open(F16)), json.load(open(F18))
N, kte, kva, ktr, b = 132, 20, 20, 92, 2.0
out = {"script": __file__, "sha256_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "insumos": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (FK, F16, F18)}}
# (1) m=1
m1 = dk["m1_retencao_por_papel_B"]
dec = {}
for lote in ("s100", "extra"):
    n = sum(m1[f"{lote}|papelB={k}"][0] for k in range(3))
    dec[lote] = {"n_no_sorteios_m1": n, "P_B_teste": m1[f"{lote}|papelB=2"][0] / n,
                 "retidos_com_B_treino_sobre_total": m1[f"{lote}|papelB=0"][1] / n,
                 "retidos_com_B_val_sobre_total": m1[f"{lote}|papelB=1"][1] / n,
                 "taxa_retencao_dado_B_treino": m1[f"{lote}|papelB=0"][1] / m1[f"{lote}|papelB=0"][0],
                 "taxa_retencao_dado_B_val": m1[f"{lote}|papelB=1"][1] / m1[f"{lote}|papelB=1"][0],
                 "p_m1_pooled": sum(m1[f"{lote}|papelB={k}"][1] for k in range(3)) / n}
ps = np.array([s["m1"][1] / s["m1"][0] for s in dk["por_seed"]])
out["m1"] = {"forma_fechada": (kte - 1) / (N - 1), "k16_1_media_nodal_1.6": d16["por_celula"]["bauru_Q1"]["perfil_por_classe_k"]["1"]["media_p_cond_te"],
             "k16_1_reproduzido": dk["reproducao_1.6_media_nodal_k16"]["1"], "m1_media_nodal_s100": dk["media_nodal_p_cond_te_por_m_s100"]["1"],
             "parcela_defeito_de_classe": dk["reproducao_1.6_media_nodal_k16"]["1"] - dk["media_nodal_k16_1_restrita_a_nao_fantasma"],
             "n_nos_fantasma": dk["n_nos_vizinho_fantasma_1.6"], "por_lote": dec,
             "p_m1_por_seed_media_200": float(ps.mean()), "p_m1_por_seed_dp_200": float(ps.std(ddof=1)),
             "p_m1_SE_200": float(ps.std(ddof=1) / np.sqrt(ps.size)), "cota_discretizacao_lmax_sobre_b": 30.910491943359375e-3 / b}
# (2) m=2 segunda ordem
rng = np.random.default_rng(20260925)
x = rng.uniform(0, b, 4_000_000); y = rng.uniform(0, b, 4_000_000)
reg = x**2 + y**2 >= b**2; x, y = x[reg], y[reg]
IL = (x**2 + (y - b)**2 > b**2).astype(float); IB = ((x - b)**2 + y**2 > b**2).astype(float)
qD = ktr / (N - 3)
p2 = ((kte - 1) * (kte - 2) + kva * (kte - 1) * (IL + IB) * qD + kva * (kva - 1) * IL * IB * qD) / ((N - 1) * (N - 2))
out["m2"] = {"forma_fechada_1a_ordem": (kte - 1) * (kte - 2) / ((N - 1) * (N - 2)), "previsto_2a_ordem_media_regiao": float(p2.mean()),
             "frac_regiao_IL": float(IL.mean()), "frac_regiao_IL_e_IB": float((IL * IB).mean()),
             "empirico_media_nodal_s100": dk["media_nodal_p_cond_te_por_m_s100"]["2"],
             "empirico_pooled_s100": dk["pooled_por_lote_e_m"]["s100|m=2"][1] / dk["pooled_por_lote_e_m"]["s100|m=2"][0],
             "empirico_pooled_extra": dk["pooled_por_lote_e_m"]["extra|m=2"][1] / dk["pooled_por_lote_e_m"]["extra|m=2"][0]}
out["m3"] = {"forma_fechada": (kte - 1) * (kte - 2) * (kte - 3) / ((N - 1) * (N - 2) * (N - 3)), "empirico_media_nodal_s100": dk["media_nodal_p_cond_te_por_m_s100"]["3"]}
# (3) cadeia
fr, ff = dk["frac_nos_por_m"], dk["forma_fechada_por_m"]
ff_nodal = sum(fr[j] * ff[j] for j in ("0", "1", "2", "3"))
com2a = fr["0"] + fr["1"] * ff["1"] + fr["2"] * out["m2"]["previsto_2a_ordem_media_regiao"] + fr["3"] * ff["3"]
r18 = d18["por_celula"]["bauru_Q1"]
out["cadeia_bauru_Q1"] = {"lei_eqR_1.8": r18["lei_te"], "forma_fechada_nodal_classe_corrigida": ff_nodal,
                          "forma_fechada_nodal_mais_2a_ordem_m2": com2a,
                          "media_nodal_empirica_s100": dk["media_nodal_geral_s100"], "retencao_nodal_1.8_200seeds": r18["nodal_te_media"],
                          "passo_lei_para_ff_rel_1.8": (ff_nodal - r18["lei_te"]) / r18["nodal_te_media"],
                          "passo_ff_para_empirico_rel_1.8": (dk["media_nodal_geral_s100"] - ff_nodal) / r18["nodal_te_media"],
                          "passo_empirico_para_1.8_rel": (r18["nodal_te_media"] - dk["media_nodal_geral_s100"]) / r18["nodal_te_media"],
                          "contrib_discretizacao_m1_rel_1.8": fr["1"] * (dk["media_nodal_p_cond_te_por_m_s100"]["1"] - ff["1"]) / r18["nodal_te_media"],
                          "contrib_2a_ordem_m2_rel_1.8": fr["2"] * (dk["media_nodal_p_cond_te_por_m_s100"]["2"] - ff["2"]) / r18["nodal_te_media"],
                          "erro_lei_1.8": r18["erro_lei_vs_nodal_te"]}
json.dump(out, open(R / "fase2/fismat_inclusao_decomp.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("m1", "m2", "m3", "cadeia_bauru_Q1")}, indent=1))
