#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""e3_agregar.py — passo 4 do E3 (pré-registro, Adendo 1): H1/H2 sobre as 16 células seed 42.
Lê os JSON por célula/braço de _R2_2026-09-24/e3_predicoes/, grava E3_resultado_por_celula.json
e E3_agregado.json no fio. Critérios copiados do pré-registro (fixados antes de rodar):
  H1: nos sentinela, mediana_celulas(|MAE_gnn − MAE_mlp|) ≤ 0,05 dB e mediana de ambos ≤ 0,25 dB.
  H2: sinal por cidade de (GNN−MLP) nos cobertos (mae_rssi_db) coincide com o sinal de
      (GNN−MLP) em mae_pl_db (cobertos, PL) em ≥ 3 de 4 cidades.
Unidade de inferência: cidade (n=4); célula só descritivo. Tudo provisório até três votos.
"""
import json, hashlib, statistics as st, sys
from pathlib import Path
AQUI = Path(__file__).resolve().parent.parent
PRED = AQUI / "_R2_2026-09-24" / "e3_predicoes"
FIO = Path("/trabalho/HERMES/AGENTES/_FIOS/2026-09-24_gnn_rf_artigo2_mathematics_r2/artefatos")
CIDADES = ["bauru", "campinas", "lins", "sorocaba"]; QS = ["Q1", "Q2", "Q3", "Q4"]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
celulas = []
for c in CIDADES:
    for q in QS:
        m = json.load(open(PRED / f"mlp_{c}_{q}_s42.json")); g = json.load(open(PRED / f"gnn_{c}_{q}_s42.json"))
        assert m["particoes"]["test"]["idx_sha256_global_recomputado"] == g["particoes"]["test"]["idx_sha256_global_recomputado"], (c, q)
        assert m["verificacoes"]["rf_data_sha256_ok"] and g["verificacoes"]["rf_data_sha256_ok"], (c, q)
        row = {"celula": f"{c}_{q}", "cidade": c, "Q": q, "pi_sentinela": g["populacoes"]["sentinela"]["frac_pi_sentinela"],
               "n_test": g["populacoes"]["todos"]["n"]}
        for pop in ("todos", "cobertos", "sentinela"):
            row[f"mae_rssi_{pop}_mlp"] = m["populacoes"][pop]["mae_rssi_db"]; row[f"mae_rssi_{pop}_gnn"] = g["populacoes"][pop]["mae_rssi_db"]
            row[f"delta_rssi_{pop}"] = g["populacoes"][pop]["mae_rssi_db"] - m["populacoes"][pop]["mae_rssi_db"]
        row["mae_pl_cobertos_mlp"] = m["populacoes"]["cobertos"]["mae_pl_db"]; row["mae_pl_cobertos_gnn"] = g["populacoes"]["cobertos"]["mae_pl_db"]
        row["delta_pl_cobertos"] = row["mae_pl_cobertos_gnn"] - row["mae_pl_cobertos_mlp"]
        row["testemunha_ok"] = all([m["testemunha_contra_run_json"]["mae_rssi_db_criterio_1e-3_ok"], g["testemunha_contra_run_json"]["mae_rssi_db_criterio_1e-3_ok"],
                                    m["testemunha_contra_run_json"]["mae_pl_db_criterio_1e-3_ok"], g["testemunha_contra_run_json"]["mae_pl_db_criterio_1e-3_ok"]])
        row["_fontes"] = {"mlp": sha(PRED / f"mlp_{c}_{q}_s42.json"), "gnn": sha(PRED / f"gnn_{c}_{q}_s42.json")}
        celulas.append(row)
# H1
abs_delta_sent = [abs(r["delta_rssi_sentinela"]) for r in celulas]
med_abs = st.median(abs_delta_sent); med_mlp = st.median(r["mae_rssi_sentinela_mlp"] for r in celulas); med_gnn = st.median(r["mae_rssi_sentinela_gnn"] for r in celulas)
H1 = {"mediana_abs_delta_sentinela_db": med_abs, "mediana_mae_sentinela_mlp_db": med_mlp, "mediana_mae_sentinela_gnn_db": med_gnn,
      "criterio": "med|Δ| ≤ 0,05 e ambas medianas ≤ 0,25", "sustentada": bool(med_abs <= 0.05 and med_mlp <= 0.25 and med_gnn <= 0.25),
      "max_abs_delta_sentinela_db": max(abs_delta_sent), "celulas_abs_delta_gt_0_05": [r["celula"] for r in celulas if abs(r["delta_rssi_sentinela"]) > 0.05]}
# H2 por cidade
por_cidade = {}
for c in CIDADES:
    rs = [r for r in celulas if r["cidade"] == c]
    d_cob = st.mean(r["delta_rssi_cobertos"] for r in rs); d_pl = st.mean(r["delta_pl_cobertos"] for r in rs)
    d_tod = st.mean(r["delta_rssi_todos"] for r in rs); d_sen = st.mean(r["delta_rssi_sentinela"] for r in rs)
    por_cidade[c] = {"delta_rssi_todos": d_tod, "delta_rssi_cobertos": d_cob, "delta_rssi_sentinela": d_sen, "delta_pl_cobertos": d_pl,
                     "sinal_cobertos_rssi": "gnn_menor" if d_cob < 0 else "mlp_menor", "sinal_cobertos_pl": "gnn_menor" if d_pl < 0 else "mlp_menor",
                     "coincide": (d_cob < 0) == (d_pl < 0), "pi_medio": st.mean(r["pi_sentinela"] for r in rs)}
n_coinc = sum(v["coincide"] for v in por_cidade.values())
H2 = {"cidades_coincidentes": n_coinc, "criterio": "≥ 3 de 4", "sustentada": bool(n_coinc >= 3)}
# teste de sinal por cidade (n=4), descritivo
sinais = [por_cidade[c]["delta_rssi_cobertos"] < 0 for c in CIDADES]
agreg = {"experimento": "E3 inferência por população, 16 células seed 42, 2 braços", "protocolo": "E3_protocolo_preregistro.md (Adendo 1)",
         "script_agregador": str(Path(__file__).resolve()), "script_agregador_sha256": sha(__file__), "script_e3_sha256": celulas[0]["_fontes"] and json.load(open(PRED / "mlp_bauru_Q1_s42.json"))["script_sha256"],
         "n_celulas": len(celulas), "testemunhas_ok": sum(r["testemunha_ok"] for r in celulas), "H1": H1, "H2": H2, "por_cidade": por_cidade,
         "cidades_gnn_menor_cobertos": sum(sinais), "status": "provisorio_ate_tres_votos",
         "nota_nao_duplicacao": "MAE em 'todos' e 'cobertos' das corridas do T11 pertencem à IEEE; para o Artigo 2 entram a população sentinela, razões e contagens."}
(FIO / "E3_resultado_por_celula.json").write_text(json.dumps(celulas, ensure_ascii=False, indent=1))
(FIO / "E3_agregado.json").write_text(json.dumps(agreg, ensure_ascii=False, indent=1))
print(json.dumps({"H1": H1, "H2": H2, "por_cidade": {c: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} for c, d in por_cidade.items()}}, ensure_ascii=False, indent=1))
