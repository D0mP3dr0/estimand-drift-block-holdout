#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""e3_agregar_v2.py — E3 com várias seeds (Adendo 2 do pré-registro, 25/09 08:15).
Lê <braco>_<cidade>_<Q>_s<seed>.json em _R2_2026-09-24/e3_predicoes/ para todas as seeds presentes.
Critérios (copiados do Adendo 2, fixados antes do lote 2):
 (a) H1 ROBUSTA se mediana_celulas( media_seeds |Δ_sentinela| ) ≤ 0,05 dB E H1 (do pré-registro) vale em ≥ 4 das 5 seeds isoladamente;
 (b) célula com PL fora do limiar C3 e RSSI dentro: mantida e marcada;  (c) RSSI fora: descartada;
 (d) H2 usa a referência literal do R1: sinal de mae_pl_db por cidade em E1_sinal_por_populacao.json (por_cidade_n4.sinal_cobertos).
Sensibilidade: H1 por seed com e sem as células marcadas por (b). Publicáveis (regra da eng-IA aceita): π, MAE na sentinela, razão GNN/MLP na sentinela, contagens.
"""
import json, hashlib, statistics as st, glob, re
from pathlib import Path
AQUI = Path(__file__).resolve().parent.parent
PRED = AQUI / "_R2_2026-09-24" / "e3_predicoes"
FIO = Path("/trabalho/HERMES/AGENTES/_FIOS/2026-09-24_gnn_rf_artigo2_mathematics_r2/artefatos")
CIDADES = ["bauru", "campinas", "lins", "sorocaba"]; QS = ["Q1", "Q2", "Q3", "Q4"]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
seeds = sorted({int(m.group(1)) for f in PRED.glob("gnn_*_s*.json") for m in [re.search(r"_s(\d+)\.json$", f.name)] if m})
E1 = json.load(open(FIO / "E1_sinal_por_populacao.json"))
ref_pl = {c: E1["por_cidade_n4"][c]["sinal_cobertos"] for c in CIDADES}   # (d)
def carga(b, c, q, s):
    p = PRED / f"{b}_{c}_{q}_s{s}.json"
    return json.load(open(p)) if p.exists() else None
linhas = []; por_seed = {}
for s in seeds:
    cel = []
    for c in CIDADES:
        for q in QS:
            m, g = carga("mlp", c, q, s), carga("gnn", c, q, s)
            if not (m and g): continue
            assert m["particoes"]["test"]["idx_sha256_global_recomputado"] == g["particoes"]["test"]["idx_sha256_global_recomputado"]
            t_m, t_g = m["testemunha_contra_run_json"], g["testemunha_contra_run_json"]
            rssi_ok = t_m["mae_rssi_db_criterio_1e-3_ok"] and t_g["mae_rssi_db_criterio_1e-3_ok"]
            pl_ok = t_m["mae_pl_db_criterio_1e-3_ok"] and t_g["mae_pl_db_criterio_1e-3_ok"]
            if not rssi_ok: estado = "descartada_rssi_fora"          # (c)
            elif not pl_ok: estado = "mantida_marcada_pl_fora"      # (b)
            else: estado = "ok"
            sm, sg = m["populacoes"]["sentinela"]["mae_rssi_db"], g["populacoes"]["sentinela"]["mae_rssi_db"]
            cm, cg = m["populacoes"]["cobertos"]["mae_rssi_db"], g["populacoes"]["cobertos"]["mae_rssi_db"]
            row = {"seed": s, "celula": f"{c}_{q}", "cidade": c, "Q": q, "estado": estado,
                   "pi": g["populacoes"]["sentinela"]["frac_pi_sentinela"],
                   "mae_sent_mlp": sm, "mae_sent_gnn": sg, "delta_sent": sg - sm, "razao_sent_gnn_mlp": sg / sm if sm else None,
                   "delta_cob_rssi": cg - cm, "delta_cob_pl": g["populacoes"]["cobertos"]["mae_pl_db"] - m["populacoes"]["cobertos"]["mae_pl_db"],
                   "_fontes": {"mlp": sha(PRED / f"mlp_{c}_{q}_s{s}.json"), "gnn": sha(PRED / f"gnn_{c}_{q}_s{s}.json")}}
            cel.append(row); linhas.append(row)
    usa = [r for r in cel if r["estado"] != "descartada_rssi_fora"]
    sem = [r for r in cel if r["estado"] == "ok"]
    def h1(rs):
        if len(rs) < 8: return None
        ma = st.median(abs(r["delta_sent"]) for r in rs); mm = st.median(r["mae_sent_mlp"] for r in rs); mg = st.median(r["mae_sent_gnn"] for r in rs)
        return {"n": len(rs), "med_abs_delta": ma, "med_mlp": mm, "med_gnn": mg, "sustentada": bool(ma <= 0.05 and mm <= 0.25 and mg <= 0.25)}
    # H2 (d): sinal por cidade de delta_cob_rssi (média das Q) vs referência literal do R1
    h2 = {}
    for c in CIDADES:
        rs = [r for r in usa if r["cidade"] == c]
        if not rs: continue
        d = st.mean(r["delta_cob_rssi"] for r in rs); h2[c] = {"delta_cob_rssi": d, "sinal": -1 if d < 0 else 1, "ref_R1": ref_pl[c], "coincide": (-1 if d < 0 else 1) == ref_pl[c]}
    por_seed[s] = {"n_celulas": len(cel), "marcadas_pl": [r["celula"] for r in cel if r["estado"] == "mantida_marcada_pl_fora"],
                   "descartadas": [r["celula"] for r in cel if r["estado"] == "descartada_rssi_fora"],
                   "H1_com_marcadas": h1(usa), "H1_sem_marcadas": h1(sem),
                   "H2": {"coincidentes": sum(v["coincide"] for v in h2.values()), "de": len(h2), "sustentada": bool(sum(v["coincide"] for v in h2.values()) >= 3), "por_cidade": h2},
                   "contagens_sentinela": {"celulas_gnn_menor": sum(r["delta_sent"] < 0 for r in usa), "de": len(usa),
                                            "cidades_gnn_menor": sum(st.mean(r["delta_sent"] for r in usa if r["cidade"] == c) < 0 for c in CIDADES if any(r["cidade"] == c for r in usa))}}
# (a) robustez sobre as seeds presentes
por_cel = {}
for r in linhas:
    if r["estado"] == "descartada_rssi_fora": continue
    por_cel.setdefault(r["celula"], []).append(abs(r["delta_sent"]))
med_media = st.median(st.mean(v) for v in por_cel.values()) if por_cel else None
h1_por_seed = [por_seed[s]["H1_com_marcadas"]["sustentada"] for s in seeds if por_seed[s]["H1_com_marcadas"]]
robusta = (med_media is not None and med_media <= 0.05 and sum(h1_por_seed) >= 4) if len(seeds) >= 5 else None
pub = {"pi_por_celula_seed42": {r["celula"]: r["pi"] for r in linhas if r["seed"] == 42},
       "mae_sentinela_max": {"mlp": max(r["mae_sent_mlp"] for r in linhas), "gnn": max(r["mae_sent_gnn"] for r in linhas)},
       "razao_sent_gnn_mlp_mediana_todas_seeds": st.median(r["razao_sent_gnn_mlp"] for r in linhas if r["razao_sent_gnn_mlp"]),
       "celulas_seed42_gnn_menor_sentinela": sum(r["delta_sent"] < 0 for r in linhas if r["seed"] == 42)}
out = {"experimento": "E3 multi-seed (Adendo 2)", "seeds_presentes": seeds, "n_linhas": len(linhas), "referencia_H2_R1": ref_pl,
       "robustez_a": {"mediana_celulas_da_media_seeds_abs_delta": med_media, "H1_valida_em_seeds": sum(h1_por_seed), "de": len(h1_por_seed), "robusta": robusta,
                      "nota": "só decidível com 5 seeds; None = lote incompleto"},
       "por_seed": por_seed, "publicaveis": pub, "script_sha256": sha(__file__), "status": "provisorio_ate_tres_votos"}
(FIO / "E3_agregado_v2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
(FIO / "E3_linhas_v2.json").write_text(json.dumps(linhas, ensure_ascii=False, indent=1))
print(json.dumps({"seeds": seeds, "robustez_a": out["robustez_a"], "por_seed": {s: {k: v for k, v in por_seed[s].items() if k != "H2"} | {"H2": {"coincidentes": por_seed[s]["H2"]["coincidentes"], "sustentada": por_seed[s]["H2"]["sustentada"]}} for s in seeds}, "publicaveis": pub}, ensure_ascii=False, indent=1, default=str)[:3500])
