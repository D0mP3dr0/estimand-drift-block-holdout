#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""fismat_vies_ht.py -- forum-fisico-matematico, 2026-09-25 (D6/F2, entrega 3).
Decompoe o vies do protocolo contra o dominio, por celula/preditor/populacao, com os
campos de fase2/2.3_estimando_ht_baselines.json:
  vies total (b-a) = termo de desenho (c-a) = Cov_U(p,e)/pbar  +  termo de razao (b-c) = -Cov(Err,M|A)/E[M|A]
e mede se os vieses sao resolvidos pelo proprio Monte Carlo: SE(d) = d_HT_dp/sqrt(n_sorteios_d)
(campo do 2.3); SE(b) indicativo = dp_Err (2.1 parcial, 20 sorteios, via fismat_razao_cov.json,
so preditor constante) / sqrt(n_sorteios_b). So aritmetica sobre JSON.
"""
import hashlib, json, math
from pathlib import Path
R = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/fase2")
F23, FRC = R / "2.3_estimando_ht_baselines.json", R / "fismat_razao_cov.json"
d23, drc = json.load(open(F23)), json.load(open(FRC))
rows = []
for c, v in d23["por_celula"].items():
    for pred, w in v.items():
        for pop, x in w.items():
            if pop == "sentinela" and pred == "constante":
                continue  # e_i = 0 identicamente (constante = sentinela): sem conteudo
            se_d = x["d_HT_dp_dB"] / math.sqrt(x["n_sorteios_d"])
            r = {"celula": c, "preditor": pred, "populacao": pop, "a": x["a_mae_dominio_dB"], "b": x["b_media_sorteios_mae_teste_dB"],
                 "c": x["c_razao_esperancas_p_te_dB"], "d": x["d_HT_media_dB"],
                 "termo_desenho_c_menos_a": round(x["c_razao_esperancas_p_te_dB"] - x["a_mae_dominio_dB"], 3),
                 "termo_razao_b_menos_c": x["diferenca_b_menos_c_dB"], "vies_total_b_menos_a": x["vies_protocolo_b_menos_a_dB"],
                 "vies_HT_d_menos_a": x["vies_HT_d_menos_a_dB"], "dp_HT_por_sorteio": x["d_HT_dp_dB"],
                 "SE_media_d": round(se_d, 3), "vies_HT_em_SE": round(x["vies_HT_d_menos_a_dB"] / se_d, 2) if se_d > 0 else None,
                 "dp_HT_sobre_abs_vies_total": round(x["d_HT_dp_dB"] / abs(x["vies_protocolo_b_menos_a_dB"]), 1) if x["vies_protocolo_b_menos_a_dB"] else None}
            if pred == "constante" and c in drc["A_razao_aleatoria_constante_16x20"]:
                dp = drc["A_razao_aleatoria_constante_16x20"][c][pop]["dp_Err_dB"]
                r["SE_b_indicativo_2.1"] = round(dp / math.sqrt(x["n_sorteios_b"]), 3)
                r["vies_total_em_SE_b_indicativo"] = round(x["vies_protocolo_b_menos_a_dB"] / r["SE_b_indicativo_2.1"], 2)
            rows.append(r)
def faixa(k, **f):
    v = [r[k] for r in rows if all(r[kk] == vv for kk, vv in f.items()) and r.get(k) is not None]
    return [min(v), max(v)] if v else None
out = {"script": __file__, "sha256_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "insumos": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (F23, FRC)}, "linhas": rows, "faixas": {}}
for pred in ("constante", "fspl_calibrado_b"):
    for pop in ("validos", "sentinela", "todos"):
        if pred == "constante" and pop == "sentinela":
            continue
        out["faixas"][f"{pred}|{pop}"] = {k: faixa(k, preditor=pred, populacao=pop) for k in
            ("termo_desenho_c_menos_a", "termo_razao_b_menos_c", "vies_total_b_menos_a", "vies_HT_d_menos_a", "dp_HT_por_sorteio",
             "SE_media_d", "vies_HT_em_SE", "dp_HT_sobre_abs_vies_total", "SE_b_indicativo_2.1", "vies_total_em_SE_b_indicativo")}
json.dump(out, open(R / "fismat_vies_ht.json", "w"), indent=1)
print(json.dumps(out["faixas"], indent=1))
