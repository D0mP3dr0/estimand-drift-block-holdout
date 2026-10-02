"""G1/G2 do criterio criterios/criterio_resposta_cego_v3-10.json (adendo): comparacao justa entre
dp entre sorteios e dp entre celulas NUM SORTEIO FIXO, para a retencao nodal (G1) e para o erro dos
modelos treinados no lote A4 (G2). Saida: _v3_2026-09-25/fase3/3.8_resposta_cego_G1G2.json"""
import json, statistics as st
from pathlib import Path
B = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
def res(v):
    v = sorted(v); return {"n": len(v), "mediana": st.median(v), "min": v[0], "max": v[-1]}
d = json.load(open(B / "fase1/1.8_referencia_nodal_200seeds.json")); pc = d["por_celula"]
cels = list(pc.keys()); n = len(d["seeds"])
assert all(len(pc[c]["retencoes_te"]) == n for c in cels)
por_seed = [st.stdev([pc[c]["retencoes_te"][i] for c in cels]) for i in range(n)]
g1 = {"n_celulas": len(cels), "n_sorteios": n, "dp_entre_celulas_por_sorteio": res(por_seed),
      "dp_entre_sorteios_por_celula_media": st.mean(st.stdev(pc[c]["retencoes_te"]) for c in cels),
      "dp_entre_sorteios_por_celula_faixa": [min(st.stdev(pc[c]["retencoes_te"]) for c in cels), max(st.stdev(pc[c]["retencoes_te"]) for c in cels)],
      "dp_entre_medias_das_celulas": st.stdev(st.mean(pc[c]["retencoes_te"]) for c in cels)}
a = json.load(open(B / "gpu/A4/agregado_A4.json"))["celulas"]; g2 = {}
for mod in ("gnn", "mlp"):
    sorteios = sorted(a["bauru_Q1"]["por_modelo"][mod]["mae_por_sorteio"].keys())
    ps = [st.stdev([a[c]["por_modelo"][mod]["mae_por_sorteio"][s]["mae_rssi_validos_db"] for c in a]) for s in sorteios]
    g2[mod] = {"n_celulas": len(a), "n_sorteios": len(sorteios), "dp_entre_celulas_por_sorteio": res(ps),
               "dp_entre_sorteios_por_celula": {c: st.stdev([a[c]["por_modelo"][mod]["mae_por_sorteio"][s]["mae_rssi_validos_db"] for s in sorteios]) for c in a},
               "dp_entre_medias_das_celulas": st.stdev(st.mean(a[c]["por_modelo"][mod]["mae_por_sorteio"][s]["mae_rssi_validos_db"] for s in sorteios) for c in a)}
out = {"criterio": "criterios/criterio_resposta_cego_v3-10.json (adendo G1_G2)", "G1_retencao_teste": g1, "G2_modelos_A4_validos_dB": g2}
(B / "fase3/3.8_resposta_cego_G1G2.json").write_text(json.dumps(out, indent=1, ensure_ascii=False)); print(json.dumps(out, indent=1))
