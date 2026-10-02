# -*- coding: utf-8 -*-
"""
Segundo canal da deriva do estimando: ESTRUTURA DO GRAFO por particao.
Grau terrain->terrain e densidade de arestas antena->terrain em train/val/test,
para as 16 celulas g10b2 (seed base s42). Fonte: run_*.json -> subgrafos.*

Carimbo: 2026-09-11_deriva_grafo_mdpi.json em ATAS.
"""
import json, os, glob
import statistics as st

TREINOS = r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF\gnn_rf_ieee_access\FIRST_RESPONSE_REVIEW_IEEE_ACESSES\EVIDENCIA_RESUBMISSAO\treinos"
OUT = r"D:\HERMES_AGENTE_PESQUISA\AGENTES\forum-fisico-matematico\ATAS\2026-09-11_deriva_grafo_mdpi.json"

cel = {}
for d in sorted(os.listdir(TREINOS)):
    p = os.path.join(TREINOS, d)
    if not os.path.isdir(p) or "g10b2" not in d or not d.startswith("c0c1cf_"):
        continue
    for f in glob.glob(os.path.join(p, "run_*.json")):
        j = json.load(open(f, encoding="utf-8"))
        if j.get("smoke"):
            continue
        sg = j.get("subgrafos") or {}
        if not sg or "train" not in sg:
            continue
        toks = d.split("_")
        key = (toks[1], next(t for t in toks if t.startswith("Q")))
        if key in cel:
            continue
        row = {}
        base_tt = sg["train"]["n_arestas_ter_ter_base"]
        base_at = sg["train"]["n_arestas_ant_ter_base"]
        n_base = None
        for part in ("train", "val", "test"):
            s = sg[part]
            row[part] = {
                "n_terrain": s["n_terrain"],
                "grau_tt": s["n_arestas_ter_ter"] / s["n_terrain"],
                "dens_at": s["n_arestas_ant_ter"] / s["n_terrain"],
                "frac_nos_da_base": None,
                "cruzando_fronteira": s["arestas_cruzando_fronteira"],
            }
        n_base_nodes_guess = None
        row["_base"] = {"arestas_tt_base": base_tt, "arestas_at_base": base_at}
        row["razao_dens_at_test_sobre_train"] = row["test"]["dens_at"] / row["train"]["dens_at"]
        row["razao_dens_at_val_sobre_train"] = row["val"]["dens_at"] / row["train"]["dens_at"]
        row["razao_grau_tt_test_sobre_train"] = row["test"]["grau_tt"] / row["train"]["grau_tt"]
        cel[key] = row

r_at = [v["razao_dens_at_test_sobre_train"] for v in cel.values()]
r_atv = [v["razao_dens_at_val_sobre_train"] for v in cel.values()]
r_tt = [v["razao_grau_tt_test_sobre_train"] for v in cel.values()]

out = {
    "gerado_em": "2026-09-11", "script": os.path.abspath(__file__),
    "fonte": TREINOS + r"\c0c1cf_*_g10b2\run_*.json -> subgrafos.{train,val,test}",
    "n_celulas": len(cel),
    "definicoes": {
        "grau_tt": "n_arestas_ter_ter / n_terrain da particao (subgrafo INDUZIDO)",
        "dens_at": "n_arestas_ant_ter / n_terrain da particao",
        "nota_cruzando_fronteira": "0 por CONSTRUCAO (subgraph/bipartite_subgraph com "
                                   "relabel_nodes=True); a evidencia real do buffer e "
                                   "split.verificacao.pares[].dist_min_km >= buffer_km",
    },
    "resumo": {
        "razao_dens_at_test_sobre_train": {"min": min(r_at), "med": st.median(r_at),
                                           "max": max(r_at), "media": st.mean(r_at)},
        "razao_dens_at_val_sobre_train": {"min": min(r_atv), "med": st.median(r_atv),
                                          "max": max(r_atv), "media": st.mean(r_atv)},
        "razao_grau_tt_test_sobre_train": {"min": min(r_tt), "med": st.median(r_tt),
                                           "max": max(r_tt), "media": st.mean(r_tt)},
    },
    "por_celula": {f"{k[0]}_{k[1]}": v for k, v in sorted(cel.items())},
}
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for k, v in sorted(cel.items()):
    print("%-16s grau_tt tr %.4f va %.4f te %.4f | dens_at tr %.4f va %.4f te %.4f | at te/tr %.3f" % (
        k[0] + "_" + k[1], v["train"]["grau_tt"], v["val"]["grau_tt"], v["test"]["grau_tt"],
        v["train"]["dens_at"], v["val"]["dens_at"], v["test"]["dens_at"],
        v["razao_dens_at_test_sobre_train"]))
print(json.dumps(out["resumo"], indent=1))
print("gravado:", OUT)
