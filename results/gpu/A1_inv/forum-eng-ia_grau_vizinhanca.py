"""forum-eng-ia 26/09 (seguimento, regra de avaliacao): distribuicao do grau de ENTRADA
de cada no de terreno por relacao (ET_TT: terreno->terreno; ET_AT: antena->terreno) nos
grafos induzidos train e test do dado do gate (Bauru Q2, 100k, split 42), fracao de nos com
grau > k (k_terrain=8, k_antenna=20: onde a amostragem sorteia) e razao de nos por semente
vizinhanca completa / amostrada (proxy de VRAM da avaliacao disjunta). CPU, sem modelo."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modelo_v3"))
import v3_common as v3
v3.patch_decoder_gnn()
mod = v3.carregar_modulo_congelado(v3.FROZEN_GNN_SCRIPT, "train_gnn_c0_spatial_grau")
ctx = v3.carregar_base_e_particoes(
    mod=mod, graph_dir=v3.GRAPH_DIR_DEFAULT,
    rf_data_file="transfer_dataset_bauru_v19_Q2_enriched_v2.pt", graph_file="bauru_v19_Q2_gpu.pt",
    max_nodes=100_000, window_anchor="cobertura", grid_km=5.0, buffer_km=2.0,
    split_frac=(0.70, 0.15, 0.15), split_seed=42, smoke_geometria=True, mmap=True,
    precisa_arestas_ter_ter=True, log=lambda m: None)
K = {"TT": 8, "AT": 20}
out = {"k": K, "particoes": {}}
def desc(deg, k):
    d = deg.numpy().astype(float)
    return {"media": d.mean(), "p50": float(np.percentile(d, 50)), "p90": float(np.percentile(d, 90)),
            "p99": float(np.percentile(d, 99)), "max": d.max(), "frac_grau_gt_k": float((d > k).mean()),
            "frac_grau_0": float((d == 0).mean()),
            "media_vizinhos_amostrados": float(np.minimum(d, k).mean()),
            "media_vizinhos_descartados_por_no": float(np.maximum(d - k, 0).mean())}
for part in ["train", "test"]:
    g, _ = mod.induzir_particao(ctx.base, torch.from_numpy(ctx.parts_local[part]), ctx.n_antenna,
                                lambda m: None, part)
    n = g["terrain"].num_nodes
    r = {"n_terreno": int(n)}
    for nome, et in [("TT", mod.ET_TT), ("AT", mod.ET_AT)]:
        ei = g[et].edge_index
        r[nome + "_autolacos_frac"] = float((ei[0] == ei[1]).float().mean()) if nome == "TT" else 0.0
        deg = torch.bincount(ei[1], minlength=n)
        r[nome] = desc(deg, K[nome])
    full = 1 + r["TT"]["media"] + r["AT"]["media"]
    samp = 1 + r["TT"]["media_vizinhos_amostrados"] + r["AT"]["media_vizinhos_amostrados"]
    r["nos_por_semente_completa"] = full; r["nos_por_semente_amostrada"] = samp
    r["razao_completa_sobre_amostrada"] = full / samp
    out["particoes"][part] = r
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
print(json.dumps(out, indent=1))
