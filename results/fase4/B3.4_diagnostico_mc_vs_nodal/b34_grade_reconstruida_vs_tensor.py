# B3.4 auxiliar: a grade reconstruida (build_synthetic_grid) coincide com as posicoes gravadas no tensor?
# Mede desvio em metros e quantos nos mudariam de bloco (g=10 e g=5) -- SO LEITURA.
import json, sys, importlib.util, numpy as np, torch
spec = importlib.util.spec_from_file_location("geo", "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts/varredura_split_geometria.py")
geo = importlib.util.module_from_spec(spec); spec.loader.exec_module(geo)
c11 = json.load(open("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/fase1/1.1_geo_celulas_16.json"))["por_celula"]
out = {}
for cid, q in (("bauru", "Q1"), ("lins", "Q3")):
    P = f"/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2/graph_data/transfer_dataset_{cid}_v19_{q}_enriched_v2_cftudo.pt"
    d = torch.load(P, weights_only=False, map_location="cpu"); pr = d["terrain"].pos.numpy().astype(np.float64); del d
    c = c11[f"{cid}_{q}"]
    lon, lat = geo.build_synthetic_grid(c["bbox_lon_min_deg"], c["bbox_lon_max_deg"], c["bbox_lat_min_deg"], c["bbox_lat_max_deg"])
    ps = np.stack([lon, lat], 1)
    # mesma ordem de nos? (linha = latitude decrescente, coluna = longitude crescente)
    dlon = (pr[:, 0] - ps[:, 0]) * 111000 * np.cos(np.radians(pr[:, 1])); dlat = (pr[:, 1] - ps[:, 1]) * 111000
    xr = geo.latlon_graus_para_metros(pr[:, 0], pr[:, 1]) / 1000; xs = geo.latlon_graus_para_metros(ps[:, 0], ps[:, 1]) / 1000
    e = {"mesma_ordem_desvio_max_m": float(np.max(np.hypot(dlon, dlat)))}
    for g in (10.0, 5.0):
        e[f"nos_com_bloco_diferente_g{int(g)}"] = int(np.sum(geo.assign_groups(xr, g) != geo.assign_groups(xs, g)))
    out[f"{cid}_{q}"] = e; print(cid, q, e, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
