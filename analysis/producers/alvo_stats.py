"""Bloco B4+B6: estatísticas do PRÓPRIO ALVO por quadrante (Bauru s42).

B4 (demanda D3 do físico): corr(RSSI_alvo, -dist_nearest) e
corr(RSSI_alvo, -log10 d) — monotonicidade medida no alvo, não argumentada.
B6 (demanda D4): fração de nós sentinela (path loss >= 299 dB) e fração
NDVI > 0,5 — sem esses dois números as métricas físicas não são
interpretáveis. Tudo por leitura dos *_enriched.pt; nenhum modelo envolvido.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
GRAPH = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2\graph_data")
OUT = BASE / "dados" / "alvo" / "alvo_stats_bauru_s42.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).mean() * (b * b).mean())
    return float((a * b).mean() / d) if d > 0 else float("nan")


res = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
       "definicoes": {
           "sentinela": "y[:,0] >= 299 dB (sem cobertura; RSSI imputado -110 dBm)",
           "veg": "NDVI (features_raw col 12) > 0.5",
           "corr": "Pearson; tambem reportada so sobre nos validos (nao-sentinela)"},
       "quadrantes": {}}

for q in ["Q1", "Q2", "Q3", "Q4"]:
    p = GRAPH / f"transfer_dataset_bauru_v19_{q}_enriched.pt"
    rf = torch.load(p, map_location="cpu", weights_only=False)
    y = rf["terrain"].y
    y = (torch.from_numpy(y) if isinstance(y, np.ndarray) else y).float().numpy()
    ndvi = rf["terrain"].features_raw
    ndvi = (torch.from_numpy(ndvi) if isinstance(ndvi, np.ndarray) else ndvi).float().numpy()[:, 12]
    d = rf["terrain"].dist_nearest_m.float().numpy()
    del rf
    pl, rssi = y[:, 0], y[:, 3]
    sent = pl >= 299.0
    val = ~sent
    res["quadrantes"][q] = {
        "n_nodes": int(len(pl)),
        "frac_sentinela": float(sent.mean()),
        "frac_ndvi_gt_0p5": float((ndvi > 0.5).mean()),
        "corr_rssi_alvo_vs_neg_dist_todos": corr(rssi, -d),
        "corr_rssi_alvo_vs_neg_log10d_todos": corr(rssi, -np.log10(np.maximum(d, 1.0))),
        "corr_rssi_alvo_vs_neg_dist_validos": corr(rssi[val], -d[val]),
        "corr_rssi_alvo_vs_neg_log10d_validos": corr(rssi[val], -np.log10(np.maximum(d[val], 1.0))),
        "corr_pl_alvo_vs_log10d_validos": corr(pl[val], np.log10(np.maximum(d[val], 1.0))),
    }
    print(q, res["quadrantes"][q])

OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print("[OK]", OUT)
