#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste G0 (e): vazamento de label por correlacao disfarcada.
Calcula correlacao de Pearson entre cada feature de entrada (17 features_raw
+ dist_nearest_m) e cada uma das 5 colunas de alvo (rf_targets), sobre uma
amostra do dataset REAL (Bauru Q1, v2, mmap — sem materializar os 13M nos
inteiros; le so uma fatia). Achado se |corr| > 0.98 sem explicacao fisica.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

RF_PATH = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2/graph_data/"
               "transfer_dataset_bauru_v19_Q1_enriched_v2.pt")
OUT = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/gpu/G0")

FEATURE_NAMES = [f"feat_{i}" for i in range(17)]
FEATURE_NAMES[0] = "elev"; FEATURE_NAMES[1] = "slope"; FEATURE_NAMES[6] = "rough"
FEATURE_NAMES[12] = "ndvi"; FEATURE_NAMES[13] = "ndwi"
TARGET_NAMES = ["path_loss_total", "path_loss_vegetation", "path_loss_terrain",
                "rssi", "coverage"]

t0 = time.time()
d = torch.load(RF_PATH, map_location="cpu", weights_only=False, mmap=True)
n_total = d["terrain"].y.shape[0]

rng = np.random.RandomState(42)
n_sample = 300_000
idx = np.sort(rng.choice(n_total, size=n_sample, replace=False))
idx_t = torch.from_numpy(idx)

feats = np.asarray(d["terrain"].features_raw[idx]).astype(np.float64)
dist = d["terrain"].dist_nearest_m[idx_t].numpy().astype(np.float64).reshape(-1, 1)
targets = d["terrain"].y[idx_t].numpy().astype(np.float64)

X = np.concatenate([feats, dist], axis=1)
feat_names_full = FEATURE_NAMES + ["dist_nearest_m"]

# alvo valido de PL (mesma definicao do script congelado: < 299 dB)
pl_valid = targets[:, 0] < 299.0

corr_table = {}
achados = []
for j, tname in enumerate(TARGET_NAMES):
    y = targets[:, j]
    mask = pl_valid if tname.startswith("path_loss") else np.ones_like(y, dtype=bool)
    yv = y[mask]
    if yv.std() < 1e-9:
        continue
    linha = {}
    for i, fname in enumerate(feat_names_full):
        xv = X[mask, i]
        if xv.std() < 1e-9:
            linha[fname] = None
            continue
        c = float(np.corrcoef(xv, yv)[0, 1])
        linha[fname] = None if np.isnan(c) else round(c, 4)
        if linha[fname] is not None and abs(linha[fname]) > 0.98:
            achados.append({"feature": fname, "target": tname, "corr": linha[fname]})
    corr_table[tname] = linha

resultado = {
    "teste": "vazamento_label_por_correlacao (e)",
    "comando": ("/trabalho/ambientes/s33_amb_virtual/.venv/bin/python "
                "teste_e_vazamento_correlacao.py"),
    "dataset": str(RF_PATH),
    "n_total_dataset": int(n_total),
    "n_amostra": n_sample,
    "seed_amostra": 42,
    "criterio": "|corr(feature, alvo)| > 0.98 sem explicacao fisica = achado",
    "tabela_correlacao": corr_table,
    "achados_acima_do_limiar": achados,
    "tempo_s": time.time() - t0,
    "passa": bool(len(achados) == 0),
}
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "teste_e_vazamento_correlacao.json", "w", encoding="utf-8") as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)
print(json.dumps(resultado, indent=2, ensure_ascii=False))
