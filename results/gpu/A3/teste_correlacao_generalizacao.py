#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu/A3/teste_correlacao_generalizacao.py -- verifica se o achado de
correlacao quase perfeita (feature roughness/flow_acc x alvo canal 2
path_loss_terrain, corr>0.98) encontrado em bauru Q3 (dataset cftudo)
generaliza para outra celula do lote A3 (lins Q1), e mede sobre TODA a
particao de treino (nao so 1 batch de 8192), para descartar ruido de
amostra pequena. So carrega dado e computa correlacao -- SEM treino.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODELO_V3_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/"
                      "_v3_2026-09-25/gpu/modelo_v3")
sys.path.insert(0, str(MODELO_V3_DIR))
import v3_common as v3  # noqa: E402

FEATURE_NAMES = ["elevation", "slope", "aspect", "curvature", "tpi", "tri",
                 "roughness", "flow_acc", "B02", "B03", "B04", "B08",
                 "NDVI", "NDWI", "BSI", "shadow_idx", "has_lidar", "dist_ant",
                 "dist_ant_zscored_extra"]
TARGET_NAMES = ["path_loss_total", "path_loss_vegetation", "path_loss_terrain",
                "rssi", "coverage_prob"]


def correlacao_full(x: torch.Tensor, y: torch.Tensor, limite: float = 0.98) -> dict:
    x = x.float(); y = y.float()
    xc = x - x.mean(dim=0, keepdim=True); yc = y - y.mean(dim=0, keepdim=True)
    xs = xc.std(dim=0, keepdim=True).clamp(min=1e-8)
    ys = yc.std(dim=0, keepdim=True).clamp(min=1e-8)
    xn = xc / xs; yn = yc / ys
    n = x.shape[0]
    corr = (xn.T @ yn) / n
    corr_np = corr.detach().cpu().numpy()
    achados = []
    for i in range(corr_np.shape[0]):
        for j in range(corr_np.shape[1]):
            v = float(corr_np[i, j])
            if abs(v) > limite:
                fname = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"col{i}"
                tname = TARGET_NAMES[j] if j < len(TARGET_NAMES) else f"col{j}"
                achados.append({"feature_col": i, "feature_nome": fname,
                                 "target_col": j, "target_nome": tname, "corr": v})
    return {"corr_max_abs": float(np.nanmax(np.abs(corr_np))),
            "achados_corr_gt_%.2f" % limite: achados, "n": int(n)}


def main():
    cidade, quadrante = sys.argv[1], sys.argv[2]
    saida = Path(sys.argv[3])
    t0 = time.perf_counter()
    rf_data_file = f"transfer_dataset_{cidade}_v19_{quadrante}_enriched_cftudo.pt"
    graph_file = f"{cidade}_v19_{quadrante}_gpu.pt"
    graph_dir = v3.GRAPH_DIR_DEFAULT

    mod = v3.carregar_modulo_congelado(v3.FROZEN_GNN_SCRIPT, f"train_gnn_c0_spatial_corrcheck_{cidade}{quadrante}")
    ctx = v3.carregar_base_e_particoes(
        mod=mod, graph_dir=graph_dir, rf_data_file=rf_data_file, graph_file=graph_file,
        max_nodes=100000, window_anchor="cobertura",
        grid_km=5.0, buffer_km=2.0, split_frac=(0.70, 0.15, 0.15),
        split_seed=42, smoke_geometria=True, mmap=True,
        precisa_arestas_ter_ter=False,
        log=lambda m: print(f"[+{time.perf_counter()-t0:.1f}s] {m}", flush=True),
    )
    tr_loc = torch.from_numpy(ctx.parts_local["train"])
    x_train = ctx.x_full[tr_loc]
    y_train = ctx.base["terrain"].rf_targets[tr_loc]
    res = correlacao_full(x_train, y_train)
    res["cidade"] = cidade
    res["quadrante"] = quadrante
    res["dataset"] = rf_data_file
    res["populacao"] = "particao TREINO INTEIRA (nao 1 batch)"
    res["tempo_total_s"] = time.perf_counter() - t0

    with open(saida, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"gravado {saida}: corr_max_abs={res['corr_max_abs']:.4f} n_achados={len(res['achados_corr_gt_0.98'])}")


if __name__ == "__main__":
    sys.exit(main() or 0)
