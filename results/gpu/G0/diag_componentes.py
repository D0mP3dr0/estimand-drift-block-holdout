#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostico complementar: separa prediction_loss vs constraint terms
para explicar o plateau do teste overfit-1-batch (G0-a)."""
import sys
from pathlib import Path
import torch
from torch.optim import AdamW
from torch_geometric.data import HeteroData

BASE_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2")
sys.path.append(str(BASE_DIR / "02_models"))
from gnn_rf_model import GNNRFModel
from physics_loss import CurriculumRFLoss

sys.path.insert(0, "/tmp/claude-1000/-trabalho-HERMES/12b0b65b-ed42-4c45-82f8-2ee37cae72d8/scratchpad")
from teste_overfit_gradiente import build_synthetic_batch

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
batch, dist_b, ndvi_b = build_synthetic_batch(seed=42, device=device)

model = GNNRFModel(terrain_dim=17, antenna_dim=6, hidden_dim=64, num_layers=4,
                    heads=4, edge_dim=2, output_dim=5, dropout=0.0,
                    use_physics_constraints=True).to(device)
model.train()
loss_fn = CurriculumRFLoss(frequency_mhz=900.0, distance_gradient_weight=0.05,
                            distance_gradient_n_pairs=64, variance_weight=0.02,
                            shadowing_ndvi_weight=0.03, shadowing_ndvi_min_corr=0.15,
                            learnable_weights=True).to(device)
optimizer = AdamW(list(model.parameters()) + list(loss_fn.parameters()), lr=3e-3)

for step in range(200):
    optimizer.zero_grad(set_to_none=True)
    out = model(batch)
    preds = out["predictions"]
    targets = batch["terrain"].rf_targets
    loss, ldict = loss_fn(preds, targets, distances=dist_b, dist_to_ant=dist_b, ndvi=ndvi_b)
    loss.backward()
    optimizer.step()
    if step % 40 == 0 or step == 199:
        comp = {k: float(v) for k, v in ldict.items() if torch.is_tensor(v) and v.numel() == 1}
        print(step, {k: round(v, 4) for k, v in comp.items()})
