#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste de bug silencioso G0 (a) overfit em 1 batch + (b) gradiente fluindo,
sobre as classes FROZEN de treino (GNNRFModel + CurriculumRFLoss), com dado
sintetico pequeno (nao usa o dataset de producao -- teste barato e fora do
orcamento de treino longo).

Script de origem lido (READ-ONLY, nao modificado):
  /trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2/02_models/gnn_rf_model.py
  /trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2/02_models/physics_loss.py
mesmas classes importadas pelo script congelado
  EVIDENCIA_RESUBMISSAO/scripts/train_gnn_c0_spatial.py (linhas 851-853, 1232-1263).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.data import HeteroData

BASE_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2")
sys.path.append(str(BASE_DIR / "02_models"))
sys.path.append(str(BASE_DIR / "03_training"))

from gnn_rf_model import GNNRFModel          # noqa: E402
from physics_loss import CurriculumRFLoss     # noqa: E402

OUT = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/gpu/G0")


def build_synthetic_batch(seed=42, n_terrain=256, n_antenna=20, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    d = HeteroData()
    terrain_dim, antenna_dim = 17, 6
    d["terrain"].x = torch.randn(n_terrain, terrain_dim, generator=g)
    d["terrain"].num_nodes = n_terrain
    # alvo com sinal real ligado a uma combinacao linear das features + ruido,
    # para que overfit em 1 batch seja um teste honesto (nao alvo constante)
    w = torch.randn(terrain_dim, 5, generator=g) * 0.5
    targets = d["terrain"].x @ w
    targets = targets + torch.randn(n_terrain, 5, generator=g) * 0.05
    d["terrain"].rf_targets = targets

    d["antenna"].x = torch.randn(n_antenna, antenna_dim, generator=g)
    d["antenna"].num_nodes = n_antenna

    # antenna -> terrain (k=5 vizinhos por terreno)
    k = 5
    src = torch.randint(0, n_antenna, (n_terrain * k,), generator=g)
    dst = torch.repeat_interleave(torch.arange(n_terrain), k)
    d["antenna", "propagates_to", "terrain"].edge_index = torch.stack([src, dst])
    d["antenna", "propagates_to", "terrain"].edge_attr = torch.rand(n_terrain * k, 2, generator=g)
    d["terrain", "in_range_of", "antenna"].edge_index = torch.stack([dst, src])

    # terrain -> terrain (grafo aleatorio esparso, k=6)
    kt = 6
    src_t = torch.randint(0, n_terrain, (n_terrain * kt,), generator=g)
    dst_t = torch.repeat_interleave(torch.arange(n_terrain), kt)
    mask = src_t != dst_t
    d["terrain", "connects_to", "terrain"].edge_index = torch.stack([src_t[mask], dst_t[mask]])
    d["terrain", "connects_to", "terrain"].edge_attr = torch.rand(int(mask.sum()), 2, generator=g)

    dist_to_ant = torch.rand(n_terrain, generator=g) * 3000.0 + 50.0
    ndvi = d["terrain"].x[:, 12]
    return d.to(device), dist_to_ant.to(device), ndvi.to(device)


def named_module_groups(model):
    """Agrupa parametros por sub-modulo de alto nivel, para norma de gradiente por 'camada'."""
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parts = name.split(".")
        key = ".".join(parts[:3]) if len(parts) >= 3 else name
        groups.setdefault(key, []).append(p)
    return groups


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    batch, dist_b, ndvi_b = build_synthetic_batch(seed=42, device=device)

    model = GNNRFModel(terrain_dim=17, antenna_dim=6, hidden_dim=64, num_layers=4,
                        heads=4, edge_dim=2, output_dim=5, dropout=0.0,
                        use_physics_constraints=True).to(device)
    model.train()

    loss_fn = CurriculumRFLoss(frequency_mhz=900.0, distance_gradient_weight=0.05,
                                distance_gradient_n_pairs=64, variance_weight=0.02,
                                shadowing_ndvi_weight=0.03, shadowing_ndvi_min_corr=0.15,
                                learnable_weights=True).to(device)

    optimizer = AdamW(list(model.parameters()) + list(loss_fn.parameters()),
                       lr=3e-3, weight_decay=0.0)

    groups = named_module_groups(model)
    n_steps = 200
    losses = []
    grad_norm_hist = {k: [] for k in groups}

    t0 = time.perf_counter()
    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        preds = out["predictions"]
        targets = batch["terrain"].rf_targets
        loss, ldict = loss_fn(preds, targets, distances=dist_b, dist_to_ant=dist_b, ndvi=ndvi_b)
        loss.backward()

        if step % 20 == 0 or step == n_steps - 1:
            for k, params in groups.items():
                total = 0.0
                for p in params:
                    if p.grad is not None:
                        total += float(p.grad.detach().pow(2).sum())
                grad_norm_hist[k].append((step, total ** 0.5))

        optimizer.step()
        losses.append(float(loss.item()))

    elapsed = time.perf_counter() - t0

    loss_inicial = float(np.mean(losses[:5]))
    loss_final = float(np.mean(losses[-5:]))
    razao = loss_final / max(loss_inicial, 1e-12)

    # criterio overfit-1-batch: queda de pelo menos 90% (loss_final <= 0.10 * loss_inicial)
    overfit_passa = bool(razao <= 0.10)

    # criterio gradiente fluindo: nenhuma camada com norma ~0 (< 1e-8) na ultima medicao
    ultima_medicao = {k: v[-1][1] for k, v in grad_norm_hist.items()}
    camadas_mortas = [k for k, v in ultima_medicao.items() if v < 1e-8]
    gradiente_passa = bool(len(camadas_mortas) == 0)

    resultado = {
        "teste": "overfit_1_batch_e_gradiente_fluindo",
        "comando": ("/trabalho/ambientes/s33_amb_virtual/.venv/bin/python "
                    "teste_overfit_gradiente.py"),
        "device": device,
        "n_steps": n_steps,
        "n_terrain": 256, "n_antenna": 20,
        "loss_primeiros_5_steps_media": loss_inicial,
        "loss_ultimos_5_steps_media": loss_final,
        "razao_final_sobre_inicial": razao,
        "criterio_overfit": "razao <= 0.10 (queda >= 90%)",
        "overfit_passa": overfit_passa,
        "loss_trajetoria_amostrada": losses[::20] + [losses[-1]],
        "grad_norm_por_grupo_ultima_medicao": ultima_medicao,
        "grad_norm_historico": {k: v for k, v in grad_norm_hist.items()},
        "camadas_com_gradiente_zero": camadas_mortas,
        "gradiente_passa": gradiente_passa,
        "tempo_s": elapsed,
        "passa": bool(overfit_passa and gradiente_passa),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "teste_a_b_overfit_gradiente.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in resultado.items()
                       if k not in ("grad_norm_historico",)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
