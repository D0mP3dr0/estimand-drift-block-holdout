#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
G0 (continuacao) — overfit em 1 batch REAL (nao sintetico) + gradiente por
grupo + fracao de saturacao de softplus/sigmoid, sobre o MESMO carregamento
de dado que o smoke real (train_gnn_c0_spatial.py --smoke --max-nodes 100000
sobre bauru_v19_Q2), usando GNNRFModel de producao (hidden=256, 4 camadas,
4 heads, use_physics_constraints=True) e CurriculumRFLoss de producao
(mesmos hyperparametros/defaults do argparse do script congelado).

NAO edita o script congelado nem rf_decoder.py. So IMPORTA (importlib) o
script congelado para reusar as MESMAS funcoes de carga/janela/split/
inducao de subgrafo que ele usa, e reconstroi manualmente a mesma sequencia
que o main() dele executa ate o loader de treino, parando ali para extrair
UM batch real.
"""
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import bipartite_subgraph, subgraph

BASE_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2")
GRAPH_DIR = BASE_DIR / "graph_data"
SCRIPT_PATH = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
    "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/scripts/"
    "train_gnn_c0_spatial.py")
OUT = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/gpu/G0")

RF_DATA_FILE = "transfer_dataset_bauru_v19_Q2_enriched_v2.pt"
GRAPH_FILE = "bauru_v19_Q2_gpu.pt"
MAX_NODES = 100_000
SEED = 42


def load_frozen_module():
    spec = importlib.util.spec_from_file_location("train_gnn_c0_spatial_frozen", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # so define funcoes/classes; main() guardado por __main__
    return mod


def build_real_train_batch(mod, device):
    """Reproduz main() do script congelado ATE o loader de treino (mesma
    logica, mesmos argumentos de smoke), e devolve o PRIMEIRO batch real
    do loader de treino (ja em GPU) + metadados."""
    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    ET_AT = ("antenna", "propagates_to", "terrain")
    ET_TA = ("terrain", "in_range_of", "antenna")
    ET_TT = ("terrain", "connects_to", "terrain")
    PL_TARGET_MAX_VALID = mod.PL_TARGET_MAX_VALID

    rf_path = GRAPH_DIR / RF_DATA_FILE
    struct_path = GRAPH_DIR / GRAPH_FILE
    load_kw = dict(map_location="cpu", weights_only=False)
    t0 = time.perf_counter()
    rf_data = torch.load(rf_path, mmap=True, **load_kw)
    print(f"[+{time.perf_counter()-t0:.1f}s] rf_data carregado (mmap)")

    ty = rf_data["terrain"].y
    n_ter_total = int(ty.shape[0])

    dist_pre = None
    if hasattr(rf_data["terrain"], "dist_nearest_m"):
        c = rf_data["terrain"].dist_nearest_m.float()
        if c.shape[0] == n_ter_total and float(c.std()) > 1.0:
            dist_pre = c

    feats = rf_data["terrain"].features_raw
    feats = torch.from_numpy(feats) if isinstance(feats, np.ndarray) else feats
    feats = feats.float()

    pos_deg = rf_data["terrain"].pos.float()
    tgt_all = torch.as_tensor(rf_data["terrain"].y).float()
    ant_x = torch.as_tensor(rf_data["antenna"].x).float()
    n_antenna = int(ant_x.shape[0])
    ant_pos = (rf_data["antenna"].pos.float() if hasattr(rf_data["antenna"], "pos") else ant_x[:, :2])

    ei_at_full = rf_data[ET_AT].edge_index.long()
    ea_at_full = rf_data[ET_AT].edge_attr.float() if hasattr(rf_data[ET_AT], "edge_attr") else None

    tt_in_rf = ET_TT in rf_data.edge_types
    if tt_in_rf:
        ei_tt_full = rf_data[ET_TT].edge_index.long()
        ea_tt_full = rf_data[ET_TT].edge_attr.float() if hasattr(rf_data[ET_TT], "edge_attr") else None
    else:
        sd = torch.load(struct_path, **load_kw)
        t2t = sd["dem", "adjacent_to", "dem"]
        ei_tt_full = t2t.edge_index.long().clone()
        ea_tt_full = t2t.edge_attr.float().clone() if hasattr(t2t, "edge_attr") and t2t.edge_attr is not None else None
        del sd, t2t
        gc.collect()

    del rf_data
    gc.collect()
    print(f"[+{time.perf_counter()-t0:.1f}s] grafo completo: {n_ter_total:,} terrain")

    pos_m_full, geo = mod.latlon_graus_para_metros(pos_deg)
    ancora = (tgt_all[:, 0] < PL_TARGET_MAX_VALID).numpy()
    g_idx = mod.janela_contigua(pos_m_full, MAX_NODES, ancora)
    subamostrado = int(g_idx.size) != n_ter_total
    g_idx_t = torch.from_numpy(g_idx)

    base = HeteroData()
    base["terrain"].pos = pos_deg[g_idx_t]
    base["terrain"].rf_targets = tgt_all[g_idx_t]
    base["antenna"].x = ant_x
    base["antenna"].num_nodes = n_antenna
    x_base = feats[g_idx_t]
    if subamostrado:
        ei_at_full, ea_at_full = bipartite_subgraph(
            (torch.arange(n_antenna), g_idx_t), ei_at_full, ea_at_full,
            relabel_nodes=True, size=(n_antenna, n_ter_total))
        ei_tt_full, ea_tt_full = subgraph(g_idx_t, ei_tt_full, ea_tt_full,
                                          relabel_nodes=True, num_nodes=n_ter_total)
    base[ET_AT].edge_index = ei_at_full
    if ea_at_full is not None:
        base[ET_AT].edge_attr = ea_at_full
    base[ET_TA].edge_index = ei_at_full[[1, 0]]
    base[ET_TT].edge_index = ei_tt_full
    base[ET_TT].edge_attr = (ea_tt_full if ea_tt_full is not None
                             else torch.zeros((ei_tt_full.shape[1], 2), dtype=torch.float32))
    del tgt_all, pos_deg
    gc.collect()

    n_ter = int(x_base.shape[0])
    pos_m = pos_m_full[g_idx]
    dist_all = (dist_pre[g_idx_t] if dist_pre is not None else None)

    # split espacial (mesma logica/ajuste de smoke)
    grid_km, buffer_km = 5.0, 2.0
    fracs = (0.70, 0.15, 0.15)
    ext = min((pos_m[:, 0].max() - pos_m[:, 0].min()) / 1000.0,
              (pos_m[:, 1].max() - pos_m[:, 1].min()) / 1000.0)
    if ext / max(grid_km, 1e-9) < 6.0:
        novo = float(ext / 6.0)
        buffer_km = novo * (buffer_km / grid_km)
        grid_km = novo

    def _log(msg):
        print(f"[+{time.perf_counter()-t0:.1f}s] {msg}")

    sys.path.append(str(BASE_DIR / "03_training"))
    from spatial_cv import SpatialKFold
    parts_local, split_info = mod.split_espacial_3vias(
        pos_m, grid_km, buffer_km, fracs, 42, SpatialKFold, _log)
    tr_loc = torch.from_numpy(parts_local["train"])

    d_mean_tr = float(dist_all[tr_loc].mean())
    d_std_tr = max(float(dist_all[tr_loc].std()), 1.0)
    x_full = torch.cat([x_base, ((dist_all - d_mean_tr) / d_std_tr).unsqueeze(1)], dim=1)
    base["terrain"].x = x_full

    graphs = {}
    for k in ("train",):
        graphs[k], _info = mod.induzir_particao(base, torch.from_numpy(parts_local[k]), n_antenna, _log, k)

    nn_kw = {ET_AT: [20], ET_TT: [8], ET_TA: [20]}
    loader = NeighborLoader(data=graphs["train"], num_neighbors=nn_kw,
                            input_nodes=("terrain", None), batch_size=8192,
                            shuffle=True, num_workers=0)
    dist_train_local = dist_all[tr_loc].contiguous()

    batch = next(iter(loader))
    batch = batch.to(device)
    bs = batch["terrain"].batch_size
    seed_ids = batch["terrain"].n_id[:bs].cpu()
    dist_b = dist_train_local[seed_ids].to(device)
    COL_NDVI = mod.COL_NDVI
    ndvi_b = (batch["terrain"].x[:bs, COL_NDVI] if batch["terrain"].x.shape[1] > COL_NDVI else None)
    targets_b = batch["terrain"].rf_targets[:bs]

    meta = {
        "n_terrain_dataset_completo": n_ter_total,
        "n_terrain_janela": n_ter,
        "n_terrain_train_particao": int(tr_loc.numel()),
        "batch_size_semente": int(bs),
        "terrain_dim": int(x_full.shape[1]),
        "antenna_dim": int(ant_x.shape[1]),
        "grid_km_ajustado": grid_km, "buffer_km_ajustado": buffer_km,
        "rf_data_file": RF_DATA_FILE, "graph_file": GRAPH_FILE,
        "alvo_e_real": True,
        "alvo_colunas": ["path_loss_total_dB", "path_loss_vegetation_dB",
                         "path_loss_terrain_dB", "rssi_dBm", "coverage_prob"],
        "alvo_amostra_stats": {
            "mean": targets_b.mean(dim=0).tolist(),
            "std": targets_b.std(dim=0).tolist(),
        },
    }
    return batch, dist_b, ndvi_b, x_full.shape[1], int(ant_x.shape[1]), meta


def named_module_groups(model):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parts = name.split(".")
        key = ".".join(parts[:3]) if len(parts) >= 3 else name
        groups.setdefault(key, []).append(p)
    return groups


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    mod = load_frozen_module()
    sys.path.append(str(BASE_DIR / "02_models"))
    sys.path.append(str(BASE_DIR / "03_training"))
    from gnn_rf_model import GNNRFModel
    from physics_loss import CurriculumRFLoss

    batch, dist_b, ndvi_b, terrain_dim, antenna_dim, meta = build_real_train_batch(mod, device)
    print("meta:", json.dumps(meta, indent=2))

    torch.manual_seed(SEED)

    # ==== modelo/loss de PRODUCAO (mesmos kwargs de train_gnn_c0_spatial.py:1228-1263) ====
    model = GNNRFModel(terrain_dim=terrain_dim, antenna_dim=antenna_dim,
                        hidden_dim=256, num_layers=4, heads=4, edge_dim=2,
                        output_dim=5, dropout=0.1, use_physics_constraints=True).to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    dead_heads = ["decoder.path_loss_head", "decoder.rssi_head", "decoder.coverage_head"]
    n_dead = sum(p.numel() for name, p in model.named_parameters()
                 if any(name.startswith(h) for h in dead_heads))

    loss_fn = CurriculumRFLoss(frequency_mhz=900.0, distance_gradient_weight=0.05,
                                distance_gradient_n_pairs=512, variance_weight=0.02,
                                shadowing_ndvi_weight=0.03, shadowing_ndvi_min_corr=0.15).to(device)

    optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    groups = named_module_groups(model)
    n_steps = 300
    losses = []
    grad_norm_hist = {k: [] for k in groups}
    sat_hist = []

    bs = batch["terrain"].batch_size
    targets = batch["terrain"].rf_targets[:bs]

    # hook para capturar a pre-ativacao (raw_predictions) do decoder e medir
    # saturacao de softplus (|x|>4 => derivada<~2%) / sigmoid (|x|>4 => idem)
    raw_holder = {}

    def hook_decoder(mod_, inp, out):
        raw_holder["raw"] = inp[0].detach() if isinstance(inp, tuple) else None

    # localizamos o Linear final do decoder.layers (penultima op antes do sigmoid/softplus)
    decoder_layers = model.decoder.layers
    handle = decoder_layers.register_forward_hook(
        lambda m, i, o: raw_holder.__setitem__("raw", o.detach()))

    t0 = time.perf_counter()
    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        preds = out["predictions"][:bs]
        loss, ldict = loss_fn(preds, targets, distances=dist_b, dist_to_ant=dist_b, ndvi=ndvi_b)
        loss.backward()

        if step % 20 == 0 or step == n_steps - 1:
            for k, params in groups.items():
                total = 0.0
                for p in params:
                    if p.grad is not None:
                        total += float(p.grad.detach().pow(2).sum())
                grad_norm_hist[k].append((step, total ** 0.5))
            raw = raw_holder.get("raw")
            if raw is not None:
                # canais 0-2 (softplus): saturado se |raw|>4 (softplus quase linear/quase 0)
                sat_softplus = float((raw[:, :3].abs() > 4.0).float().mean())
                # canal 3 (sigmoid do rssi escalado): saturado se |raw|>4
                sat_sigmoid_rssi = float((raw[:, 3].abs() > 4.0).float().mean())
                # canal 4 (coverage sigmoid, aplicado dentro do RFDecoder.forward antes do
                # PhysicsConstrainedDecoder sobrescrever): idem
                sat_sigmoid_cov = float((raw[:, 4].abs() > 4.0).float().mean())
            else:
                sat_softplus = sat_sigmoid_rssi = sat_sigmoid_cov = None
            sat_hist.append({"step": step, "frac_saturado_softplus_canais0-2": sat_softplus,
                             "frac_saturado_sigmoid_rssi_canal3": sat_sigmoid_rssi,
                             "frac_saturado_sigmoid_coverage_canal4": sat_sigmoid_cov})

        optimizer.step()
        losses.append(float(loss.item()))

    handle.remove()
    elapsed = time.perf_counter() - t0

    loss_inicial = float(np.mean(losses[:5]))
    loss_final = float(np.mean(losses[-5:]))
    razao = loss_final / max(loss_inicial, 1e-12)
    overfit_passa = bool(razao <= 0.10)

    ultima_medicao = {k: v[-1][1] for k, v in grad_norm_hist.items()}
    camadas_mortas = [k for k, v in ultima_medicao.items() if v < 1e-8]
    gradiente_passa = bool(len(camadas_mortas) == 0)

    # loss usa colunas 0-4 da saida constrained? checar via inspecao do dict ldict
    # e via alinhamento de preds.shape[1]==5 com targets.shape[1]==5
    usa_colunas_0a4 = bool(preds.shape[1] == 5 and targets.shape[1] == 5)

    resultado = {
        "teste": "overfit_1_batch_REAL_e_gradiente_producao",
        "comando": (f"/trabalho/ambientes/s33_amb_virtual/.venv/bin/python "
                    f"{Path(__file__).resolve()}"),
        "device": device,
        "dado_real": True,
        "meta_dado": meta,
        "modelo": {"hidden_dim": 256, "num_layers": 4, "heads": 4,
                   "use_physics_constraints": True, "dropout": 0.1,
                   "n_params_total": int(n_params),
                   "n_params_3_heads_mortos": int(n_dead),
                   "fracao_params_mortos": n_dead / n_params},
        "n_steps": n_steps, "lr": 3e-3,
        "loss_primeiros_5_steps_media": loss_inicial,
        "loss_ultimos_5_steps_media": loss_final,
        "razao_final_sobre_inicial": razao,
        "criterio_overfit": "razao <= 0.10 (queda >= 90%)",
        "overfit_passa": overfit_passa,
        "loss_trajetoria_amostrada": losses[::20] + [losses[-1]],
        "grad_norm_por_grupo_ultima_medicao": ultima_medicao,
        "grad_norm_historico": grad_norm_hist,
        "camadas_com_gradiente_zero": camadas_mortas,
        "gradiente_passa": gradiente_passa,
        "saturacao_softplus_sigmoid_por_passo": sat_hist,
        "loss_usa_colunas_0a4_da_saida_constrained": usa_colunas_0a4,
        "tempo_s": elapsed,
        "passa": bool(overfit_passa and gradiente_passa),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "teste_overfit_real.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in resultado.items()
                       if k not in ("grad_norm_historico", "meta_dado")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
