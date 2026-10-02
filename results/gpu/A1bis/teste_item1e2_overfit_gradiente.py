#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu/A1bis/teste_item1e2_overfit_gradiente.py -- gate A1bis (ia-bug-silencioso),
CONTRA-AUDITORIA independente sobre o modelo_v3 CORRIGIDO (rf_decoder_v3.py +
v3_common.py + train_gnn_v3.py/train_mlp_v3.py). Script PROPRIO desta frente,
NAO reusa oficina-pipeline_A0/A0bis/A0ter.json nem forum-eng-ia_*.json como
numero final: recarrega dado real e reconstroi tudo do zero (init sem
cadeia), so IMPORTA v3_common/rf_decoder_v3 e os congelados por importlib
(mesma tecnica de gpu/A1/teste_abce_overfit_v3.py).

Item 1 (overfit de 1 batch real, dezenas de passos): razao
loss_final(media 5 ultimos)/loss_inicial(media 5 primeiros) <= 0.10.

Item 2 (gradiente vivo REESCRITO pelo forum-eng-ia): gradiente > 0 em TODO
parametro treinavel, EXCETO a lista nomeada dos 328.478 parametros mortos por
ARQUITETURA (gpu/A1_inv/forum-eng-ia_contagem_efetiva.json, confirmada
tambem no run JSON do A0ter, campo capacidade.parametros_sem_gradiente).
Controle positivo (GNN): com ET_TA povoado num batch SINTETICO (mesma ideia
de forum-eng-ia_contagem_efetiva.py, reexecutada por CODIGO PROPRIO aqui,
nao importado daquele script), lin_l.weight das camadas 0-2 da hetero-conv
terrain->antenna tem de GANHAR gradiente nao-nulo -- sem isso a sonda de
"gradiente zero" nao prova nada.
"""
from __future__ import annotations

import argparse
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

# Lista nomeada dos 328.478 parametros SEM gradiente por ARQUITETURA
# (fanout de 1 salto: lin_l.weight das 3 primeiras camadas da hetero-conv
# terrain->antenna nunca agrega nada porque nenhuma aresta ET_TA entra num
# batch de 1 salto; camada 3 inteira de antenna e descartada porque so
# terrain_embeddings alimenta o decoder; 3 cabecas do decoder nunca entram
# no forward do AffineDecoderV3/RFDecoder base).
PARAMS_MORTOS_POR_ARQUITETURA_GNN = {
    "decoder.coverage_head.bias", "decoder.coverage_head.weight",
    "decoder.path_loss_head.bias", "decoder.path_loss_head.weight",
    "decoder.rssi_head.bias", "decoder.rssi_head.weight",
    "encoder.convs.0.convs.<terrain___in_range_of___antenna>.lin_l.weight",
    "encoder.convs.1.convs.<terrain___in_range_of___antenna>.lin_l.weight",
    "encoder.convs.2.convs.<terrain___in_range_of___antenna>.lin_l.weight",
    "encoder.convs.3.convs.<terrain___in_range_of___antenna>.lin_l.bias",
    "encoder.convs.3.convs.<terrain___in_range_of___antenna>.lin_l.weight",
    "encoder.convs.3.convs.<terrain___in_range_of___antenna>.lin_r.weight",
    "encoder.layer_norms.3.antenna.bias",
    "encoder.layer_norms.3.antenna.weight",
}
PARAMS_MORTOS_POR_ARQUITETURA_MLP = {
    "decoder.coverage_head.bias", "decoder.coverage_head.weight",
    "decoder.path_loss_head.bias", "decoder.path_loss_head.weight",
    "decoder.rssi_head.bias", "decoder.rssi_head.weight",
}


def log(t0, msg):
    print(f"[item1e2_v3 +{time.perf_counter()-t0:.1f}s] {msg}", flush=True)


def grad_norms_por_parametro(model) -> dict:
    out = {}
    for name, p in model.named_parameters():
        out[name] = None if p.grad is None else float(p.grad.detach().norm().item())
    return out


def controle_positivo_et_ta_povoado() -> dict:
    """CPU, sintetico, CODIGO PROPRIO (nao importa forum-eng-ia_contagem_efetiva.py):
    constroi HeteroData minimo com a relacao terrain->antenna POVOADA (arestas
    reais, sem edge_attr -- mesma condicao 'fanout_com_ET_TA_povoado' do
    parecer) e confere que lin_l.weight das camadas 0-2 da hetero-conv
    terrain->antenna GANHA gradiente nao-nulo."""
    import sys as _sys
    _sys.path.insert(0, "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2/02_models")
    _sys.path.insert(0, "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2")
    from torch_geometric.data import HeteroData
    from gnn_rf_model import GNNRFModel

    AT = ("antenna", "propagates_to", "terrain")
    TT = ("terrain", "connects_to", "terrain")
    TA = ("terrain", "in_range_of", "antenna")

    torch.manual_seed(123)
    bs, nh, na = 50, 150, 4
    nt = bs + nh
    d = HeteroData()
    d["terrain"].x = torch.randn(nt, 18)
    d["antenna"].x = torch.randn(na, 6)
    src = torch.randint(0, na, (bs * 3,))
    dst = torch.arange(bs).repeat(3)
    d[AT].edge_index = torch.stack([src, dst])
    d[AT].edge_attr = torch.randn(bs * 3, 2)
    s2 = torch.randint(bs, nt, (bs * 3,))
    d2 = torch.arange(bs).repeat(3)
    d[TT].edge_index = torch.stack([s2, d2])
    d[TT].edge_attr = torch.randn(bs * 3, 2)
    # ET_TA POVOADO de fato (fonte=terrain, destino=antenna), SEM edge_attr
    # (para nao cair na armadilha do parecer P2.3 -- este teste e so o
    # controle positivo do gradiente de lin_l, nao o teste da armadilha).
    s3 = torch.randint(0, nt, (60,))
    d3 = torch.randint(0, na, (60,))
    d[TA].edge_index = torch.stack([s3, d3])

    torch.manual_seed(0)
    m = GNNRFModel(terrain_dim=18, antenna_dim=6, hidden_dim=256, num_layers=4,
                    heads=4, edge_dim=2, output_dim=5, dropout=0.1)
    m.train()
    pred = m(d)["predictions"][:bs]
    pred.float().pow(2).mean().backward()

    alvo = [
        "encoder.convs.0.convs.<terrain___in_range_of___antenna>.lin_l.weight",
        "encoder.convs.1.convs.<terrain___in_range_of___antenna>.lin_l.weight",
        "encoder.convs.2.convs.<terrain___in_range_of___antenna>.lin_l.weight",
    ]
    normas = {}
    for n, p in m.named_parameters():
        if n in alvo:
            normas[n] = None if p.grad is None else float(p.grad.detach().norm().item())
    ganhou_gradiente = all(
        (normas.get(n) is not None and normas[n] > 0.0) for n in alvo)
    return {
        "descricao": "batch sintetico CPU com ET_TA povoado (60 arestas terrain->antenna, sem edge_attr)",
        "normas_lin_l_camadas_0a2": normas,
        "controle_positivo_passou": bool(ganhou_gradiente),
    }


def correlacao_feature_x_alvo(x: torch.Tensor, y: torch.Tensor, limite: float = 0.98) -> dict:
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
                achados.append({"feature_col": i, "target_col": j, "corr": v})
    return {"corr_max_abs": float(np.nanmax(np.abs(corr_np))),
            "achados_corr_gt_%.2f" % limite: achados}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tipo", choices=["gnn", "mlp"], required=True)
    ap.add_argument("--cidade", default="bauru")
    ap.add_argument("--quadrante", default="Q2")
    ap.add_argument("--max-nodes", type=int, default=100000)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--saida", required=True)
    args = ap.parse_args()

    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    rf_data_file = f"transfer_dataset_{args.cidade}_v19_{args.quadrante}_enriched_v2.pt"
    graph_file = f"{args.cidade}_v19_{args.quadrante}_gpu.pt"
    graph_dir = v3.GRAPH_DIR_DEFAULT

    if args.tipo == "gnn":
        v3.patch_decoder_gnn()
        import gnn_rf_model
        assert gnn_rf_model.PhysicsConstrainedDecoder is v3.AffineDecoderV3
        mod = v3.carregar_modulo_congelado(v3.FROZEN_GNN_SCRIPT, "train_gnn_c0_spatial_a1bis")
    else:
        v3.patch_decoder_mlp()
        import rf_decoder
        assert rf_decoder.PhysicsConstrainedDecoder is v3.AffineDecoderV3
        mod = v3.carregar_modulo_congelado(v3.FROZEN_MLP_SCRIPT, "train_mlp_c0_spatial_a1bis")

    log(t0, f"tipo={args.tipo} device={device} carregando dado real ({args.cidade} {args.quadrante})...")

    ctx = v3.carregar_base_e_particoes(
        mod=mod, graph_dir=graph_dir, rf_data_file=rf_data_file, graph_file=graph_file,
        max_nodes=args.max_nodes, window_anchor="cobertura",
        grid_km=5.0, buffer_km=2.0, split_frac=(0.70, 0.15, 0.15),
        split_seed=42, smoke_geometria=True, mmap=True,
        precisa_arestas_ter_ter=(args.tipo == "gnn"),
        log=lambda m: log(t0, m),
    )
    tr_loc_np = ctx.parts_local["train"]
    tr_loc = torch.from_numpy(tr_loc_np)
    dist_train_local = ctx.dist_all[tr_loc].contiguous()
    COL_NDVI = mod.COL_NDVI

    if args.tipo == "gnn":
        from gnn_rf_model import GNNRFModel
        model = GNNRFModel(
            terrain_dim=int(ctx.x_full.shape[1]), antenna_dim=int(ctx.ant_x.shape[1]),
            hidden_dim=256, num_layers=4, heads=4, edge_dim=2,
            output_dim=5, dropout=0.1, use_physics_constraints=True).to(device)

        graph_train, _info = mod.induzir_particao(
            ctx.base, tr_loc, ctx.n_antenna, lambda m: log(t0, m), "train")
        # producao (A0-ter): k_terrain/k_antenna default -1 (vizinhanca completa)
        nn_kw = {mod.ET_AT: [-1], mod.ET_TT: [-1], mod.ET_TA: [-1]}
        from torch_geometric.loader import NeighborLoader
        loader = NeighborLoader(data=graph_train, num_neighbors=nn_kw,
                                 input_nodes=("terrain", None), batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)
        batch = next(iter(loader)).to(device)
        bs = batch["terrain"].batch_size
        seed_ids_local = batch["terrain"].n_id[:bs].cpu()
        dist_b = dist_train_local[seed_ids_local].to(device)
        targets_s = batch["terrain"].rf_targets[:bs].to(device)
        ndvi_b = (batch["terrain"].x[:bs, COL_NDVI].to(device)
                  if batch["terrain"].x.shape[1] > COL_NDVI else None)
        x_feat_batch = batch["terrain"].x[:bs].detach().cpu()
        params_mortos_esperados = PARAMS_MORTOS_POR_ARQUITETURA_GNN

        def forward_fn():
            out = model(batch)
            return out["predictions"][:bs]
    else:
        from rf_decoder import PhysicsConstrainedDecoder as DecoderPatched
        v3.patch_larguras_mlp(mod, v3.MLP_LARGURAS_NOVAS, v3.N_PARAMS_ALVO_NOMINAL_NOVO)
        MLPRFModel = getattr(mod, "MLPRFModel")
        model = MLPRFModel(DecoderPatched, terrain_dim=int(ctx.x_full.shape[1]),
                            larguras=v3.MLP_LARGURAS_NOVAS, dropout=0.1).to(device)

        x_train = ctx.x_full[tr_loc]
        y_train = ctx.base["terrain"].rf_targets[tr_loc]
        bs = min(args.batch_size, x_train.shape[0])
        x_batch = x_train[:bs].to(device)
        targets_s = y_train[:bs].to(device)
        dist_b = dist_train_local[:bs].to(device)
        ndvi_b = (x_batch[:, COL_NDVI] if x_batch.shape[1] > COL_NDVI else None)
        x_feat_batch = x_batch.detach().cpu()
        params_mortos_esperados = PARAMS_MORTOS_POR_ARQUITETURA_MLP

        def forward_fn():
            out = model(x_batch)
            return out["predictions"]

    n_params_treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(t0, f"modelo construido: n_params_treinaveis={n_params_treinaveis} bs_real={bs}")

    from physics_loss import CurriculumRFLoss
    loss_fn = CurriculumRFLoss(
        frequency_mhz=900.0, distance_gradient_weight=0.05,
        distance_gradient_n_pairs=512, variance_weight=0.02,
        shadowing_ndvi_weight=0.03, shadowing_ndvi_min_corr=0.15,
    ).to(device)
    loss_fn.set_phase(5)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_traj = []
    grad_norms_por_passo = {}
    model.train()
    t_treino0 = time.perf_counter()
    for step in range(args.n_steps):
        optimizer.zero_grad(set_to_none=True)
        preds = forward_fn()
        loss, ldict = loss_fn(preds, targets_s, distances=dist_b, ndvi=ndvi_b)
        loss.backward()
        loss_traj.append(float(loss.detach().item()))
        if step % 10 == 0 or step == args.n_steps - 1:
            grad_norms_por_passo[step] = grad_norms_por_parametro(model)
        optimizer.step()
    tempo_treino_s = time.perf_counter() - t_treino0

    razao = float(np.mean(loss_traj[-5:]) / np.mean(loss_traj[:5]))

    todos_nomes = set(grad_norms_por_passo[0].keys())
    zero_em_todas = []
    for nome in todos_nomes:
        normas = [grad_norms_por_passo[s].get(nome) for s in grad_norms_por_passo]
        if all((v is not None and v < 1e-12) for v in normas):
            zero_em_todas.append(nome)
    ativos_com_grad_zero = sorted(set(zero_em_todas) - params_mortos_esperados)
    mortos_confirmados = sorted(set(zero_em_todas) & params_mortos_esperados)
    mortos_nao_confirmados = sorted(params_mortos_esperados - set(zero_em_todas))
    gradiente_passa = (len(ativos_com_grad_zero) == 0)

    controle_positivo = controle_positivo_et_ta_povoado() if args.tipo == "gnn" else {
        "descricao": "controle positivo rodado so no braco GNN (MLP nao tem hetero-conv terrain->antenna)",
        "controle_positivo_passou": True}

    corr_res = correlacao_feature_x_alvo(x_feat_batch, targets_s.detach().cpu())
    correlacao_passa = len(corr_res["achados_corr_gt_0.98"]) == 0

    overfit_passa = razao <= 0.10
    item2_passa = gradiente_passa and bool(controle_positivo["controle_positivo_passou"])

    resultado = {
        "teste": f"A1bis_item1_overfit_item2_gradiente_reescrito_{args.tipo}",
        "tipo_modelo": args.tipo,
        "device": device,
        "cidade": args.cidade, "quadrante": args.quadrante, "max_nodes": args.max_nodes,
        "n_steps": args.n_steps, "lr": args.lr,
        "batch_size_pedido": args.batch_size, "batch_size_real": bs,
        "n_params_treinaveis": n_params_treinaveis,
        "tempo_treino_s": tempo_treino_s,
        "item1_overfit_1_batch": {
            "loss_primeiros_5_media": float(np.mean(loss_traj[:5])),
            "loss_ultimos_5_media": float(np.mean(loss_traj[-5:])),
            "razao_final_sobre_inicial": razao,
            "criterio": "razao <= 0.10",
            "passa": overfit_passa,
            "loss_trajetoria_amostrada": loss_traj[::10] + [loss_traj[-1]],
        },
        "item2_gradiente_reescrito": {
            "passos_medidos": sorted(grad_norms_por_passo.keys()),
            "params_mortos_por_arquitetura_esperados": sorted(params_mortos_esperados),
            "ativos_com_gradiente_zero_em_todas_medicoes": ativos_com_grad_zero,
            "mortos_esperados_confirmados_zero": mortos_confirmados,
            "mortos_esperados_NAO_confirmados_zero_ACHADO": mortos_nao_confirmados,
            "gradiente_vivo_fora_da_lista_nomeada": gradiente_passa,
            "controle_positivo_et_ta_povoado": controle_positivo,
            "passa": item2_passa,
        },
        "vazamento_correlacao_bonus": {**corr_res, "passa": correlacao_passa},
        "veredito_script": {
            "item1_overfit": "passa" if overfit_passa else "bloqueia",
            "item2_gradiente": "passa" if item2_passa else "bloqueia",
        },
        "tempo_total_s": time.perf_counter() - t0,
    }

    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    log(t0, f"gravado: {saida}")
    log(t0, f"razao_overfit={razao:.4f} gradiente_item2_passa={item2_passa}")


if __name__ == "__main__":
    sys.exit(main() or 0)
