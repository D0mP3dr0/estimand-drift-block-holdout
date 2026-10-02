#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu/A1/teste_abce_overfit_v3.py -- gate A1 (ia-bug-silencioso), CONTRA-AUDITORIA
independente da medicao adiantada `modelo_v3/smoke/overfit_1_batch_v3.json`
(NAO reusa aquele JSON: recarrega dado real e re-roda tudo do zero, script
proprio, so IMPORTA v3_common/rf_decoder_v3 e os congelados por importlib).

Testes classicos (a),(b),(c),(e) rodados JUNTOS sobre o MESMO 1 batch REAL
(Bauru Q2, --mmap, --max-nodes 100000), para GNN (--tipo gnn) ou MLP
(--tipo mlp):
  (a) overfit de 1 batch: 300 passos de Adam sobre o MESMO batch fixo,
      razao loss_final(media 5 ultimos)/loss_inicial(media 5 primeiros).
  (b) norma de gradiente por parametro a cada 20 passos; flag se algum
      parametro ATIVO (fora das 3 cabecas mortas por design: path_loss_head,
      rssi_head, coverage_head, que nunca entram no forward do
      AffineDecoderV3/RFDecoder base) tiver norma ~0 em TODAS as medicoes.
  (c) saturacao: fracao |pre-ativacao|>4 nos canais 0-3 (deveria ser 0 por
      construcao afim) e fracao saturada do sigmoid no canal 4
      (out<0.01 ou >0.99); fracao no clamp fisico() por canal, medida FORA
      do treino (so para diagnostico, apos os 300 passos).
  (e) correlacao feature x alvo no MESMO batch real (Pearson, |corr|>0.98
      sem explicacao fisica = achado de vazamento).

Nao editar nada em modelo_v3/ nem nos congelados. So leitura de dado real.
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


def log(t0, msg):
    print(f"[teste_abce_v3 +{time.perf_counter()-t0:.1f}s] {msg}", flush=True)


def grad_norms_por_parametro(model) -> dict:
    out = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            out[name] = None
        else:
            out[name] = float(p.grad.detach().norm().item())
    return out


def saturacao_canais(raw: torch.Tensor, out: torch.Tensor) -> dict:
    """raw = pre-ativacao [N,5] (saida de model.decoder.layers, ANTES da
    transformacao afim/sigmoid); out = saida do decoder (pos afim/sigmoid)."""
    d = {}
    for c in range(4):
        d[f"canal{c}_frac_pre_ativ_abs_gt4"] = float((raw[:, c].abs() > 4.0).float().mean())
    sig = out[:, 4]
    d["canal4_frac_sigmoid_saturado_lt01_ou_gt099"] = float(
        (((sig < 0.01) | (sig > 0.99)).float().mean()))
    d["canal4_min"] = float(sig.min())
    d["canal4_max"] = float(sig.max())
    return d


def fracao_no_clamp(pred: torch.Tensor, fisico_out: torch.Tensor) -> dict:
    d = {}
    nomes = ["path_loss_total", "path_loss_vegetation", "path_loss_terrain", "rssi", "coverage_prob"]
    for c, nome in enumerate(nomes):
        diff = (pred[:, c] - fisico_out[:, c]).abs() > 1e-6
        d[nome] = float(diff.float().mean())
    return d


def correlacao_feature_x_alvo(x: torch.Tensor, y: torch.Tensor, limite: float = 0.98) -> dict:
    """Pearson entre cada coluna de x (features) e cada coluna de y (alvos),
    sobre as linhas fornecidas (batch real). Retorna so os pares com
    |corr| > limite (achado de vazamento a investigar) + o maximo geral."""
    x = x.float()
    y = y.float()
    xc = x - x.mean(dim=0, keepdim=True)
    yc = y - y.mean(dim=0, keepdim=True)
    xs = xc.std(dim=0, keepdim=True).clamp(min=1e-8)
    ys = yc.std(dim=0, keepdim=True).clamp(min=1e-8)
    xn = xc / xs
    yn = yc / ys
    n = x.shape[0]
    corr = (xn.T @ yn) / n  # [n_feat, n_target]
    corr_np = corr.detach().cpu().numpy()
    achados = []
    for i in range(corr_np.shape[0]):
        for j in range(corr_np.shape[1]):
            v = float(corr_np[i, j])
            if abs(v) > limite:
                achados.append({"feature_col": i, "target_col": j, "corr": v})
    return {
        "shape_features_x_alvos": list(corr_np.shape),
        "corr_max_abs": float(np.nanmax(np.abs(corr_np))),
        "achados_corr_gt_%.2f" % limite: achados,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tipo", choices=["gnn", "mlp"], required=True)
    ap.add_argument("--cidade", default="bauru")
    ap.add_argument("--quadrante", default="Q2")
    ap.add_argument("--max-nodes", type=int, default=100000)
    ap.add_argument("--n-steps", type=int, default=300)
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
        mod = v3.carregar_modulo_congelado(v3.FROZEN_GNN_SCRIPT, "train_gnn_c0_spatial_teste_abce")
    else:
        v3.patch_decoder_mlp()
        import rf_decoder
        assert rf_decoder.PhysicsConstrainedDecoder is v3.AffineDecoderV3
        mod = v3.carregar_modulo_congelado(v3.FROZEN_MLP_SCRIPT, "train_mlp_c0_spatial_teste_abce")

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
    log(t0, f"dado carregado: n_ter_total={ctx.n_ter_total} janela={ctx.n_ter} "
             f"train={len(tr_loc_np)}")

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
        nn_kw = {mod.ET_AT: [20], mod.ET_TT: [8], mod.ET_TA: [20]}
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

        def forward_fn():
            out = model(batch)
            return out["predictions"][:bs]
    else:
        from rf_decoder import PhysicsConstrainedDecoder as DecoderPatched
        MLPRFModel = getattr(mod, "MLPRFModel")
        MLP_LARGURAS = getattr(mod, "MLP_LARGURAS")
        model = MLPRFModel(DecoderPatched, terrain_dim=int(ctx.x_full.shape[1]),
                            larguras=MLP_LARGURAS, dropout=0.1).to(device)

        x_train = ctx.x_full[tr_loc]
        y_train = ctx.base["terrain"].rf_targets[tr_loc]
        bs = min(args.batch_size, x_train.shape[0])
        x_batch = x_train[:bs].to(device)
        targets_s = y_train[:bs].to(device)
        dist_b = dist_train_local[:bs].to(device)
        ndvi_b = (x_batch[:, COL_NDVI] if x_batch.shape[1] > COL_NDVI else None)
        x_feat_batch = x_batch.detach().cpu()

        def forward_fn():
            out = model(x_batch)
            return out["predictions"]

    n_params_treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(t0, f"modelo construido: n_params_treinaveis={n_params_treinaveis} bs_real={bs}")

    from physics_loss import CurriculumRFLoss  # 02_models, ja no sys.path via patch_decoder_*
    loss_fn = CurriculumRFLoss(
        frequency_mhz=900.0, distance_gradient_weight=0.05,
        distance_gradient_n_pairs=512, variance_weight=0.02,
        shadowing_ndvi_weight=0.03, shadowing_ndvi_min_corr=0.15,
    ).to(device)
    loss_fn.set_phase(5)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dead_heads_esperadas = {
        "decoder.path_loss_head.weight", "decoder.path_loss_head.bias",
        "decoder.rssi_head.weight", "decoder.rssi_head.bias",
        "decoder.coverage_head.weight", "decoder.coverage_head.bias",
    }

    loss_traj = []
    grad_norms_por_passo = {}
    sat_por_passo = []
    model.train()
    t_treino0 = time.perf_counter()
    for step in range(args.n_steps):
        optimizer.zero_grad(set_to_none=True)
        preds = forward_fn()
        loss, ldict = loss_fn(preds, targets_s, distances=dist_b, ndvi=ndvi_b)
        loss.backward()
        loss_traj.append(float(loss.detach().item()))

        if step % 20 == 0 or step == args.n_steps - 1:
            grad_norms_por_passo[step] = grad_norms_por_parametro(model)
            with torch.no_grad():
                emb = (model.encoder(batch)["terrain"] if args.tipo == "gnn"
                       else model.encoder(x_batch))
                raw = model.decoder.layers(emb)
                out_now = model.decoder(emb)
                sat_por_passo.append({"step": step, **saturacao_canais(raw, out_now)})
        optimizer.step()
    tempo_treino_s = time.perf_counter() - t_treino0

    razao = float(np.mean(loss_traj[-5:]) / np.mean(loss_traj[:5]))

    # -- (b) gradiente vivo: parametro ATIVO (fora das 3 cabecas mortas) com
    #    norma ~0 em TODAS as medicoes registradas --
    todos_nomes = set(grad_norms_por_passo[0].keys())
    zero_em_todas = []
    for nome in todos_nomes:
        normas = [grad_norms_por_passo[s].get(nome) for s in grad_norms_por_passo]
        if all((v is not None and v < 1e-12) for v in normas):
            zero_em_todas.append(nome)
    camadas_ativas_com_grad_zero = sorted(set(zero_em_todas) - dead_heads_esperadas)
    camadas_mortas_conhecidas_confirmadas = sorted(set(zero_em_todas) & dead_heads_esperadas)
    gradiente_vivo_fora_das_cabecas_mortas = (len(camadas_ativas_com_grad_zero) == 0)

    # -- (c) saturacao final + fracao no clamp fisico() (fora do treino, so
    #    diagnostico) --
    with torch.no_grad():
        emb_final = (model.encoder(batch)["terrain"] if args.tipo == "gnn"
                     else model.encoder(x_batch))
        raw_final = model.decoder.layers(emb_final)
        pred_final = model.decoder(emb_final)
        fisico_final = model.decoder.fisico(pred_final)
    sat_final = saturacao_canais(raw_final, pred_final)
    clamp_final = fracao_no_clamp(pred_final, fisico_final)
    canais_0a3_saturados_ok = all(
        sat_final[f"canal{c}_frac_pre_ativ_abs_gt4"] < 1e-9 for c in range(4))
    clamp_0a3_ok = all(clamp_final[n] < 0.01 for n in
                        ["path_loss_total", "path_loss_vegetation", "path_loss_terrain", "rssi"])

    # -- (e) correlacao feature x alvo, no MESMO batch real --
    corr_res = correlacao_feature_x_alvo(x_feat_batch, targets_s.detach().cpu())

    overfit_passa = razao <= 0.10
    gradiente_passa = gradiente_vivo_fora_das_cabecas_mortas
    saturacao_passa = canais_0a3_saturados_ok and clamp_0a3_ok
    correlacao_passa = len(corr_res["achados_corr_gt_0.98"]) == 0

    resultado = {
        "teste": f"A1_overfit_gradiente_saturacao_correlacao_1batch_REAL_{args.tipo}",
        "decoder": "AffineDecoderV3",
        "tipo_modelo": args.tipo,
        "device": device,
        "cidade": args.cidade,
        "quadrante": args.quadrante,
        "max_nodes": args.max_nodes,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "batch_size_pedido": args.batch_size,
        "batch_size_real": bs,
        "n_params_treinaveis": n_params_treinaveis,
        "tempo_treino_300_passos_s": tempo_treino_s,
        "a_overfit_1_batch": {
            "loss_primeiros_5_media": float(np.mean(loss_traj[:5])),
            "loss_ultimos_5_media": float(np.mean(loss_traj[-5:])),
            "razao_final_sobre_inicial": razao,
            "criterio": "razao <= 0.10",
            "passa": overfit_passa,
            "loss_trajetoria_amostrada": loss_traj[::20] + [loss_traj[-1]],
        },
        "b_gradiente_por_camada": {
            "passos_medidos": sorted(grad_norms_por_passo.keys()),
            "grad_norm_por_parametro_ultimo_passo_medido": grad_norms_por_passo[max(grad_norms_por_passo.keys())],
            "camadas_ATIVAS_com_gradiente_zero_em_todas_as_medicoes": camadas_ativas_com_grad_zero,
            "camadas_mortas_conhecidas_confirmadas_zero": camadas_mortas_conhecidas_confirmadas,
            "dead_heads_esperadas_por_design": sorted(dead_heads_esperadas),
            "gradiente_vivo_fora_das_cabecas_mortas": gradiente_vivo_fora_das_cabecas_mortas,
            "passa": gradiente_passa,
        },
        "c_saturacao": {
            "por_passo_amostrado": sat_por_passo,
            "final": sat_final,
            "fracao_no_clamp_fisico_final_por_canal": clamp_final,
            "canais_0a3_pre_ativacao_ok_zero_saturado": canais_0a3_saturados_ok,
            "canais_0a3_clamp_fisico_ok_quase_zero": clamp_0a3_ok,
            "criterio": "fracao saturada ~0 nos canais 0-3",
            "passa": saturacao_passa,
        },
        "e_correlacao_feature_x_alvo": {
            **corr_res,
            "criterio": "nenhum |corr| > 0.98 sem explicacao fisica",
            "passa": correlacao_passa,
        },
        "veredito_script": {
            "a_overfit": "passa" if overfit_passa else "bloqueia",
            "b_gradiente": "passa" if gradiente_passa else "bloqueia",
            "c_saturacao": "passa" if saturacao_passa else "bloqueia",
            "e_correlacao": "passa" if correlacao_passa else "bloqueia",
        },
        "tempo_total_s": time.perf_counter() - t0,
    }

    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    log(t0, f"gravado: {saida}")
    log(t0, f"razao overfit={razao:.4f} gradiente_vivo={gradiente_vivo_fora_das_cabecas_mortas} "
             f"saturacao_ok={saturacao_passa} correlacao_ok={correlacao_passa}")


if __name__ == "__main__":
    sys.exit(main() or 0)
