#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu/A2_inv/regravar_diagnostico_saida_A2.py -- diagnostico do chefe/dono
26/09/2026, anomalia (1): `diagnostico_saida.canal_0` (path_loss_total) dos
8 run JSON do A2 misturava a populacao "PL valido" com a populacao
"sentinela" (`target[:,0] >= PL_TARGET_MAX_VALID = 299 dB`, marcador de
"sem alvo de PL", NAO um valor fisico -- mesma convencao de
`mod.PL_TARGET_MAX_VALID`/`metricas_particao.pl_valid` do congelado). Sem
mascara, o clamp de producao ([0,200]) TRAVA a predicao em 200 para a
populacao sentinela (cujo alvo e ~300), o que **aumenta** o erro ali de
~0 para ~100 dB -- violando a pre-condicao R-c0 do parecer ("fracao de
alvos fora da faixa do clamp = 0"), que o proprio parecer ja avisava que
pioraria o numero se violada. NAO e dupla escala nem coluna trocada: o
`AffineDecoderV3.forward()` ja aplica escala+offset (raw*100+100 etc.) e
`.fisico()` so aplica o CLAMP -- conferido lendo `rf_decoder_v3.py`. O
canal 3 (RSSI) nunca teve esse problema (nao usa a convencao de
sentinela) e os canais 1/2 tambem nao (seus alvos nunca chegam a ~300).

Correcao: `v3_common.diagnostico_saida_canais` (editado nesta entrega)
agora recebe `sentinela` e, para o canal 0, reporta como PRIMARIO o bloco
mascarado (so populacao "PL valido"), preservando os numeros antigos (sem
mascara) sob sufixo `_todos_com_sentinela` -- nada apagado, so
reclassificado.

Este script REGRAVA os 8 run JSON do A2 com o `diagnostico_saida`
corrigido, recomputado por INFERENCIA REAL sobre o checkpoint salvo (SO
forward, sem backward/otimizador) -- nao a partir de `pred_afim` salvo em
disco (o `.npz` de teste so grava `pred` fisico pos-clamp, nunca a saida
afim pre-clamp; ver `gerar_predicoes_teste_gnn`/`_mlp`, campo `pred_afim`
NAO persistido no `.npz`). O `diagnostico_saida` ORIGINAL (com o bug) e
preservado em `diagnostico_saida_ANTES_correcao_sentinela_A2` -- nada
apagado. Contexto (`ctx`, grafo/particoes) e carregado 1x POR CIDADE e
reusado entre rep1/rep2 e GNN/MLP (mesmo dataset/split/config nas 4
corridas de cada cidade), para nao gastar GPU/CPU 8x.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODELO_V3_DIR = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/gpu/modelo_v3")
sys.path.insert(0, str(MODELO_V3_DIR))
import v3_common as v3  # noqa: E402

A2_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/gpu/A2")

RUNS = [
    {"tipo": "gnn", "run_label": "gnn_v3_a2_bauru_Q1_rep1", "cidade": "bauru"},
    {"tipo": "gnn", "run_label": "gnn_v3_a2_bauru_Q1_rep2", "cidade": "bauru"},
    {"tipo": "gnn", "run_label": "gnn_v3_a2_lins_Q1_rep1", "cidade": "lins"},
    {"tipo": "gnn", "run_label": "gnn_v3_a2_lins_Q1_rep2", "cidade": "lins"},
    {"tipo": "mlp", "run_label": "mlp_v3_a2_bauru_Q1_rep1", "cidade": "bauru"},
    {"tipo": "mlp", "run_label": "mlp_v3_a2_bauru_Q1_rep2", "cidade": "bauru"},
    {"tipo": "mlp", "run_label": "mlp_v3_a2_lins_Q1_rep1", "cidade": "lins"},
    {"tipo": "mlp", "run_label": "mlp_v3_a2_lins_Q1_rep2", "cidade": "lins"},
]

_CTX_CACHE: dict = {}


def log(t0, msg):
    print(f"[regravar_A2 +{time.perf_counter()-t0:.1f}s] {msg}", flush=True)


def carregar_ctx_cidade(cidade: str, cfg: dict, mod, t0):
    if cidade in _CTX_CACHE:
        return _CTX_CACHE[cidade]
    ctx = v3.carregar_base_e_particoes(
        mod=mod, graph_dir=Path(cfg["graph_dir"]), rf_data_file=cfg["rf_data_file"],
        graph_file=cfg["graph_file"], max_nodes=int(cfg.get("max_nodes", 0)),
        window_anchor=cfg.get("window_anchor", "cobertura"),
        grid_km=cfg["grid_km"], buffer_km=cfg["buffer_km"],
        split_frac=tuple(float(x) for x in cfg["split_frac"].split(",")),
        split_seed=int(cfg["split_seed"]), smoke_geometria=False,
        mmap=bool(cfg.get("mmap", True)), precisa_arestas_ter_ter=True,
        log=lambda m: log(t0, f"[{cidade}] {m}"))
    _CTX_CACHE[cidade] = ctx
    return ctx


def processar_gnn(item: dict, t0) -> dict:
    run_dir = A2_DIR / item["run_label"]
    run_json_path = run_dir / f"run_{item['run_label']}.json"
    rec = json.load(open(run_json_path, "r", encoding="utf-8"))
    cfg = rec["config"]

    v3.patch_decoder_gnn()
    import gnn_rf_model
    assert gnn_rf_model.PhysicsConstrainedDecoder is v3.AffineDecoderV3
    mod = v3.carregar_modulo_congelado(v3.FROZEN_GNN_SCRIPT, f"train_gnn_c0_spatial_regravar_{item['run_label']}")
    v3.patch_neighborloader_disjoint(mod, disjoint_treino=False)

    ctx = carregar_ctx_cidade(item["cidade"], cfg, mod, t0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = gnn_rf_model.GNNRFModel(
        terrain_dim=int(ctx.x_full.shape[1]), antenna_dim=int(ctx.ant_x.shape[1]),
        hidden_dim=rec["modelo"]["hidden_dim"], num_layers=rec["modelo"]["num_layers"],
        heads=rec["modelo"]["heads"], edge_dim=2, output_dim=rec["modelo"]["output_dim"],
        dropout=rec["modelo"]["dropout"], use_physics_constraints=True).to(device)
    st = torch.load(run_dir / "checkpoints" / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(st["model_state_dict"])
    model.eval()

    eval_bs = int(cfg.get("eval_batch_size") or cfg["batch_size"])
    K_EVAL = -1  # A0-ter: avaliacao SEMPRE com vizinhanca completa, independente do k de treino
    saida_test = v3.gerar_predicoes_teste_gnn(
        mod=mod, ctx=ctx, model=model, device=device, k_antenna=K_EVAL, k_terrain=K_EVAL,
        eval_batch_size=eval_bs, seed=int(cfg["seed"]), log=lambda m: log(t0, m),
        particao="test", disjoint=True)
    saida_val = v3.gerar_predicoes_teste_gnn(
        mod=mod, ctx=ctx, model=model, device=device, k_antenna=K_EVAL, k_terrain=K_EVAL,
        eval_batch_size=eval_bs, seed=int(cfg["seed"]), log=lambda m: log(t0, m),
        particao="val", disjoint=True)
    return _gravar_correcao(rec, run_json_path, saida_val, saida_test)


def processar_mlp(item: dict, t0) -> dict:
    run_dir = A2_DIR / item["run_label"]
    run_json_path = run_dir / f"run_{item['run_label']}.json"
    rec = json.load(open(run_json_path, "r", encoding="utf-8"))
    cfg = rec["config"]

    v3.patch_decoder_mlp()
    import rf_decoder
    assert rf_decoder.PhysicsConstrainedDecoder is v3.AffineDecoderV3
    mod = v3.carregar_modulo_congelado(v3.FROZEN_MLP_SCRIPT, f"train_mlp_c0_spatial_regravar_{item['run_label']}")

    ctx = carregar_ctx_cidade(item["cidade"], cfg, mod, t0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    larguras = tuple(rec["modelo"]["larguras_encoder"])
    from rf_decoder import PhysicsConstrainedDecoder as DecoderPatched
    MLPRFModel = getattr(mod, "MLPRFModel")
    model = MLPRFModel(DecoderPatched, terrain_dim=int(ctx.x_full.shape[1]),
                        larguras=larguras, dropout=rec["modelo"]["dropout"]).to(device)
    st = torch.load(run_dir / "checkpoints" / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(st["model_state_dict"])
    model.eval()

    eval_bs = int(cfg.get("eval_batch_size") or cfg["batch_size"])
    saida_test = v3.gerar_predicoes_teste_mlp(
        mod=mod, ctx=ctx, model=model, device=device, eval_batch_size=eval_bs,
        seed=int(cfg["seed"]), log=lambda m: log(t0, m), particao="test")
    saida_val = v3.gerar_predicoes_teste_mlp(
        mod=mod, ctx=ctx, model=model, device=device, eval_batch_size=eval_bs,
        seed=int(cfg["seed"]), log=lambda m: log(t0, m), particao="val")
    return _gravar_correcao(rec, run_json_path, saida_val, saida_test)


def _gravar_correcao(rec, run_json_path, saida_val, saida_test):
    diag_novo = {
        "val": v3.diagnostico_saida_canais(saida_val["pred_afim"], saida_val["pred"],
                                            saida_val["target"], sentinela=saida_val["sentinela"]),
        "test": v3.diagnostico_saida_canais(saida_test["pred_afim"], saida_test["pred"],
                                             saida_test["target"], sentinela=saida_test["sentinela"]),
    }
    if "diagnostico_saida_ANTES_correcao_sentinela_A2" not in rec:
        rec["diagnostico_saida_ANTES_correcao_sentinela_A2"] = rec.get("diagnostico_saida")
    rec["diagnostico_saida"] = diag_novo
    rec["nota_correcao_A2_sentinela"] = (
        "26/09/2026: canal_0 (path_loss_total) do diagnostico_saida original misturava a "
        "populacao 'PL valido' com a populacao 'sentinela' (target[:,0]>=299 dB, marcador de "
        "ausencia de alvo, nao valor fisico); o clamp [0,200] travava a predicao em 200 e o erro "
        "contra o alvo ~300 explodia so por causa dessa populacao (violacao de R-c0 do parecer). "
        "Recomputado por INFERENCIA REAL sobre o checkpoint salvo (nao a partir do .npz, que so "
        "grava pred fisico, nao pred_afim); diagnostico_saida.canal_0 agora reporta a populacao "
        "'PL valido' como PRIMARIO e preserva os numeros antigos sob sufixo "
        "'_todos_com_sentinela'; original completo preservado em "
        "diagnostico_saida_ANTES_correcao_sentinela_A2.")
    with open(run_json_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    c0 = diag_novo["test"]["canal_0"]
    return {
        "mae_pos_clamp_valido_test": c0["mae_pos_clamp"],
        "mae_pos_clamp_todos_com_sentinela_test": c0["mae_pos_clamp_todos_com_sentinela"],
        "frac_alvo_sentinela_test": c0["frac_alvo_sentinela"],
        "mae_pl_db_producao_referencia": (
            rec.get("selecao", {}).get("test_no_melhor_ckpt", {}).get("mae_pl_db")),
    }


def main() -> int:
    t0 = time.perf_counter()
    resumo = {}
    for item in RUNS:
        log(t0, f"=== {item['run_label']} ({item['tipo']}) ===")
        fn = processar_gnn if item["tipo"] == "gnn" else processar_mlp
        resumo[item["run_label"]] = fn(item, t0)
        log(t0, f"{item['run_label']}: {resumo[item['run_label']]}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out_path = Path(__file__).resolve().parent / "resumo_regravar_diagnostico_saida_A2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    log(t0, f"gravado: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
