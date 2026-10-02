#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu/modelo_v3/e3_inferencia_checkpoints_antigos.py -- A0-ter item 4 (E3).

Parecer `forum-eng-ia_fanout_capacidade.md`, secao "Regra de avaliacao (3)":
"Teste so de inferencia nos checkpoints antigos (E3; o resultado vai para o
dono, sem citacao no Artigo 2 por D2)". SO INFERENCIA -- nenhum treino, zero
otimizador, zero backward.

Diferenca crucial para este script: usa o modelo ORIGINAL com
`PhysicsConstrainedDecoder` (o decodificador congelado de producao) --
`v3_common.patch_decoder_gnn()` NUNCA e chamado aqui. `gnn_rf_model` e
importado fresco (mesmo processo, nenhum outro modulo desta pasta o
patcheia antes), entao `GNNRFModel(..., use_physics_constraints=True)`
resolve `PhysicsConstrainedDecoder` para a classe ORIGINAL do congelado,
compativel com os pesos salvos nos checkpoints do E3 (que foram treinados
com esse decoder, nao com o `AffineDecoderV3`).

Checkpoints (seed 42, Q1, 100 % C0+C1):
  - bauru: EVIDENCIA_RESUBMISSAO/treinos/c0c1_bauru_s42_Q1_g10b2/
           (nome EXATO pedido pela tarefa -- existe).
  - lins:  EVIDENCIA_RESUBMISSAO/treinos/c0c1_lins_s42_Q1_g10b2/ NAO EXISTE
           neste disco (so existem variantes c0c1cf_lins_s42_Q1_g10b2 e
           c0c1v2_lins_s42_Q1_g10b2, ambas Q1/seed42/g10b2). Uso
           `c0c1cf_lins_s42_Q1_g10b2` (mesma familia de sufixo "cf" que
           `c0c1cf_bauru_s42_Q1_g10b2`, que TAMBEM existe ao lado do
           `c0c1_bauru...` sem sufixo) -- substituicao DECLARADA no JSON
           de saida (`checkpoints_usados.lins.substituicao`), nao decidida
           a olho: e a unica opcao Q1/seed42/g10b2 disponivel para lins.

Escala: o grafo completo de cada cidade Q1 tem 12.960.000 nos terrain
(`max_nodes=0` no run original) -- nao cabe no teto de 25 min de GPU deste
lote junto com o resto da entrega A0-ter. Uso janela de 100.000 nos
(`--max-nodes` implicito = 100000, mesma janela dos outros smokes desta
pasta), com `smoke_geometria=True` para a grade se ajustar ao tamanho da
janela (o grid_km=10 original nao caberia numa janela de 100k nos sem
esse ajuste). DECLARADO: (1) a particao de teste usada aqui e DIFERENTE
da particao de teste do run original (que usou o grafo inteiro) -- o hash
gravado no run JSON original NAO bate; (2) a normalizacao da feature de
distancia (`d_mean_tr`/`d_std_tr`) e recalculada sobre o TREINO desta
janela pequena, nao sobre o treino do grafo inteiro -- os MAE absolutos
aqui NAO sao os MAE de producao, so servem para a comparacao INTERNA entre
as 24 avaliacoes (A/B/C/D), que e o que o parecer pede (isolar sorteio e
campo receptivo, nao repetir o numero de producao).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_common as v3  # noqa: E402

TREINOS_DIR = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
    "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/treinos")

CHECKPOINTS = {
    "bauru": {
        "run_dir": "c0c1_bauru_s42_Q1_g10b2",
        "run_json": "run_c0c1_bauru_s42_Q1_g10b2.json",
        "substituicao": None,
    },
    "lins": {
        "run_dir": "c0c1v2_lins_s42_Q1_g10b2",
        "run_json": "run_c0c1v2_lins_s42_Q1_g10b2.json",
        "substituicao": (
            "c0c1_lins_s42_Q1_g10b2 nao existe neste disco (so ha variantes "
            "com sufixo). Usado c0c1v2_lins_s42_Q1_g10b2 -- MESMO dataset base "
            "(transfer_dataset_lins_v19_Q1_enriched_v2.pt, sem enriquecimento "
            "extra 'cf*') e MESMA arquitetura/config (GNNRFModel, hidden=256, "
            "heads=4, layers=4, dropout=0.1, n_params=1.812.515) do checkpoint "
            "de bauru usado acima; rejeitadas c0c1cf_lins_s42_Q1_g10b2 e "
            "cfslope_lins_s42_Q1_g10b2 por usarem dataset enriquecido "
            "'_cftudo'/'_cfslope' (feature set diferente do usado em bauru, "
            "quebraria a comparabilidade entre as duas cidades)."),
    },
}

MAX_NODES = 100_000
SPLIT_SEED = 42
SPLIT_FRAC = (0.70, 0.15, 0.15)
N_ORDENS = 5


def log(t0, msg):
    print(f"[e3_checkpoints_antigos +{time.perf_counter()-t0:.1f}s] {msg}", flush=True)


def carregar_ctx_e_modelo(cidade: str, info: dict, t0):
    run_dir = TREINOS_DIR / info["run_dir"]
    run_json_path = run_dir / info["run_json"]
    rec = json.load(open(run_json_path, "r", encoding="utf-8"))
    cfg = rec["config"]
    modelo_cfg = rec["modelo"]
    graph_dir = v3.GRAPH_DIR_DEFAULT
    rf_data_file = cfg["rf_data_file"]
    graph_file = cfg["graph_file"]

    # modulo congelado carregado FRESCO, SEM patch_decoder_gnn (item 4:
    # "use o modelo ORIGINAL com PhysicsConstrainedDecoder").
    mod = v3.carregar_modulo_congelado(v3.FROZEN_GNN_SCRIPT, f"train_gnn_c0_spatial_e3_{cidade}")
    import gnn_rf_model  # NAO patcheado (nenhum patch_decoder_gnn chamado neste processo)

    geom = rec.get("geometria", {})
    ctx = v3.carregar_base_e_particoes(
        mod=mod, graph_dir=graph_dir, rf_data_file=rf_data_file, graph_file=graph_file,
        max_nodes=MAX_NODES, window_anchor="cobertura",
        grid_km=geom.get("grid_km_usado", 10.0), buffer_km=geom.get("buffer_km_usado", 2.0),
        split_frac=SPLIT_FRAC, split_seed=SPLIT_SEED, smoke_geometria=True,
        mmap=True, precisa_arestas_ter_ter=True,
        log=lambda m: log(t0, f"[{cidade}] {m}"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = gnn_rf_model.GNNRFModel(
        terrain_dim=int(ctx.x_full.shape[1]), antenna_dim=int(ctx.ant_x.shape[1]),
        hidden_dim=modelo_cfg["hidden_dim"], num_layers=modelo_cfg["num_layers"],
        heads=modelo_cfg["heads"], edge_dim=2, output_dim=modelo_cfg["output_dim"],
        dropout=modelo_cfg["dropout"], use_physics_constraints=True).to(device)
    assert type(model.decoder).__name__ == "PhysicsConstrainedDecoder", (
        f"decoder inesperado {type(model.decoder).__name__} -- este script exige o ORIGINAL")

    ckpt_path = run_dir / "checkpoints" / "checkpoint_best.pt"
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(st["model_state_dict"])
    model.eval()
    return mod, ctx, model, device, rec


def rodar_grade(cidade: str, mod, ctx, model, device, t0):
    n_p = int(ctx.parts_local["test"].shape[0])
    eval_bs = 8192

    rng_seeds = np.random.default_rng(SPLIT_SEED)
    ordens = [rng_seeds.permutation(n_p) for _ in range(N_ORDENS)]

    def _avaliar2(k_terrain, k_antenna, disjoint, ordem):
        return v3.gerar_predicoes_teste_gnn(
            mod=mod, ctx=ctx, model=model, device=device,
            k_antenna=k_antenna, k_terrain=k_terrain,
            eval_batch_size=eval_bs, seed=SPLIT_SEED,
            log=lambda m: log(t0, f"[{cidade}] {m}"), particao="test",
            disjoint=disjoint, ordem_nos=ordem)

    log(t0, f"[{cidade}] grade A: loader original (k=8/20, nao disjunto), {N_ORDENS} ordens")
    grade_a = [_avaliar2(k_terrain=8, k_antenna=20, disjoint=False, ordem=o) for o in ordens]
    log(t0, f"[{cidade}] grade B: disjunto k=8/20, {N_ORDENS} ordens")
    grade_b = [_avaliar2(k_terrain=8, k_antenna=20, disjoint=True, ordem=o) for o in ordens]
    log(t0, f"[{cidade}] grade C: disjunto, vizinhanca completa")
    saida_c = _avaliar2(k_terrain=-1, k_antenna=-1, disjoint=True, ordem=None)
    log(t0, f"[{cidade}] grade D: NAO disjunto, vizinhanca completa")
    saida_d = _avaliar2(k_terrain=-1, k_antenna=-1, disjoint=False, ordem=None)

    idx_ref = grade_a[0]["idx_global"]
    for nome, grade in (("A", grade_a), ("B", grade_b)):
        for s in grade:
            assert np.array_equal(s["idx_global"], idx_ref), f"idx_global diverge na grade {nome}"
    assert np.array_equal(saida_c["idx_global"], idx_ref)
    assert np.array_equal(saida_d["idx_global"], idx_ref)

    def _std_por_no_entre_ordens(grade):
        pilha = np.stack([s["pred"][:, 3] for s in grade], axis=0)  # canal RSSI
        std_no = pilha.std(axis=0, ddof=0)
        return {"mediana": float(np.median(std_no)), "p95": float(np.percentile(std_no, 95)),
                "max": float(std_no.max())}

    def _mae_pop_por_ordem(grade):
        maes = [v3.mae_rssi_por_populacao(s["pred"], s["target"], s["sentinela"]) for s in grade]
        out = {}
        for pop in ("todos", "valido", "sentinela"):
            vals = [m[pop]["mae_rssi_db"] for m in maes if m[pop]["mae_rssi_db"] is not None]
            out[pop] = {
                "media": float(np.mean(vals)) if vals else None,
                "dp": float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0,
                "amplitude": float(max(vals) - min(vals)) if vals else None,
                "valores_por_ordem": vals,
            }
        return out

    mae_c = v3.mae_rssi_por_populacao(saida_c["pred"], saida_c["target"], saida_c["sentinela"])
    mae_d = v3.mae_rssi_por_populacao(saida_d["pred"], saida_d["target"], saida_d["sentinela"])

    # ensemble das 5 ordens de B contra C (sinal de Jensen: media de |erro|
    # de execucoes com ruido de sorteio >= |erro| do ensemble/determinismo).
    pred_ensemble_b = np.mean(np.stack([s["pred"] for s in grade_b], axis=0), axis=0)
    sentinela_ref = grade_b[0]["sentinela"]
    target_ref = grade_b[0]["target"]
    mae_ensemble_b = v3.mae_rssi_por_populacao(pred_ensemble_b, target_ref, sentinela_ref)

    mae_pop_a = _mae_pop_por_ordem(grade_a)
    mae_pop_b = _mae_pop_por_ordem(grade_b)
    std_a = _std_por_no_entre_ordens(grade_a)
    std_b = _std_por_no_entre_ordens(grade_b)

    delta_sorteio = {
        pop: (mae_pop_b[pop]["media"] - mae_c[pop]["mae_rssi_db"])
        if (mae_pop_b[pop]["media"] is not None and mae_c[pop]["mae_rssi_db"] is not None)
        else None
        for pop in ("todos", "valido", "sentinela")
    }
    delta_campo_receptivo = {
        pop: (mae_d[pop]["mae_rssi_db"] - mae_c[pop]["mae_rssi_db"])
        if (mae_d[pop]["mae_rssi_db"] is not None and mae_c[pop]["mae_rssi_db"] is not None)
        else None
        for pop in ("todos", "valido", "sentinela")
    }
    sinal_jensen = {
        pop: {
            "mae_medio_individual_B": mae_pop_b[pop]["media"],
            "mae_do_ensemble_de_B": mae_ensemble_b[pop]["mae_rssi_db"],
            "mae_C_referencia": mae_c[pop]["mae_rssi_db"],
            "ensemble_mais_proximo_de_C_que_individual": bool(
                mae_ensemble_b[pop]["mae_rssi_db"] is not None
                and mae_pop_b[pop]["media"] is not None
                and mae_c[pop]["mae_rssi_db"] is not None
                and abs(mae_ensemble_b[pop]["mae_rssi_db"] - mae_c[pop]["mae_rssi_db"])
                <= abs(mae_pop_b[pop]["media"] - mae_c[pop]["mae_rssi_db"])),
        }
        for pop in ("todos", "valido", "sentinela")
    }

    criterio_dp_gt_1e3 = {
        pop: bool((mae_pop_a[pop]["dp"] or 0.0) > 1e-3 or (mae_pop_b[pop]["dp"] or 0.0) > 1e-3)
        for pop in ("todos", "valido", "sentinela")
    }

    return {
        "n_nos_teste": n_p,
        "eval_batch_size": eval_bs,
        "dp_por_no_entre_ordens": {"grade_A_k8_nao_disjunta": std_a, "grade_B_k8_disjunta": std_b},
        "mae_rssi_por_populacao_por_ordem": {"grade_A": mae_pop_a, "grade_B": mae_pop_b},
        "mae_rssi_por_populacao_C_completa_disjunta": mae_c,
        "mae_rssi_por_populacao_D_completa_nao_disjunta": mae_d,
        "mae_rssi_por_populacao_ensemble_5ordens_B": mae_ensemble_b,
        "delta_penalidade_sorteio_B_menos_C": delta_sorteio,
        "delta_campo_receptivo_D_menos_C": delta_campo_receptivo,
        "sinal_jensen_ensemble_B_vs_individual_vs_C": sinal_jensen,
        "criterio_dp_maior_1e-3_dB_indica_efeito": criterio_dp_gt_1e3,
    }


def main() -> int:
    t0 = time.perf_counter()
    resultado = {
        "tarefa": "A0-ter item 4 (E3): inferencia nos checkpoints antigos (SO inferencia, "
                  "modelo ORIGINAL com PhysicsConstrainedDecoder, sem patch v3)",
        "parecer": "forum-eng-ia_fanout_capacidade.md, secao 'Regra de avaliacao (3)'",
        "checkpoints_usados": {
            cidade: {"run_dir": str(TREINOS_DIR / info["run_dir"]), "substituicao": info["substituicao"]}
            for cidade, info in CHECKPOINTS.items()
        },
        "escala_declarada": (
            f"janela de {MAX_NODES} nos (grafo Q1 completo tem 12.960.000 nos, nao cabe no "
            "teto de 25 min de GPU desta entrega); particao de teste desta janela e "
            "DIFERENTE da particao do run original (hash nao bate); normalizacao de "
            "distancia recalculada sobre o treino desta janela -- MAE absolutos NAO sao "
            "os de producao, validos so para a comparacao INTERNA entre as 12 avaliacoes "
            "de cada cidade (isolar sorteio e campo receptivo, conforme pedido)."),
        "grade_por_checkpoint": "A(loader original k=8/20 x5 ordens) + B(disjunto k=8/20 x5 ordens) "
                                 "+ C(disjunto, vizinhanca completa, 1x) + D(nao disjunto, "
                                 "vizinhanca completa, 1x) = 12 avaliacoes/cidade, 24 no total",
        "cidades": {},
    }
    for cidade, info in CHECKPOINTS.items():
        log(t0, f"=== {cidade} ===")
        mod, ctx, model, device, rec_original = carregar_ctx_e_modelo(cidade, info, t0)
        resultado["cidades"][cidade] = rodar_grade(cidade, mod, ctx, model, device, t0)
        del mod, ctx, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    out_path = Path(__file__).resolve().parent / "e3_inferencia_checkpoints_antigos.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    log(t0, f"gravado: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
