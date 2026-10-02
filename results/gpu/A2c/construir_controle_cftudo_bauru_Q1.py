#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Constroi gpu/A2c/controle_cftudo_bauru_Q1.json (item (2) da entrega
oficina-pipeline, ERRATA DE PROVENIENCIA): le os 2 run JSONs + 2 .npz de
predicoes do CONTROLE (GNN e MLP, Bauru Q1, cftudo, split_seed 42, seed 42,
8 epocas, bs 12288, k=-1, disjoint) recem-rodados em gpu/A2c/, recomputa
MAE de RSSI por populacao (valido/sentinela/todos) via
v3_common.mae_rssi_por_populacao (mesma funcao usada pelo E3), compara com
o run publicado (run_c0c1cf_bauru_s42_Q1_g10b2.json,
selecao.test_no_melhor_ckpt.mae_rssi_db) e com o MLP antigo
(run_mlpcf_bauru_s42_Q1_g10b2.json). Nao interpreta se o resultado e "bom"
(isso e do rigor) -- so recomputa e grava os numeros dos artefatos reais.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

MODELO_V3_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/gpu/modelo_v3")
sys.path.insert(0, str(MODELO_V3_DIR))
import v3_common as v3  # noqa: E402

A2C_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/gpu/A2c")
DADOS_TREINOS_C1 = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/FIRST_RESPONSE_REVIEW_IEEE_ACESSES/"
    "EVIDENCIA_RESUBMISSAO/dados/treinos_c1")
RUN_PUBLICADO_GNN = DADOS_TREINOS_C1 / "run_c0c1cf_bauru_s42_Q1_g10b2.json"
RUN_ANTIGO_MLP = DADOS_TREINOS_C1 / "run_mlpcf_bauru_s42_Q1_g10b2.json"


def carregar_npz_e_mae(run_label: str) -> dict:
    run_dir = A2C_DIR / run_label
    run_json_path = run_dir / f"run_{run_label}.json"
    npz_path = run_dir / f"predicoes_{run_label}.npz"
    with open(run_json_path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    npz = np.load(npz_path)
    pred, target, sentinela = npz["pred"], npz["target"], npz["sentinela"]
    mae_pop = v3.mae_rssi_por_populacao(pred, target, sentinela)
    return {
        "run_label": run_label,
        "run_json": str(run_json_path),
        "npz": str(npz_path),
        "insumos": rec.get("insumos"),
        "melhor_epoca": (rec.get("selecao") or {}).get("melhor_epoca"),
        "custo_tempo_total_s": (rec.get("custo") or {}).get("tempo_total_s"),
        "pred_rssi_min": float(np.min(pred[:, 3])),
        "pred_rssi_max": float(np.max(pred[:, 3])),
        "n_test_nos": int(pred.shape[0]),
        "mae_rssi_por_populacao": mae_pop,
        "curva_val_por_epoca_mae_rssi_db": [
            (e.get("val") or {}).get("mae_rssi_db") for e in (rec.get("epocas") or [])],
        "diagnostico_saida_canal3_test": (rec.get("diagnostico_saida", {}).get("test", {}) or {}).get("canal_3"),
    }


def carregar_referencia(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    sel = d.get("selecao", {}) or {}
    t = sel.get("test_no_melhor_ckpt", {}) or {}
    return {
        "arquivo": str(path),
        "melhor_epoca": sel.get("melhor_epoca"),
        "mae_rssi_db_agregado_todos": t.get("mae_rssi_db"),
        "rmse_rssi_db_agregado_todos": t.get("rmse_rssi_db"),
        "mae_pl_db_agregado_todos": t.get("mae_pl_db"),
        "nota": ("run original NAO reporta MAE por populacao (valido/sentinela); "
                 "so o agregado 'todos' esta disponivel para comparacao direta."),
    }


def main() -> int:
    gnn = carregar_npz_e_mae("gnn_v3_controle_cftudo_bauru_Q1")
    mlp = carregar_npz_e_mae("mlp_v3_controle_cftudo_bauru_Q1")
    ref_gnn_publicado = carregar_referencia(RUN_PUBLICADO_GNN)
    ref_mlp_antigo = carregar_referencia(RUN_ANTIGO_MLP)

    def _sentinela_ok(bloco):
        v = bloco["mae_rssi_por_populacao"]["sentinela"]["mae_rssi_db"]
        return (v is not None) and (v < 1.0)

    def _validos_ok(bloco):
        v = bloco["mae_rssi_por_populacao"]["valido"]["mae_rssi_db"]
        return (v is not None) and (v < 5.0)

    razoavel = bool(_sentinela_ok(gnn) and _validos_ok(gnn) and _sentinela_ok(mlp) and _validos_ok(mlp))

    saida = {
        "id": "controle_cftudo_bauru_Q1",
        "gravado_em_utc": "2026-09-26",
        "autor": "oficina-pipeline",
        "protocolo": "PROTOCOLO_GPU_v3_2026-09-25.md (ERRATA DE PROVENIENCIA, item (2) das entregas)",
        "config": {
            "cidade": "bauru", "quadrante": "Q1", "grafo": "inteiro (max_nodes=0, --mmap)",
            "dataset": "cftudo (transfer_dataset_bauru_v19_Q1_enriched_cftudo.pt, canonico)",
            "split_seed": 42, "seed_treino": 42, "epochs": 8, "batch_size": 12288,
            "k_antenna": -1, "k_terrain": -1, "disjoint": True,
        },
        "criterio_lancamento_A2c": (
            "GNN e MLP com MAE_rssi sentinela < 1 dB E MAE_rssi validos < 5 dB "
            "(item (3) da entrega); PASS = lanca A2c, FAIL = para e diagnostica."),
        "criterio_razoavel_gnn": {"sentinela_menor_1dB": _sentinela_ok(gnn), "validos_menor_5dB": _validos_ok(gnn)},
        "criterio_razoavel_mlp": {"sentinela_menor_1dB": _sentinela_ok(mlp), "validos_menor_5dB": _validos_ok(mlp)},
        "controle_razoavel": razoavel,
        "gnn": gnn,
        "mlp": mlp,
        "referencia_gnn_publicado_run_c0c1cf_bauru_s42_Q1_g10b2": ref_gnn_publicado,
        "referencia_mlp_antigo_run_mlpcf_bauru_s42_Q1_g10b2": ref_mlp_antigo,
        "comparacao_agregada_todos": {
            "gnn_controle_mae_rssi_todos_db": gnn["mae_rssi_por_populacao"]["todos"]["mae_rssi_db"],
            "gnn_publicado_mae_rssi_todos_db": ref_gnn_publicado["mae_rssi_db_agregado_todos"],
            "mlp_controle_mae_rssi_todos_db": mlp["mae_rssi_por_populacao"]["todos"]["mae_rssi_db"],
            "mlp_antigo_mae_rssi_todos_db": ref_mlp_antigo["mae_rssi_db_agregado_todos"],
        },
        "pergunta_saida": "o MLP em cftudo sai do colapso (sentinela ~ 0,1-0,3 dB)?",
        "resposta_pergunta_saida": {
            "mlp_sentinela_mae_rssi_db": mlp["mae_rssi_por_populacao"]["sentinela"]["mae_rssi_db"],
            "dentro_da_faixa_0_1_a_0_3": (
                mlp["mae_rssi_por_populacao"]["sentinela"]["mae_rssi_db"] is not None
                and 0.1 <= mlp["mae_rssi_por_populacao"]["sentinela"]["mae_rssi_db"] <= 0.3),
        },
        "nota_gate_A1bis": (
            "O gate A1-bis (overfit 1 batch, gradiente vivo) foi feito em `_v2` (Bauru Q2), "
            "dataset PRE-correcao. Overfit/gradiente sao propriedades do OTIMIZADOR/arquitetura "
            "e NAO dependem do alvo/dataset -- continuam validos. O DIAGNOSTICO DE SAIDA "
            "(diagnostico_saida_canais, mae_rssi_por_populacao, sentinela) DEPENDE do alvo "
            "e portanto PRECISA ser refeito em cftudo -- e o que este controle faz."),
        "nota_disco": "GPU: 8,4-8,5 GB livres antes/depois de cada corrida (motor de embeddings pid 1630 intocado, ~5,3-5,4 GB).",
    }
    out_path = A2C_DIR / "controle_cftudo_bauru_Q1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    print(f"gravado: {out_path}")
    print(f"controle_razoavel = {razoavel}")
    print(json.dumps(saida["gnn"]["mae_rssi_por_populacao"], indent=2))
    print(json.dumps(saida["mlp"]["mae_rssi_por_populacao"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
