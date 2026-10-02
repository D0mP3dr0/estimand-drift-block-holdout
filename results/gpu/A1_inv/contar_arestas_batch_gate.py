#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu/A1_inv/contar_arestas_batch_gate.py -- item (b), passo 1.

Reproduz EXATAMENTE a construcao do batch usado pelo gate A1
(gpu/A1/teste_abce_overfit_v3.py --tipo gnn): mesma carga de dado real
(Bauru Q2, --mmap, --max-nodes 100000), mesmo split_seed=42, mesmo
NeighborLoader (nn_kw = {ET_AT: [20], ET_TT: [8], ET_TA: [20]},
batch_size=8192, shuffle=False, input_nodes=("terrain", None)), e conta:

  1. arestas por tipo no GRAFO INDUZIDO da particao de treino inteira
     (graph_train, antes de qualquer sampling do NeighborLoader);
  2. arestas por tipo no PRIMEIRO batch amostrado pelo loader (o MESMO
     `batch = next(iter(loader))` que o gate usa para o overfit de 300
     passos);
  3. se as features de antena (ctx.ant_x) chegam nao-zeradas ao encoder.

Edge type do parametro morto (achado A1-b-1): ET_TA = mod.ET_TA =
("terrain", "in_range_of", "antenna") -- nome de modulo PyG
"terrain___in_range_of___antenna", SAGEConv(lin_l=aggr. de vizinhos
terrain, lin_r=self antenna) em GNN_RF_V2/02_models/gnn_rf_encoder.py:140.

CPU-only por desenho (nao precisa de GPU para contar arestas; poupa o
teto de 20 min de GPU do protocolo para o script 2, que precisa treinar).
Nao editar nada em modelo_v3/ nem nos congelados -- so leitura.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

MODELO_V3_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/"
                      "_v3_2026-09-25/gpu/modelo_v3")
sys.path.insert(0, str(MODELO_V3_DIR))
import v3_common as v3  # noqa: E402


def log(t0, msg):
    print(f"[contar_arestas +{time.perf_counter()-t0:.1f}s] {msg}", flush=True)


def main():
    t0 = time.perf_counter()
    cidade, quadrante = "bauru", "Q2"
    max_nodes = 100_000
    batch_size = 8192

    rf_data_file = f"transfer_dataset_{cidade}_v19_{quadrante}_enriched_v2.pt"
    graph_file = f"{cidade}_v19_{quadrante}_gpu.pt"

    v3.patch_decoder_gnn()
    import gnn_rf_model  # noqa: F401 (so para confirmar patch, nao instanciamos modelo aqui)
    mod = v3.carregar_modulo_congelado(v3.FROZEN_GNN_SCRIPT, "train_gnn_c0_spatial_contagem")

    log(t0, f"carregando dado real ({cidade} {quadrante}, max_nodes={max_nodes})...")
    ctx = v3.carregar_base_e_particoes(
        mod=mod, graph_dir=v3.GRAPH_DIR_DEFAULT, rf_data_file=rf_data_file, graph_file=graph_file,
        max_nodes=max_nodes, window_anchor="cobertura",
        grid_km=5.0, buffer_km=2.0, split_frac=(0.70, 0.15, 0.15),
        split_seed=42, smoke_geometria=True, mmap=True,
        precisa_arestas_ter_ter=True,  # GNN usa ET_TT tambem
        log=lambda m: log(t0, m),
    )
    log(t0, f"dado carregado: n_ter_total={ctx.n_ter_total} janela={ctx.n_ter} "
             f"n_antenna={ctx.n_antenna}")

    # -- 1. arestas por tipo no grafo COMPLETO da janela de 100k nos (ctx.base) --
    ET_AT, ET_TA, ET_TT = mod.ET_AT, mod.ET_TA, mod.ET_TT
    arestas_janela_completa = {
        "___".join(ET_AT): int(ctx.base[ET_AT].edge_index.shape[1]),
        "___".join(ET_TA): int(ctx.base[ET_TA].edge_index.shape[1]),
        "___".join(ET_TT): int(ctx.base[ET_TT].edge_index.shape[1]) if ET_TT in ctx.base.edge_types else 0,
    }

    # -- 2. grafo INDUZIDO da particao de treino (mesma chamada do gate) --
    tr_loc_np = ctx.parts_local["train"]
    tr_loc = torch.from_numpy(tr_loc_np)
    graph_train, info_train = mod.induzir_particao(
        ctx.base, tr_loc, ctx.n_antenna, lambda m: log(t0, m), "train")
    arestas_graph_train = {
        "___".join(ET_AT): int(graph_train[ET_AT].edge_index.shape[1]),
        "___".join(ET_TA): int(graph_train[ET_TA].edge_index.shape[1]),
        "___".join(ET_TT): int(graph_train[ET_TT].edge_index.shape[1]) if ET_TT in graph_train.edge_types else 0,
    }
    log(t0, f"graph_train arestas: {arestas_graph_train}")

    # -- 3. o MESMO NeighborLoader e o MESMO primeiro batch do gate --
    nn_kw = {mod.ET_AT: [20], mod.ET_TT: [8], mod.ET_TA: [20]}
    from torch_geometric.loader import NeighborLoader
    loader = NeighborLoader(data=graph_train, num_neighbors=nn_kw,
                             input_nodes=("terrain", None), batch_size=batch_size,
                             shuffle=False, num_workers=0)
    batch = next(iter(loader))
    bs = batch["terrain"].batch_size
    n_terrain_no_batch = int(batch["terrain"].num_nodes)
    n_antenna_no_batch = int(batch["antenna"].num_nodes) if "antenna" in batch.node_types else 0
    arestas_no_batch = {
        "___".join(ET_AT): int(batch[ET_AT].edge_index.shape[1]) if ET_AT in batch.edge_types else 0,
        "___".join(ET_TA): int(batch[ET_TA].edge_index.shape[1]) if ET_TA in batch.edge_types else 0,
        "___".join(ET_TT): int(batch[ET_TT].edge_index.shape[1]) if ET_TT in batch.edge_types else 0,
    }
    log(t0, f"batch (bs_sementes={bs}, n_terrain={n_terrain_no_batch}, "
             f"n_antenna={n_antenna_no_batch}) arestas: {arestas_no_batch}")

    # quantos nos de antena DISTINTOS aparecem como DESTINO em ET_TA dentro
    # do batch (isto e o que decide se lin_l tem algo != 0 para agregar)
    if ET_TA in batch.edge_types and int(batch[ET_TA].edge_index.shape[1]) > 0:
        dst_antena_com_aresta_ta = int(batch[ET_TA].edge_index[1].unique().numel())
    else:
        dst_antena_com_aresta_ta = 0

    # -- feature de antena chega nao-zerada? (ctx.ant_x, features cruas) --
    ant_x = ctx.ant_x
    features_antena = {
        "shape": list(ant_x.shape),
        "todas_zero": bool(torch.all(ant_x == 0.0)),
        "fracao_linhas_todas_zero": float((ant_x.abs().sum(dim=1) == 0).float().mean()),
        "norma_media_por_linha": float(ant_x.abs().sum(dim=1).mean()),
    }
    if "antenna" in batch.node_types and hasattr(batch["antenna"], "x"):
        ant_x_batch = batch["antenna"].x
        features_antena["no_batch_shape"] = list(ant_x_batch.shape)
        features_antena["no_batch_todas_zero"] = bool(torch.all(ant_x_batch == 0.0)) if ant_x_batch.numel() else True

    resultado = {
        "item": "A1-b, passo 1: contagem de arestas no batch do gate (reproducao fiel)",
        "reproduz": "gpu/A1/teste_abce_overfit_v3.py --tipo gnn (mesma carga/split/loader/batch, sem treinar)",
        "edge_type_do_parametro_morto": {
            "nome_no_encoder": "encoder.convs.{0,1,2}.convs.<terrain___in_range_of___antenna>.lin_l.weight",
            "edge_type_tupla": list(mod.ET_TA),
            "conv_class": "SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim, aggr='mean')",
            "fonte_conv": "GNN_RF_V2/02_models/gnn_rf_encoder.py:140-144",
            "semantica": ("lin_l atua sobre a agregacao MEAN dos vizinhos 'terrain' que apontam "
                          "para cada 'antenna' (dst=antenna); lin_r atua sobre o embedding proprio "
                          "da antena (self/root). Se NENHUMA aresta terrain->antenna cai no "
                          "subgrafo amostrado, a agregacao e o vetor zero para TODAS as antenas "
                          "do batch, e lin_l nunca recebe gradiente (matriz constante x zero)."),
        },
        "cidade": cidade, "quadrante": quadrante, "max_nodes": max_nodes,
        "split_seed": 42, "batch_size_pedido": batch_size, "batch_size_real_bs_sementes": bs,
        "arestas_janela_completa_100k": arestas_janela_completa,
        "arestas_graph_train_induzido": arestas_graph_train,
        "arestas_no_batch_amostrado_pelo_neighborloader": arestas_no_batch,
        "n_antenas_destino_distintas_com_aresta_ta_no_batch": dst_antena_com_aresta_ta,
        "hipotese_do_chefe_confirmada": (arestas_no_batch["___".join(ET_TA)] == 0),
        "features_antena_ctx_completo": features_antena,
        "tempo_total_s": time.perf_counter() - t0,
    }

    saida = Path(__file__).resolve().parent / "contagem_arestas_batch_gate.json"
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    log(t0, f"gravado: {saida}")
    log(t0, f"ET_TA no batch do gate = {arestas_no_batch['___'.join(ET_TA)]} arestas "
             f"(hipotese confirmada={resultado['hipotese_do_chefe_confirmada']}); "
             f"ET_TA no grafo de treino INTEIRO (100k) = {arestas_graph_train['___'.join(ET_TA)]}")


if __name__ == "__main__":
    sys.exit(main() or 0)
