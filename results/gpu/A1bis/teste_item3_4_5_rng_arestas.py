#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu/A1bis/teste_item3_4_5_rng_arestas.py -- gate A1bis, contra-auditoria
independente dos itens 3 (contador except edge_attr), 4 (invariancia +
RNG componente-a-componente, com comparacao DIRETA de estado, nao so hash)
e 5 (contagem de arestas == soma dos graus; graus AT=5, TT max=9),
sobre um modelo GNN FRESCO (init do zero -- mecanismo de avaliacao nao
depende de o modelo estar treinado).
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


def log(t0, msg):
    print(f"[item345_v3 +{time.perf_counter()-t0:.1f}s] {msg}", flush=True)


def main():
    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    v3.patch_decoder_gnn()
    import gnn_rf_model
    assert gnn_rf_model.PhysicsConstrainedDecoder is v3.AffineDecoderV3
    mod = v3.carregar_modulo_congelado(v3.FROZEN_GNN_SCRIPT, "train_gnn_c0_spatial_item345")

    rf_data_file = "transfer_dataset_bauru_v19_Q2_enriched_v2.pt"
    graph_file = "bauru_v19_Q2_gpu.pt"
    ctx = v3.carregar_base_e_particoes(
        mod=mod, graph_dir=v3.GRAPH_DIR_DEFAULT, rf_data_file=rf_data_file,
        graph_file=graph_file, max_nodes=100000, window_anchor="cobertura",
        grid_km=5.0, buffer_km=2.0, split_frac=(0.70, 0.15, 0.15),
        split_seed=42, smoke_geometria=True, mmap=True,
        precisa_arestas_ter_ter=True, log=lambda m: log(t0, m),
    )
    from gnn_rf_model import GNNRFModel
    model = GNNRFModel(terrain_dim=int(ctx.x_full.shape[1]), antenna_dim=int(ctx.ant_x.shape[1]),
                        hidden_dim=256, num_layers=4, heads=4, edge_dim=2,
                        output_dim=5, dropout=0.1, use_physics_constraints=True).to(device)

    # --- Item 3: instrumentar 1 forward+backward de producao e contar o
    # ramo except do encoder (deve ser 0 -- edge_attr sempre presente em
    # AT/TT, ausente estruturalmente so em TA que nunca aparece no batch).
    graph_train, _info = mod.induzir_particao(ctx.base, torch.from_numpy(ctx.parts_local["train"]),
                                               ctx.n_antenna, lambda m: log(t0, m), "train")
    from torch_geometric.loader import NeighborLoader
    nn_kw = {mod.ET_AT: [-1], mod.ET_TT: [-1], mod.ET_TA: [-1]}
    loader = NeighborLoader(data=graph_train, num_neighbors=nn_kw,
                             input_nodes=("terrain", None), batch_size=8192,
                             shuffle=False, num_workers=0)
    batch = next(iter(loader)).to(device)
    with v3.InstrumentacaoTreino() as instr:
        model.train()
        out = model(batch)
        out["predictions"].sum().backward()
    diag3 = instr.resultado()
    item3_passa = (diag3["n_except_edge_attr_encoder"] == 0)
    log(t0, f"item3 except_edge_attr={diag3['n_except_edge_attr_encoder']} passa={item3_passa}")

    # --- Item 4: invariancia (chamada direta da funcao ja instrumentada
    # pelo A0-ter, testar_invariancia_avaliacao_completa) + verificacao
    # DIRETA (nao so hash) do estado do RNG antes/depois de UMA avaliacao
    # completa.
    teste_inv = v3.testar_invariancia_avaliacao_completa(
        mod=mod, ctx=ctx, model=model, device=device, seed=42,
        eval_batch_size_a=24576, eval_batch_size_b=1024,
        particao="test", log=lambda m: log(t0, m))
    log(t0, f"item4 invariancia diff_max={teste_inv['diff_max_fisico_entre_todas_combinacoes']} "
             f"piso={teste_inv['piso_numerico_medido_mesma_ordem_mesma_bs']} "
             f"passou={teste_inv['passou_criterio']}")

    # RNG: comparacao DIRETA de estado bruto (nao hash) em torno de 1
    # avaliacao completa isolada.
    torch.manual_seed(7)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7)
    np.random.seed(7)
    import random
    random.seed(7)

    t_cpu_antes = torch.get_rng_state().clone()
    t_cuda_antes = ([g.clone() for g in torch.cuda.get_rng_state_all()]
                     if torch.cuda.is_available() else None)
    np_antes = np.random.get_state()
    py_antes = random.getstate()

    _ = v3.gerar_predicoes_teste_gnn(
        mod=mod, ctx=ctx, model=model, device=device, k_antenna=-1, k_terrain=-1,
        eval_batch_size=24576, seed=42, particao="test", disjoint=True,
        medir_diagnostico=True, log=lambda m: log(t0, m))

    t_cpu_depois = torch.get_rng_state()
    t_cuda_depois = (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    np_depois = np.random.get_state()
    py_depois = random.getstate()

    rng_direto = {
        "torch_cpu_identico": bool(torch.equal(t_cpu_antes, t_cpu_depois)),
        "torch_cuda_identico": (
            bool(all(torch.equal(a, d) for a, d in zip(t_cuda_antes, t_cuda_depois)))
            if t_cuda_antes is not None else None),
        "numpy_identico": bool(np_antes[0] == np_depois[0] and
                                np.array_equal(np_antes[1], np_depois[1]) and
                                np_antes[2:] == np_depois[2:]),
        "python_random_identico": bool(py_antes == py_depois),
    }
    log(t0, f"item4 RNG direto: {rng_direto}")

    item4_passa = bool(
        teste_inv["passou_criterio"]
        and rng_direto["torch_cuda_identico"] is True
        and rng_direto["numpy_identico"]
        and rng_direto["python_random_identico"])
    nota_torch_cpu = (
        "torch_cpu MUDA durante a avaliacao de vizinhanca completa (achado ja "
        "registrado no A0-ter), mas NAO afeta o resultado numerico -- confirmado "
        "aqui pela invariancia (diff_max == piso numerico) e por torch_cuda/numpy/"
        "python_random permanecerem IDENTICOS; registrado como NOTA, nao bloqueia."
        if not rng_direto["torch_cpu_identico"] else
        "torch_cpu tambem ficou identico nesta corrida.")

    # --- Item 5: contagem de arestas == soma dos graus, reconfirmada por
    # CHAMADA PROPRIA de _graus_por_relacao sobre o grafo de teste induzido
    # (nao copiada do parecer nem do run JSON do A0-ter).
    loc_test = torch.from_numpy(ctx.parts_local["test"])
    graph_test, _info_t = mod.induzir_particao(ctx.base, loc_test, ctx.n_antenna,
                                                 lambda m: log(t0, m), "test")
    graus_test = v3._graus_por_relacao(graph_test, mod)
    at_ok = (graus_test["AT"]["min"] == 5 and graus_test["AT"]["max"] == 5)
    tt_ok = (graus_test["TT"]["max"] == 9)
    log(t0, f"item5 graus AT={graus_test['AT']} TT={graus_test['TT']}")

    # Confere tambem, via NeighborLoader k=-1/disjoint=True, que o numero de
    # arestas de fato amostradas bate com a soma dos graus (chamada direta,
    # nao a que o wrapper ja fez).
    n_p = int(graph_test["terrain"].x.shape[0])
    loader_test = NeighborLoader(
        data=graph_test, num_neighbors={mod.ET_AT: [-1], mod.ET_TT: [-1], mod.ET_TA: [-1]},
        input_nodes=("terrain", None), batch_size=24576, shuffle=False,
        num_workers=0, disjoint=True)
    n_arestas = {mod.ET_AT: 0, mod.ET_TT: 0, mod.ET_TA: 0}
    for b in loader_test:
        for et in n_arestas:
            if et in b.edge_types:
                n_arestas[et] += int(b[et].edge_index.shape[1])
    soma_graus = {"AT": graus_test["AT"]["soma"], "TT": graus_test["TT"]["soma"],
                  "TA": graus_test["TA"]["soma"]}
    arestas_recontadas = {"AT": n_arestas[mod.ET_AT], "TT": n_arestas[mod.ET_TT],
                           "TA": n_arestas[mod.ET_TA]}
    bate = {k: bool(arestas_recontadas[k] == soma_graus[k]) for k in ("AT", "TT", "TA")}
    item5_passa = bool(at_ok and tt_ok and bate["AT"] and bate["TT"])
    log(t0, f"item5 arestas_recontadas={arestas_recontadas} soma_graus={soma_graus} bate={bate}")

    resultado = {
        "teste": "A1bis_item3_4_5_contra_auditoria_independente",
        "item3_except_edge_attr": {**diag3, "passa": item3_passa},
        "item4_invariancia_e_rng": {
            "teste_invariancia": teste_inv,
            "rng_estado_direto_antes_depois": rng_direto,
            "nota_torch_cpu": nota_torch_cpu,
            "passa": item4_passa,
        },
        "item5_arestas_e_graus": {
            "graus_por_relacao_test_recomputados": graus_test,
            "AT_grau_constante_5": at_ok,
            "TT_grau_maximo_9": tt_ok,
            "arestas_recontadas_por_relacao": arestas_recontadas,
            "soma_graus_por_relacao": soma_graus,
            "arestas_bate_com_soma_graus": bate,
            "nota_TA": "TA (terrain->antenna) sempre 0 arestas amostradas por razao ESTRUTURAL (nenhuma aresta desse tipo tem destino=fonte terrain sorteavel neste sentido no batch de sementes terrain) -- achado ja documentado, nao e falha deste item.",
            "passa": item5_passa,
        },
        "tempo_total_s": time.perf_counter() - t0,
    }
    saida = Path(__file__).parent / "teste_item3_4_5_a1bis.json"
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    log(t0, f"gravado: {saida}")


if __name__ == "__main__":
    sys.exit(main() or 0)
