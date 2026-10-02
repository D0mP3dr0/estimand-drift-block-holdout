"""forum-eng-ia 26/09: o campo receptivo EFETIVO do GNN depende da composicao do batch.
Com fanout de 1 salto, so as sementes recebem arestas; mas se o vizinho terreno amostrado
de uma semente A e TAMBEM semente (B), B ja agregou os vizinhos dele na camada anterior e a
informacao de 2+ saltos chega a A. Mede, no MESMO dado do gate (Bauru Q2, 100k, split 42),
a fracao de arestas ET_TT amostradas cuja ORIGEM e semente, e a fracao de sementes com >=1
vizinho-semente, em tres regimes: treino shuffle=True bs=8192 (gate), treino shuffle=True
bs=184 (mesma densidade de sementes da producao: 24576/9.188.829 x 68.961), e avaliacao
(shuffle=False, bs=8192) nas particoes train e test. CPU, sem modelo."""
import json, sys, time
from pathlib import Path
import torch
MODELO_V3_DIR = Path(__file__).resolve().parent.parent / "modelo_v3"
sys.path.insert(0, str(MODELO_V3_DIR))
import v3_common as v3
from torch_geometric.loader import NeighborLoader
torch.manual_seed(0)
v3.patch_decoder_gnn()
mod = v3.carregar_modulo_congelado(v3.FROZEN_GNN_SCRIPT, "train_gnn_c0_spatial_coseed")
ctx = v3.carregar_base_e_particoes(
    mod=mod, graph_dir=v3.GRAPH_DIR_DEFAULT,
    rf_data_file="transfer_dataset_bauru_v19_Q2_enriched_v2.pt", graph_file="bauru_v19_Q2_gpu.pt",
    max_nodes=100_000, window_anchor="cobertura", grid_km=5.0, buffer_km=2.0,
    split_frac=(0.70, 0.15, 0.15), split_seed=42, smoke_geometria=True, mmap=True,
    precisa_arestas_ter_ter=True, log=lambda m: None)
nn_kw = {mod.ET_AT: [20], mod.ET_TT: [8], mod.ET_TA: [20]}
out = {"nn_kw": "ET_AT:[20], ET_TT:[8], ET_TA:[20]", "regimes": {}}
def medir(nome, part, bs, shuffle, max_batches=8):
    loc = torch.from_numpy(ctx.parts_local[part])
    g, _ = mod.induzir_particao(ctx.base, loc, ctx.n_antenna, lambda m: None, part)
    ld = NeighborLoader(data=g, num_neighbors=nn_kw, input_nodes=("terrain", None),
                        batch_size=bs, shuffle=shuffle, num_workers=0)
    tot_e = co_e = tot_s = co_s = nb = 0
    for b in ld:
        s = b["terrain"].batch_size
        ei = b[mod.ET_TT].edge_index
        dst_seed = ei[1] < s
        src_seed = ei[0] < s
        tot_e += int(dst_seed.sum()); co_e += int((dst_seed & src_seed).sum())
        tem = torch.zeros(s, dtype=torch.bool)
        m = dst_seed & src_seed & (ei[0] != ei[1])
        tem[ei[1][m]] = True
        tot_s += s; co_s += int(tem.sum()); nb += 1
        if nb >= max_batches: break
    out["regimes"][nome] = {"particao": part, "n_nos_particao": int(g["terrain"].num_nodes),
        "batch_size": bs, "shuffle": shuffle, "n_batches_medidos": nb,
        "frac_arestas_TT_com_origem_semente": co_e / max(tot_e, 1),
        "frac_sementes_com_vizinho_semente": co_s / max(tot_s, 1)}
    print(nome, json.dumps(out["regimes"][nome]))
medir("treino_shuffle_True_bs8192_gate", "train", 8192, True)
medir("treino_shuffle_True_bs184_densidade_producao", "train", 184, True, max_batches=40)
medir("avaliacao_shuffle_False_bs8192_particao_train", "train", 8192, False)
medir("avaliacao_shuffle_False_bs8192_particao_test", "test", 8192, False)
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
