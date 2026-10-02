#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu/modelo_v3/v3_common.py

Utilitarios COMPARTILHADOS por `train_gnn_v3.py` e `train_mlp_v3.py`.
ARQUIVO NOVO, nao edita nenhum script congelado.

Duas responsabilidades:

1. `carregar_modulo_congelado()` + `patch_decoder_*()`: importam por
   importlib os scripts congelados (`train_gnn_c0_spatial.py` /
   `train_mlp_c0_spatial.py`) e trocam a CLASSE do decodificador por
   `AffineDecoderV3` (rf_decoder_v3.py) SEM editar uma linha dos
   congelados -- so substituem o atributo de modulo que
   `GNNRFModel.__init__`/`MLPRFModel(...)`/`main()` resolvem em tempo de
   chamada (nomes globais em Python sao resolvidos no `__dict__` do modulo
   a cada execucao, nao no momento em que a funcao foi definida).

2. `reconstruir_particao_teste_*()`: depois que o congelado ja treinou e
   gravou `checkpoint_best.pt` (via `mod.main(argv)`, chamado pelo
   wrapper), reconstroi -- reusando as MESMAS funcoes do congelado
   (`janela_contigua`, `split_espacial_3vias`, `induzir_particao`,
   `latlon_graus_para_metros`, `_scatter_por_semente`) -- a particao de
   TESTE, recarrega o checkpoint num modelo com o mesmo decoder V3, roda
   forward em eval() e devolve idx_global/target/pred para salvar em
   `.npz` no formato de `_R2_2026-09-24/e3_predicoes` (idx_global, target,
   pred, sentinela).

Nenhuma funcao aqui reimplementa logica de treino, split ou perda: tudo
que existe no congelado e CHAMADO por nome (mod.<funcao>), nunca copiado.
"""
from __future__ import annotations

import datetime
import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import bipartite_subgraph, subgraph

GNN_RF_V2 = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2")
# ERRATA DE PROVENIENCIA (chefe, 26/09/2026, PROTOCOLO_GPU_v3_2026-09-25.md):
# GRAPH_DIR_V2_LEGADO e o dataset PRE-correcao do alvo (`_enriched_v2.pt`,
# copiado do nome usado no smoke do gate G0) -- NUNCA usar como default de
# novo. GRAPH_DIR_DEFAULT (canonico) e o mesmo `graph_dir` de
# `run_c0c1cf_bauru_s42_Q1_g10b2.json` (`config.graph_dir`/`config.rf_data_file`),
# com sha256 das 16 celulas `enriched_cftudo.pt` no manifest v4 (grupo
# `tensores_cftudo`).
GRAPH_DIR_V2_LEGADO = GNN_RF_V2 / "graph_data"
GRAPH_DIR_DEFAULT = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")

MANIFEST_V4_PATH = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/manifest_mathematics_v4.jsonl")

FROZEN_SCRIPTS_DIR = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
    "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/scripts")
FROZEN_GNN_SCRIPT = FROZEN_SCRIPTS_DIR / "train_gnn_c0_spatial.py"
FROZEN_MLP_SCRIPT = FROZEN_SCRIPTS_DIR / "train_mlp_c0_spatial.py"

MODELO_V3_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODELO_V3_DIR))
from rf_decoder_v3 import AffineDecoderV3, sha256_do_arquivo  # noqa: E402


def sha256_arquivo(caminho: Path, bloco: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while True:
            chunk = f.read(bloco)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _carregar_entradas_manifest_v4(grupo: str = "tensores_cftudo",
                                    manifest_path: Path = MANIFEST_V4_PATH) -> list:
    """Le `manifest_mathematics_v4.jsonl` (JSON Lines) e devolve so as
    entradas do `grupo` pedido (default `tensores_cftudo`: as 16 celulas do
    dataset canonico `enriched_cftudo.pt`, unico grupo do manifest v4 que
    cobre esses arquivos)."""
    entradas = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            d = json.loads(linha)
            if d.get("grupo") == grupo:
                entradas.append(d)
    return entradas


def sha256_com_reuso_manifest(path: Path, entradas_manifest: list) -> tuple:
    """Devolve `(sha256, metodo)`. Metodo declarado (ERRATA DE PROVENIENCIA,
    item 1 das entregas): reusa o `sha256` JA GRAVADO na entrada do manifest
    v4 cujo `caminho` resolve para o MESMO arquivo SE E SOMENTE SE
    `tamanho_bytes` e `mtime` (segundo-a-segundo, isoformat UTC) do `stat()`
    ATUAL do arquivo forem IDENTICOS aos gravados no manifest -- prova de
    que o arquivo nao foi tocado desde que o manifest foi gerado. Caso
    contrario (arquivo nao esta no manifest, ou mtime/tamanho mudaram),
    RECALCULA por leitura completa (`sha256_arquivo`, streaming, sem
    carregar o arquivo inteiro em memoria)."""
    path = Path(path).resolve()
    st = path.stat()
    mtime_iso = datetime.datetime.fromtimestamp(
        st.st_mtime, tz=datetime.timezone.utc).isoformat()
    for entrada in entradas_manifest:
        cam = entrada.get("caminho", "")
        try:
            bate_caminho = Path(cam).resolve() == path
        except OSError:
            bate_caminho = False
        if bate_caminho:
            if (entrada.get("tamanho_bytes") == st.st_size
                    and entrada.get("mtime") == mtime_iso
                    and entrada.get("sha256")):
                return entrada["sha256"], "reuso_manifest_v4_mtime_e_tamanho_identicos"
            break
    return sha256_arquivo(path), "recomputado_sha256_arquivo_stream"


def verificar_proveniencia_dataset(rf_data_path: Path, graph_path: Path,
                                    manifest_path: Path = MANIFEST_V4_PATH,
                                    grupo: str = "tensores_cftudo") -> dict:
    """ERRATA DE PROVENIENCIA (chefe, 26/09/2026): calcula/reusa o sha256 de
    `rf_data_path` (dataset enriquecido, `enriched_cftudo.pt`) e de
    `graph_path` (`_gpu.pt`, arestas terreno-terreno) e verifica o de
    `rf_data_path` CONTRA o manifest v4 (grupo `tensores_cftudo`, as UNICAS
    16 entradas -- uma por celula -- que cobrem esse `.pt`; o `_gpu.pt` NAO
    tem entrada propria no manifest v4, entao seu sha256 e gravado por
    proveniencia mas NAO participa do gate). O chamador (`train_gnn_v3.py`/
    `train_mlp_v3.py`) deve ABORTAR (rc=3) se
    `sha256_rf_data_bate_manifest_v4` for False, ANTES de gastar GPU."""
    entradas = _carregar_entradas_manifest_v4(grupo=grupo, manifest_path=manifest_path)
    sha_rf, metodo_rf = sha256_com_reuso_manifest(rf_data_path, entradas)
    sha_graph, metodo_graph = sha256_com_reuso_manifest(graph_path, entradas)

    rf_path_resolved = Path(rf_data_path).resolve()
    entrada_rf = None
    for e in entradas:
        try:
            if Path(e.get("caminho", "")).resolve() == rf_path_resolved:
                entrada_rf = e
                break
        except OSError:
            continue
    encontrado_no_manifest = entrada_rf is not None
    sha_manifest = entrada_rf.get("sha256") if entrada_rf else None
    bate = bool(encontrado_no_manifest and sha_manifest == sha_rf)
    return {
        "sha256_rf_data": sha_rf,
        "metodo_sha_rf_data": metodo_rf,
        "sha256_graph": sha_graph,
        "metodo_sha_graph": metodo_graph,
        "manifest": "v4",
        "manifest_path": str(manifest_path),
        "grupo_manifest": grupo,
        "encontrado_no_manifest_v4": encontrado_no_manifest,
        "sha256_manifest_v4": sha_manifest,
        "sha256_rf_data_bate_manifest_v4": bate,
        "nota_graph_file": ("_gpu.pt (arestas terreno-terreno) nao tem entrada propria no "
                             "manifest v4/grupo tensores_cftudo (so os 16 .pt de dado "
                             "canonico enriched_cftudo tem); sha256 gravado por "
                             "proveniencia, sem gate sobre ele."),
    }


def carregar_modulo_congelado(script_path: Path, nome_modulo: str) -> ModuleType:
    """Importa o script congelado por caminho (importlib), SEM rodar seu
    `if __name__ == "__main__"` (o `mod.__name__` fica `nome_modulo`, nao
    `"__main__"`). So define funcoes/classes; nao treina nada aqui."""
    spec = importlib.util.spec_from_file_location(nome_modulo, script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nome_modulo] = mod
    spec.loader.exec_module(mod)
    return mod


def patch_decoder_gnn(base_dir: Path = GNN_RF_V2) -> None:
    """
    GNNRFModel.__init__ (02_models/gnn_rf_model.py:70-76) referencia o
    nome global `PhysicsConstrainedDecoder`, resolvido no `__dict__` do
    MODULO `gnn_rf_model` a cada chamada -- nao no momento em que a classe
    foi definida. Basta importar `gnn_rf_model` (executa uma vez, se ainda
    nao importado) e SOBRESCREVER esse atributo de modulo; qualquer
    `GNNRFModel(..., use_physics_constraints=True)` instanciado DEPOIS
    deste patch usa `AffineDecoderV3`.
    """
    p = str(base_dir / "02_models")
    if p not in sys.path:
        sys.path.insert(0, p)
    import gnn_rf_model  # import (cacheado em sys.modules se ja importado)
    gnn_rf_model.PhysicsConstrainedDecoder = AffineDecoderV3
    # tambem no modulo rf_decoder, por robustez/paridade com o MLP (harmless).
    import rf_decoder
    rf_decoder.PhysicsConstrainedDecoder = AffineDecoderV3


def patch_decoder_mlp(base_dir: Path = GNN_RF_V2) -> None:
    """
    `train_mlp_c0_spatial.py:main()` faz
    `from rf_decoder import PhysicsConstrainedDecoder` DENTRO da funcao
    (linha ~977) -- e uma resolucao TARDIA (em tempo de chamada de
    `main()`), que busca o atributo `PhysicsConstrainedDecoder` no modulo
    `rf_decoder` (via `sys.modules['rf_decoder']`) no momento em que
    `main()` roda. Patchear o atributo do modulo `rf_decoder` ANTES de
    chamar `mod.main(argv)` e suficiente.
    """
    p = str(base_dir / "02_models")
    if p not in sys.path:
        sys.path.insert(0, p)
    import rf_decoder
    rf_decoder.PhysicsConstrainedDecoder = AffineDecoderV3


# --------------------------------------------------------------------------
# Reconstrucao POS-TREINO da particao de teste (para salvar .npz de
# predicoes), reusando as funcoes do proprio modulo congelado.
# --------------------------------------------------------------------------

def _log_stub(msg: str) -> None:
    print(f"[v3_common] {msg}", flush=True)


def carregar_base_e_particoes(
    mod: ModuleType,
    graph_dir: Path,
    rf_data_file: str,
    graph_file: str,
    max_nodes: int,
    window_anchor: str,
    grid_km: float,
    buffer_km: float,
    split_frac: tuple,
    split_seed: int,
    smoke_geometria: bool,
    mmap: bool,
    precisa_arestas_ter_ter: bool,
    log=_log_stub,
) -> SimpleNamespace:
    """
    Reproduz -- chamando SO funcoes do modulo congelado `mod` -- a mesma
    sequencia que `mod.main()` executa ate obter `parts_local` (train/val/
    test em indices LOCAIS DA JANELA) e `x_full` (features + coluna de
    distancia normalizada pelo TREINO). Usada depois do treino, para
    reconstruir exatamente a particao de teste do MESMO run (mesmos seeds
    e mesmos argumentos), sem duplicar a logica de carga/janela/split.

    `precisa_arestas_ter_ter=False` pula a carga do `graph_file` (~15 GB,
    so terreno<->terreno) quando o chamador (MLP) nao usa nenhuma aresta.
    """
    ET_AT = mod.ET_AT
    ET_TT = mod.ET_TT
    PL_TARGET_MAX_VALID = mod.PL_TARGET_MAX_VALID

    sys.path.append(str(GNN_RF_V2 / "03_training"))
    from spatial_cv import SpatialKFold  # noqa: E402

    rf_path = graph_dir / rf_data_file
    struct_path = graph_dir / graph_file
    load_kw = dict(map_location="cpu", weights_only=False)
    t0 = time.perf_counter()
    rf_data = torch.load(rf_path, mmap=mmap, **load_kw)
    log(f"[+{time.perf_counter()-t0:.1f}s] rf_data carregado (mmap={mmap})")

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
    ant_pos = (rf_data["antenna"].pos.float()
               if hasattr(rf_data["antenna"], "pos") else ant_x[:, :2])

    ei_at_full = rf_data[ET_AT].edge_index.long()
    ea_at_full = rf_data[ET_AT].edge_attr.float() if hasattr(rf_data[ET_AT], "edge_attr") else None

    ei_tt_full = ea_tt_full = None
    if precisa_arestas_ter_ter:
        tt_in_rf = ET_TT in rf_data.edge_types
        if tt_in_rf:
            ei_tt_full = rf_data[ET_TT].edge_index.long()
            ea_tt_full = rf_data[ET_TT].edge_attr.float() if hasattr(rf_data[ET_TT], "edge_attr") else None
        else:
            sd = torch.load(struct_path, **load_kw)
            t2t = sd["dem", "adjacent_to", "dem"]
            ei_tt_full = t2t.edge_index.long().clone()
            ea_tt_full = (t2t.edge_attr.float().clone()
                          if hasattr(t2t, "edge_attr") and t2t.edge_attr is not None else None)
            del sd, t2t
            gc.collect()

    del rf_data
    gc.collect()
    log(f"[+{time.perf_counter()-t0:.1f}s] grafo completo: {n_ter_total:,} terrain")

    pos_m_full, geo = mod.latlon_graus_para_metros(pos_deg)
    ancora = (tgt_all[:, 0] < PL_TARGET_MAX_VALID).numpy()
    if max_nodes and max_nodes > 0:
        g_idx = mod.janela_contigua(pos_m_full, max_nodes, ancora)
    else:
        g_idx = np.arange(n_ter_total)
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
        if precisa_arestas_ter_ter:
            ei_tt_full, ea_tt_full = subgraph(
                g_idx_t, ei_tt_full, ea_tt_full, relabel_nodes=True, num_nodes=n_ter_total)
    base[ET_AT].edge_index = ei_at_full
    if ea_at_full is not None:
        base[ET_AT].edge_attr = ea_at_full
    base[mod.ET_TA].edge_index = ei_at_full[[1, 0]]
    if precisa_arestas_ter_ter:
        base[ET_TT].edge_index = ei_tt_full
        base[ET_TT].edge_attr = (ea_tt_full if ea_tt_full is not None
                                 else torch.zeros((ei_tt_full.shape[1], 2), dtype=torch.float32))

    n_ter = int(x_base.shape[0])
    pos_m = pos_m_full[g_idx]
    dist_all = (dist_pre[g_idx_t] if dist_pre is not None else None)
    del tgt_all, pos_deg
    gc.collect()

    ext = min((pos_m[:, 0].max() - pos_m[:, 0].min()) / 1000.0,
              (pos_m[:, 1].max() - pos_m[:, 1].min()) / 1000.0)
    if smoke_geometria and ext / max(grid_km, 1e-9) < 6.0:
        novo = float(ext / 6.0)
        buffer_km = novo * (buffer_km / grid_km)
        grid_km = novo
        log(f"[SMOKE] geometria ajustada: grid={grid_km:.3f}km buffer={buffer_km:.3f}km")

    parts_local, split_info = mod.split_espacial_3vias(
        pos_m, grid_km, buffer_km, split_frac, split_seed, SpatialKFold, log)
    for k, v in parts_local.items():
        if len(v) == 0:
            raise RuntimeError(f"Particao '{k}' vazia apos buffer (grid={grid_km}, buffer={buffer_km})")

    tr_loc = torch.from_numpy(parts_local["train"])
    d_mean_tr = float(dist_all[tr_loc].mean())
    d_std_tr = max(float(dist_all[tr_loc].std()), 1.0)
    x_full = torch.cat([x_base, ((dist_all - d_mean_tr) / d_std_tr).unsqueeze(1)], dim=1)
    base["terrain"].x = x_full

    return SimpleNamespace(
        base=base, g_idx=g_idx, parts_local=parts_local, split_info=split_info,
        x_full=x_full, ant_x=ant_x, n_antenna=n_antenna, dist_all=dist_all,
        pos_m=pos_m, n_ter=n_ter, n_ter_total=n_ter_total, subamostrado=subamostrado,
        grid_km=grid_km, buffer_km=buffer_km, geo=geo,
    )


def _hash_rng_state() -> dict:
    """A0-ter item 1: hash do estado do RNG GLOBAL (torch CPU, torch CUDA,
    numpy, python random). Usado para verificar que uma avaliacao com
    vizinhanca COMPLETA (num_neighbors=-1) nao consome/mistura o gerador
    global -- ao contrario da amostragem por grau > k, que consome o RNG
    sequencialmente (achado do A0-bis, ver testar_invariancia_disjoint_gnn).
    Hash (nao o estado bruto) para manter o run JSON pequeno."""
    import random as _random
    h: dict = {}
    t_cpu = torch.get_rng_state()
    h["torch_cpu"] = hashlib.sha256(t_cpu.numpy().tobytes()).hexdigest()
    if torch.cuda.is_available():
        h["torch_cuda"] = [
            hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()
            for t in torch.cuda.get_rng_state_all()]
    else:
        h["torch_cuda"] = None
    np_state = np.random.get_state()
    h["numpy"] = hashlib.sha256(
        np_state[1].tobytes() + repr(np_state[2:]).encode()).hexdigest()
    h["python_random"] = hashlib.sha256(repr(_random.getstate()).encode()).hexdigest()
    return h


def _graus_por_relacao(graph_part: HeteroData, mod: ModuleType) -> dict:
    """A0-ter item 1: grau de entrada/saida por relacao no grafo INDUZIDO
    da particao (val ou test), recomputado no grafo carregado (nao copiado
    do parecer/`forum-eng-ia_grau_vizinhanca.json`, que media so na janela
    do gate). 'AT' e 'TT' contam grau de ENTRADA do no terrain (destino);
    'TA' conta grau de SAIDA do no terrain (fonte) -- e o grau relevante
    para cada relacao do ponto de vista da SEMENTE terrain."""
    def _stats(edge_index: torch.Tensor, lado: str, n_nos: int) -> dict:
        if edge_index is None or edge_index.numel() == 0:
            z = {"min": 0, "max": 0, "media": 0.0, "soma": 0}
            return z
        idx = edge_index[1] if lado == "dst" else edge_index[0]
        deg = torch.zeros(n_nos, dtype=torch.long)
        deg.scatter_add_(0, idx.long(), torch.ones_like(idx, dtype=torch.long))
        return {"min": int(deg.min()), "max": int(deg.max()),
                "media": float(deg.float().mean()), "soma": int(deg.sum())}

    n_ter = int(graph_part["terrain"].x.shape[0])
    n_ant = int(graph_part["antenna"].x.shape[0])
    ei_at = graph_part[mod.ET_AT].edge_index if mod.ET_AT in graph_part.edge_types else None
    ei_tt = graph_part[mod.ET_TT].edge_index if mod.ET_TT in graph_part.edge_types else None
    ei_ta = graph_part[mod.ET_TA].edge_index if mod.ET_TA in graph_part.edge_types else None
    return {
        "AT": _stats(ei_at, "dst", n_ter),   # antenna->terrain, semente=destino
        "TT": _stats(ei_tt, "dst", n_ter),   # terrain->terrain, semente=destino
        "TA": _stats(ei_ta, "src", n_ter),   # terrain->antenna, semente=fonte
        "n_terrain": n_ter, "n_antenna": n_ant,
    }


def gerar_predicoes_teste_gnn(
    mod: ModuleType, ctx: SimpleNamespace, model, device: str,
    k_antenna: int, k_terrain: int, eval_batch_size: int, seed: int,
    log=_log_stub, particao: str = "test", disjoint: bool = True,
    ordem_nos: Optional[np.ndarray] = None, medir_diagnostico: bool = False,
) -> dict:
    """Constroi o subgrafo INDUZIDO da particao pedida (default 'test'), roda
    `model.eval()` sobre TODOS os seus nos via NeighborLoader (mesmo fan-out
    da producao), e devolve idx_global/target/pred(fisico)/pred_afim
    (pre-clamp)/sentinela alinhados no-a-no.

    `disjoint=True` (item 3 do A0-bis, parecer forum-eng-ia P1b): cada arvore
    de amostragem fica isolada por semente -- sem isso, uma semente do batch
    que tambem seja vizinha de outra sementeja chega com embedding de 2+
    saltos que o treino (1,5% das sementes, fanout 1) nunca viu. Com
    disjoint=True a predicao por no fica invariante a `eval_batch_size` e a
    ordem dos nos (ver `testar_invariancia_disjoint`).

    `ordem_nos`: permutacao OPCIONAL (indices locais do grafo da particao,
    0..n_p-1) usada so pelo teste de invariancia, para forcar uma ORDEM de
    iteracao diferente da natural; o `_scatter_por_semente` do congelado
    sempre devolve o array final alinhado por indice de no, entao a ordem de
    entrada no loader nao deveria mudar o resultado (isso e exatamente o que
    o teste verifica).

    `medir_diagnostico=True` (A0-ter item 1, seguimento do parecer
    forum-eng-ia sobre a "Regra de avaliacao"): grava tambem, no dict de
    saida, `rng_hash_antes`/`rng_hash_depois`/`rng_identico` (o RNG global
    nao deveria ser consumido por uma avaliacao de vizinhanca COMPLETA,
    ao contrario da amostragem por grau > k), `arestas_amostradas_por_relacao`
    (soma, sobre TODOS os batches do loader, de `edge_index.shape[1]` por
    relacao -- que com `num_neighbors=-1` tem de bater com a soma dos graus
    das sementes) e `graus_por_relacao` (min/max/media recomputados no
    grafo INDUZIDO desta particao, via `_graus_por_relacao`)."""
    torch.manual_seed(seed)
    rng_antes = _hash_rng_state() if medir_diagnostico else None
    loc = torch.from_numpy(ctx.parts_local[particao])
    graph_part, _info = mod.induzir_particao(ctx.base, loc, ctx.n_antenna, log, particao)
    nn_kw = {mod.ET_AT: [k_antenna], mod.ET_TT: [k_terrain], mod.ET_TA: [k_antenna]}
    n_p = int(graph_part["terrain"].x.shape[0])
    seeds = ("terrain", None) if ordem_nos is None else (
        "terrain", torch.as_tensor(ordem_nos, dtype=torch.long))
    loader = NeighborLoader(data=graph_part, num_neighbors=nn_kw,
                            input_nodes=seeds, batch_size=eval_batch_size,
                            shuffle=False, num_workers=0, disjoint=disjoint)
    model.eval()
    P, T, I = [], [], []
    n_arestas = {mod.ET_AT: 0, mod.ET_TT: 0, mod.ET_TA: 0}
    with torch.no_grad():
        for b in loader:
            if medir_diagnostico:
                for et in n_arestas:
                    if et in b.edge_types:
                        n_arestas[et] += int(b[et].edge_index.shape[1])
            b = b.to(device)
            bs = b["terrain"].batch_size
            out = model(b)
            P.append(out["predictions"][:bs].float().detach().cpu())
            T.append(b["terrain"].rf_targets[:bs].float().detach().cpu())
            I.append(b["terrain"].n_id[:bs].cpu())
    po, to, visto = mod._scatter_por_semente(torch.cat(I), torch.cat(P), torch.cat(T), n_p)
    assert bool(visto.all()), f"algum no da particao '{particao}' nao recebeu predicao"

    idx_global = ctx.g_idx[ctx.parts_local[particao]]
    if hasattr(model.decoder, "fisico"):
        # AffineDecoderV3 (modelo_v3): `po` e a saida AFIM crua (pre-clamp);
        # `.fisico()` aplica as escalas + clamp fisico (rf_decoder_v3.py).
        pred_fisico = model.decoder.fisico(po.to(device)).detach().cpu().numpy().astype(np.float32)
    else:
        # PhysicsConstrainedDecoder ORIGINAL (E3, item 4 do A0-ter): o
        # forward() JA aplica softplus/sigmoid/clamp -- `po` JA e fisico,
        # nao existe uma representacao "afim" separada para este decoder.
        pred_fisico = po.numpy().astype(np.float32)
    pred_afim = po.numpy().astype(np.float32)
    target = to.numpy().astype(np.float32)
    sentinela = (target[:, 0] >= mod.PL_TARGET_MAX_VALID)
    saida = dict(idx_global=idx_global.astype(np.int64), target=target,
                 pred=pred_fisico, pred_afim=pred_afim, sentinela=sentinela)
    if medir_diagnostico:
        rng_depois = _hash_rng_state()
        graus = _graus_por_relacao(graph_part, mod)
        arestas_por_relacao = {
            "AT": n_arestas[mod.ET_AT], "TT": n_arestas[mod.ET_TT], "TA": n_arestas[mod.ET_TA]}
        soma_graus = {"AT": graus["AT"]["soma"], "TT": graus["TT"]["soma"], "TA": graus["TA"]["soma"]}
        saida["diagnostico"] = {
            "rng_hash_antes": rng_antes,
            "rng_hash_depois": rng_depois,
            "rng_identico": bool(rng_antes == rng_depois),
            "arestas_amostradas_por_relacao": arestas_por_relacao,
            "soma_graus_das_sementes_por_relacao": soma_graus,
            "arestas_bate_com_soma_graus": {
                k: bool(arestas_por_relacao[k] == soma_graus[k]) for k in ("AT", "TT", "TA")},
            "graus_por_relacao": graus,
            "k_antenna_usado": int(k_antenna), "k_terrain_usado": int(k_terrain),
            "disjoint_usado": bool(disjoint),
        }
    return saida


def gerar_predicoes_teste_mlp(
    mod: ModuleType, ctx: SimpleNamespace, model, device: str,
    eval_batch_size: int, seed: int, log=_log_stub, particao: str = "test",
) -> dict:
    """Equivalente sem grafo: fatia `x_full`/`rf_targets` pelos indices
    locais da particao pedida (default 'test') e roda o MLP em lotes, sem
    NeighborLoader (mesmo padrao [M4] do congelado `train_mlp_c0_spatial.py`).
    O MLP nao tem arestas, entao nao ha campo receptivo dependente de batch/
    ordem: item 3 do A0-bis (disjoint) e uma questao exclusiva do GNN."""
    torch.manual_seed(seed)
    loc = torch.from_numpy(ctx.parts_local[particao])
    x_test = ctx.x_full[loc]
    y_test = ctx.base["terrain"].rf_targets[loc]
    n_p = int(x_test.shape[0])
    model.eval()
    P, T = [], []
    with torch.no_grad():
        for i in range(0, n_p, eval_batch_size):
            xb = x_test[i:i + eval_batch_size].to(device)
            out = model(xb)
            P.append(out["predictions"].float().detach().cpu())
            T.append(y_test[i:i + eval_batch_size].float())
    po = torch.cat(P)
    to = torch.cat(T)

    idx_global = ctx.g_idx[ctx.parts_local[particao]]
    if hasattr(model.decoder, "fisico"):
        pred_fisico = model.decoder.fisico(po.to(device)).detach().cpu().numpy().astype(np.float32)
    else:
        # PhysicsConstrainedDecoder ORIGINAL (controle B do diagnostico A2/MLP,
        # 26/09/2026): forward() ja aplica softplus/sigmoid/clamp -- `po` ja e
        # fisico, mesmo fallback usado em gerar_predicoes_teste_gnn.
        pred_fisico = po.numpy().astype(np.float32)
    pred_afim = po.numpy().astype(np.float32)
    target = to.numpy().astype(np.float32)
    sentinela = (target[:, 0] >= mod.PL_TARGET_MAX_VALID)
    return dict(idx_global=idx_global.astype(np.int64), target=target,
                pred=pred_fisico, pred_afim=pred_afim, sentinela=sentinela)


def testar_invariancia_avaliacao_completa(
    mod: ModuleType, ctx: SimpleNamespace, model, device: str, seed: int,
    eval_batch_size_a: int, eval_batch_size_b: int,
    particao: str = "test", log=_log_stub,
) -> dict:
    """A0-ter item 1 (Regra de avaliacao do parecer, 26/09/2026): substitui
    `testar_invariancia_disjoint_gnn` do A0-bis. Vizinhanca SEMPRE COMPLETA
    (`k_antenna=k_terrain=-1`) e `disjoint=True` -- os UNICOS parametros da
    avaliacao, hardcoded aqui (a regra nao e mais "escolha do chamador").
    Sem sorteio de vizinhos (grau real <= 9 em TT, <=5 em AT, ver
    `forum-eng-ia_grau_vizinhanca.json`), a predicao por no deveria ficar
    invariante tanto a `eval_batch_size` quanto a ORDEM das sementes -- ao
    contrario do A0-bis (k=8/20), onde a ordem mudava ATE 5,27 dB por no
    porque a subamostragem consumia o RNG global sequencialmente.

    Desenho (item (4a)/(4b) do parecer):
      - `piso`: a MESMA avaliacao (ordem natural, bs=eval_batch_size_a)
        rodada 2x -- mede o ruido de determinismo numerico da GPU (reducoes
        CUDA nao sao bit-exatas entre chamadas), que NAO e o que o teste
        mede, mas define o piso abaixo do qual nenhuma diferenca e real.
      - grade 2 ORDENS permutadas x 2 `eval_batch_size` (4 avaliacoes),
        cada uma comparada contra a avaliacao-base (ordem natural,
        `eval_batch_size_a`).
    Criterio (bloco "(4a)/(4b)" do parecer, corrigido pelo proprio autor:
    o 1e-5 original era inatingivel, pois o piso de float32 em dB e
    2^-15=3,0518e-5): diferenca maxima por no <= 1e-4 dB E <= 3x o piso
    medido nesta mesma corrida."""
    K = -1
    saida_base = gerar_predicoes_teste_gnn(
        mod=mod, ctx=ctx, model=model, device=device, k_antenna=K, k_terrain=K,
        eval_batch_size=eval_batch_size_a, seed=seed, log=log,
        particao=particao, disjoint=True, ordem_nos=None)
    n_p = int(saida_base["idx_global"].shape[0])

    saida_piso = gerar_predicoes_teste_gnn(
        mod=mod, ctx=ctx, model=model, device=device, k_antenna=K, k_terrain=K,
        eval_batch_size=eval_batch_size_a, seed=seed, log=log,
        particao=particao, disjoint=True, ordem_nos=None)

    rng1 = np.random.default_rng(seed)
    ordem1 = rng1.permutation(n_p)
    rng2 = np.random.default_rng(seed + 1)
    ordem2 = rng2.permutation(n_p)

    combinacoes = {
        "ordem1_bsA": dict(eval_batch_size=eval_batch_size_a, ordem_nos=ordem1),
        "ordem1_bsB": dict(eval_batch_size=eval_batch_size_b, ordem_nos=ordem1),
        "ordem2_bsA": dict(eval_batch_size=eval_batch_size_a, ordem_nos=ordem2),
        "ordem2_bsB": dict(eval_batch_size=eval_batch_size_b, ordem_nos=ordem2),
    }

    def _cmp(s1, s2):
        idx_bate = bool(np.array_equal(s1["idx_global"], s2["idx_global"]))
        if not idx_bate:
            return {"idx_global_identico": False, "diff_max_afim": None, "diff_max_fisico": None}
        return {
            "idx_global_identico": True,
            "diff_max_afim": float(np.abs(s1["pred_afim"] - s2["pred_afim"]).max()),
            "diff_max_fisico": float(np.abs(s1["pred"] - s2["pred"]).max()),
        }

    piso = _cmp(saida_base, saida_piso)
    piso_medido = piso["diff_max_fisico"] if piso["diff_max_fisico"] is not None else 0.0

    comparacoes = {}
    for nome, kw in combinacoes.items():
        saida_i = gerar_predicoes_teste_gnn(
            mod=mod, ctx=ctx, model=model, device=device, k_antenna=K, k_terrain=K,
            eval_batch_size=kw["eval_batch_size"], seed=seed, log=log,
            particao=particao, disjoint=True, ordem_nos=kw["ordem_nos"])
        comparacoes[nome] = _cmp(saida_base, saida_i)

    diffs_fisico = [c["diff_max_fisico"] for c in comparacoes.values()
                    if c["diff_max_fisico"] is not None]
    diff_max_geral = max(diffs_fisico) if diffs_fisico else None
    limiar_dB = 1e-4
    limiar_3x_piso = 3.0 * piso_medido
    passou = bool(
        all(c["idx_global_identico"] for c in comparacoes.values())
        and diff_max_geral is not None
        and diff_max_geral <= limiar_dB
        and diff_max_geral <= limiar_3x_piso)
    return {
        "particao": particao,
        "n_nos": n_p,
        "regra": "vizinhanca COMPLETA (k_antenna=k_terrain=-1), disjoint=True, sempre",
        "eval_batch_size_a": int(eval_batch_size_a),
        "eval_batch_size_b": int(eval_batch_size_b),
        "piso_numerico_medido_mesma_ordem_mesma_bs": piso,
        "comparacoes_ordem_x_batch_size": comparacoes,
        "diff_max_fisico_entre_todas_combinacoes": diff_max_geral,
        "limiar_absoluto_dB": limiar_dB,
        "limiar_3x_piso_medido": limiar_3x_piso,
        "passou_criterio": passou,
        "nota": ("com vizinhanca COMPLETA nao ha sorteio de vizinhos (grau real <= 9 em TT, "
                 "<=5 em AT -- forum-eng-ia_grau_vizinhanca.json): a unica fonte de diferenca "
                 "esperada e o ruido de reducao numerica da GPU (nao bit-exata entre chamadas), "
                 "medido aqui pelo 'piso' e usado como referencia de escala do criterio."),
    }


# --------------------------------------------------------------------------
# Item 1/1b da entrega A0-bis: paridade de LARGURAS do MLP pelos parametros
# EFETIVOS da GNN (forum-eng-ia_larguras_mlp_efetivo.json, parecer
# `forum-eng-ia_fanout_capacidade.md`, recomendacao 1(b)/1(c)/P2.6).
# --------------------------------------------------------------------------
MLP_LARGURAS_NOVAS = (654, 654, 600, 636, 256)
N_PARAMS_ALVO_NOMINAL_NOVO = 1_484_067          # == enc(654,654,600,636)+DEC
N_PARAMS_ALVO_EFETIVO_GNN = 1_484_037           # gnn_efetivos_fanout1 (contagem_efetiva.json)
MLP_LARGURAS_ANTIGAS = (712, 712, 720, 688, 256)  # larguras nominais originais (braco de sensibilidade E2)
N_PARAMS_ALVO_NOMINAL_ANTIGO = 1_812_515
FONTE_LARGURAS = (
    "gpu/A1_inv/forum-eng-ia_larguras_mlp_efetivo.json e "
    "gpu/A1_inv/forum-eng-ia_contagem_efetiva.json (forum-eng-ia, 26/09/2026)")


def patch_larguras_mlp(mod: ModuleType, larguras: tuple, alvo_nominal: int) -> None:
    """Monkeypatch de ATRIBUTO DE MODULO (nao edita `train_mlp_c0_spatial.py`):
    `MLPRFModel.__init__` e `main()` resolvem `MLP_LARGURAS`/`N_PARAMS_ALVO_GNN`
    como nomes GLOBAIS do modulo a cada chamada (`larguras=MLP_LARGURAS` e o
    `if n_params_tr != N_PARAMS_ALVO_GNN` dentro de `main()`, linhas ~1360 e
    ~1366 do congelado) -- sobrescrever os dois atributos do MODULO `mod`
    ANTES de `mod.main(argv)` e suficiente para trocar a largura E o alvo da
    guarda de paridade, sem tocar uma linha do arquivo."""
    mod.MLP_LARGURAS = tuple(larguras)
    mod.N_PARAMS_ALVO_GNN = int(alvo_nominal)


# --------------------------------------------------------------------------
# Item 3: NeighborLoader disjunto tambem no LOADER DE TREINO/VAL/TEST que o
# script congelado constroi DENTRO de `main()` (nao ha hook exposto).
# Monkeypatch da CLASSE `NeighborLoader` no namespace do modulo carregado.
# --------------------------------------------------------------------------
class _NeighborLoaderDisjointAuto(NeighborLoader):
    """Subclasse usada SO por monkeypatch do atributo `mod.NeighborLoader`
    (o congelado importa `from torch_geometric.loader import NeighborLoader`
    no topo do arquivo e resolve esse nome GLOBAL a cada chamada de
    `make_loader()` dentro de `main()` -- sobrescrever `mod.NeighborLoader`
    ANTES de `mod.main()` troca a classe usada la dentro, sem editar nada).

    `make_loader()` do congelado chama sempre com `shuffle=` explicito:
    True SO para o loader de TREINO, False para VAL e TESTE (linhas
    ~1173-1183 de `train_gnn_c0_spatial.py`). Usamos esse `shuffle` como o
    unico sinal disponivel (sem acesso a variavel local) para decidir onde
    aplicar `disjoint=True`: SEMPRE que `shuffle=False` (val/test, sempre,
    conforme parecer P1b/recomendacao 1(c)) e, alem disso, quando
    `shuffle=True` (treino) SE a flag de classe `ativar_treino` estiver
    ligada (`--disjoint-treino` do wrapper)."""
    ativar_treino = False

    def __init__(self, *args, **kwargs):
        shuffle = kwargs.get("shuffle", None)
        aplica_val_test = (shuffle is False)
        aplica_treino = (shuffle is True and _NeighborLoaderDisjointAuto.ativar_treino)
        if (aplica_val_test or aplica_treino) and "disjoint" not in kwargs:
            kwargs["disjoint"] = True
        super().__init__(*args, **kwargs)


def patch_neighborloader_disjoint(mod: ModuleType, disjoint_treino: bool = False) -> None:
    """Aplica o monkeypatch acima no modulo congelado JA carregado. Chamar
    ANTES de `mod.main(argv)`."""
    _NeighborLoaderDisjointAuto.ativar_treino = bool(disjoint_treino)
    mod.NeighborLoader = _NeighborLoaderDisjointAuto


# --------------------------------------------------------------------------
# Item 2: contagem de parametros EFETIVOS (com gradiente vivo) medida NUM
# BATCH DE PRODUCAO real (forward+backward), nao copiada do parecer.
# --------------------------------------------------------------------------
def contar_parametros_efetivos_via_backward(model, preds: torch.Tensor) -> dict:
    """Roda `.sum().backward()` sobre predicoes JA calculadas por um forward
    REAL (batch de producao) e classifica cada parametro TREINAVEL do modelo:
    'None' (nunca entra no grafo computacional -- parede arquitetural, ex.
    camada 3/antenna e as 3 cabecas mortas do decodificador) ou 'zero'
    (entra no grafo mas o gradiente da exatamente 0 -- ex. `lin_l.weight` do
    SAGEConv terrain->antenna quando o fanout de 1 salto nunca amostra
    aresta cujo destino e antena, achado `forum-eng-ia_contagem_efetiva`/
    `diagnostico_A1.json`). Usar `.sum()` (nao a CurriculumRFLoss) e
    suficiente e mais robusto para ESTA auditoria de capacidade: qualquer
    parametro cuja contribuicao para QUALQUER canal de saida seja
    estruturalmente nula (agregacao vazia) continua dando gradiente zero
    tambem sob `.sum()`, porque a nulidade vem da FORWARD (multiplicar por
    um tensor de agregacao vazio), nao da loss especifica."""
    for p in model.parameters():
        p.grad = None
    preds.sum().backward()
    mortos: dict = {}
    n_total = 0
    for name, p in model.named_parameters():
        n = int(p.numel())
        n_total += n
        if not p.requires_grad:
            continue
        if p.grad is None:
            mortos[name] = {"numel": n, "shape": list(p.shape), "grad": "None"}
        elif int(torch.count_nonzero(p.grad)) == 0:
            mortos[name] = {"numel": n, "shape": list(p.shape), "grad": "zero"}
    n_mortos = sum(v["numel"] for v in mortos.values())
    return {
        "n_params_total": n_total,
        "n_params_sem_gradiente": n_mortos,
        "n_params_efetivos": n_total - n_mortos,
        "parametros_sem_gradiente": mortos,
    }


def medir_capacidade_efetiva_gnn(
    mod: ModuleType, ctx: SimpleNamespace, model, device: str,
    k_antenna: int, k_terrain: int, batch_size: int, seed: int, log=_log_stub,
) -> dict:
    """Constroi 1 batch REAL do loader de TREINO (mesmo fanout/particao da
    producao) e mede a capacidade efetiva via backward real. Roda sobre uma
    COPIA do modelo (`copy.deepcopy`) para nao perturbar `running_mean/var`
    de BatchNorm/LayerNorm do modelo que ainda vai gerar as predicoes finais
    de teste."""
    import copy
    torch.manual_seed(seed)
    loc = torch.from_numpy(ctx.parts_local["train"])
    graph_train, _info = mod.induzir_particao(ctx.base, loc, ctx.n_antenna, log, "train")
    nn_kw = {mod.ET_AT: [k_antenna], mod.ET_TT: [k_terrain], mod.ET_TA: [k_antenna]}
    loader = NeighborLoader(data=graph_train, num_neighbors=nn_kw,
                            input_nodes=("terrain", None), batch_size=batch_size,
                            shuffle=True, num_workers=0)
    batch = next(iter(loader)).to(device)
    model_probe = copy.deepcopy(model).to(device)
    model_probe.train()
    out = model_probe(batch)
    bs = batch["terrain"].batch_size
    resultado = contar_parametros_efetivos_via_backward(model_probe, out["predictions"][:bs])
    del model_probe, batch, loader, graph_train
    return resultado


def medir_capacidade_efetiva_mlp(
    ctx: SimpleNamespace, model, device: str, batch_size: int, seed: int,
) -> dict:
    """Equivalente ao acima, sem grafo: 1 batch REAL fatiado de `x_full` na
    particao de TREINO."""
    import copy
    torch.manual_seed(seed)
    loc = torch.from_numpy(ctx.parts_local["train"])
    xb = ctx.x_full[loc][:batch_size].to(device)
    model_probe = copy.deepcopy(model).to(device)
    model_probe.train()
    out = model_probe(xb)
    resultado = contar_parametros_efetivos_via_backward(model_probe, out["predictions"])
    del model_probe, xb
    return resultado


# --------------------------------------------------------------------------
# Item 4: instrumentacao por corrida (GradScaler, norma de gradiente, ramo
# except do encoder) via monkeypatch de METODO DE CLASSE, instalado so
# durante a chamada a `mod.main()`.
# --------------------------------------------------------------------------
class InstrumentacaoTreino:
    """Context manager: instala 4 monkeypatches de CLASSE (nunca edita
    arquivo) e devolve os contadores agregados via `.resultado()`.

    - `torch.optim.AdamW.step`: conta toda vez que a atualizacao de peso e
      DE FATO executada (`GradScaler.step()` so chama isto quando o
      gradiente e finito -- mesma tecnica do achado_b de
      `gpu/A1_inv/epoca_real_gradiente_canais.py`/`diagnostico_A1.json`).
    - `torch.amp.GradScaler.update`: conta toda chamada (1 por passo,
      sempre) e grava a escala antes/depois.
    - `torch.nn.utils.clip_grad_norm_`: grava a norma TOTAL do gradiente
      que a propria funcao ja calcula e devolve ANTES do clip real.
    - `torch_geometric.nn.HeteroConv.forward`: conta quantas vezes o
      encoder caiu no ramo `except` (fallback sem `edge_attr_dict`) de
      `gnn_rf_encoder.py:202-208` -- reconhecido pela chamada de 2
      argumentos posicionais (sem o 3o, `edge_attr_dict`), o UNICO padrao
      de chamada que o `except` usa (`conv(x_dict, edge_index_dict)`,
      contra o normal `conv(x_dict, edge_index_dict, edge_attr_dict)`)."""

    def __init__(self):
        self.n_atualizacoes_efetivas = 0
        self.escalas = []          # (antes, depois) por chamada de update()
        self.normas_pre_clip = []
        self.n_except_edge_attr = 0
        self._orig = {}

    def __enter__(self):
        import torch.optim as optim
        import torch.amp as amp
        import torch.nn.utils as nnutils
        import torch_geometric.nn as pyg_nn

        self._orig["adamw_step"] = optim.AdamW.step
        self._orig["scaler_update"] = amp.GradScaler.update
        self._orig["clip"] = nnutils.clip_grad_norm_
        self._orig["heteroconv_forward"] = pyg_nn.HeteroConv.forward
        inst = self

        def step_patched(opt_self, *a, **kw):
            inst.n_atualizacoes_efetivas += 1
            return inst._orig["adamw_step"](opt_self, *a, **kw)

        def update_patched(scaler_self, *a, **kw):
            antes = float(scaler_self.get_scale())
            r = inst._orig["scaler_update"](scaler_self, *a, **kw)
            depois = float(scaler_self.get_scale())
            inst.escalas.append((antes, depois))
            return r

        def clip_patched(*a, **kw):
            norma = inst._orig["clip"](*a, **kw)
            inst.normas_pre_clip.append(float(norma))
            return norma

        def heteroconv_forward_patched(hc_self, x_dict, edge_index_dict,
                                        edge_attr_dict=None, *a, **kw):
            if edge_attr_dict is None:
                inst.n_except_edge_attr += 1
            return inst._orig["heteroconv_forward"](
                hc_self, x_dict, edge_index_dict, edge_attr_dict, *a, **kw)

        optim.AdamW.step = step_patched
        amp.GradScaler.update = update_patched
        nnutils.clip_grad_norm_ = clip_patched
        pyg_nn.HeteroConv.forward = heteroconv_forward_patched
        return self

    def __exit__(self, *exc):
        import torch.optim as optim
        import torch.amp as amp
        import torch.nn.utils as nnutils
        import torch_geometric.nn as pyg_nn
        optim.AdamW.step = self._orig["adamw_step"]
        amp.GradScaler.update = self._orig["scaler_update"]
        nnutils.clip_grad_norm_ = self._orig["clip"]
        pyg_nn.HeteroConv.forward = self._orig["heteroconv_forward"]
        return False

    def resultado(self, max_norm: float = 0.5) -> dict:
        n_passos = len(self.escalas)
        n_atualizados = self.n_atualizacoes_efetivas
        pulou = [depois < antes for (antes, depois) in self.escalas]
        idx_primeiro_sem_pulo = next((i for i, p in enumerate(pulou) if not p), None)
        pulos_pos_aquecimento = (
            int(sum(pulou[idx_primeiro_sem_pulo:])) if idx_primeiro_sem_pulo is not None
            else int(sum(pulou)))
        idx_ultimo_pulo = max((i for i, p in enumerate(pulou) if p), default=None)
        escalas_iniciais = [a for (a, _) in self.escalas]
        escalas_finais = [d for (_, d) in self.escalas]
        normas = np.asarray(self.normas_pre_clip, dtype=np.float64)
        finitas = normas[np.isfinite(normas)] if normas.size else normas
        n_nao_finitas = int(normas.size - finitas.size)
        return {
            "n_passos": n_passos,
            "n_atualizacoes_efetivas": n_atualizados,
            "n_pulados_total": n_passos - n_atualizados,
            "n_pulados_pos_aquecimento": pulos_pos_aquecimento,
            "definicao_pos_aquecimento": (
                "pulos apos o indice do 1o passo SEM pulo (1a atualizacao bem-sucedida)"),
            "indice_primeiro_passo_sem_pulo": idx_primeiro_sem_pulo,
            "indice_ultimo_pulo": idx_ultimo_pulo,
            "escala_inicial": escalas_iniciais[0] if escalas_iniciais else None,
            "escala_final": escalas_finais[-1] if escalas_finais else None,
            "escala_minima": min(escalas_finais) if escalas_finais else None,
            "norma_gradiente_pre_clip_p50": (
                float(np.percentile(finitas, 50)) if finitas.size else None),
            "norma_gradiente_pre_clip_p99": (
                float(np.percentile(finitas, 99)) if finitas.size else None),
            "n_normas_nao_finitas": n_nao_finitas,
            "nota_normas_nao_finitas": (
                "norma pre-clip nao-finita (inf/nan) ocorre nos passos que o GradScaler "
                "detecta e pula (P3 do parecer); excluidas do P50/P99, contadas aqui"),
            "fracao_passos_clipados": (
                float(np.mean(finitas > max_norm)) if finitas.size else None),
            "n_except_edge_attr_encoder": self.n_except_edge_attr,
        }


# --------------------------------------------------------------------------
# Item 4: diagnostico de SAIDA por canal 0-3 (regras R-c0..R-c3 do parecer,
# bloco P4) -- SO GRAVA OS NUMEROS, quem julga e o gate/criterio.
# --------------------------------------------------------------------------
_CANAL_CLAMP = {0: (0.0, 200.0), 1: (0.0, 50.0), 2: (0.0, 30.0), 3: (-150.0, 0.0)}


def diagnostico_saida_canais(pred_afim: np.ndarray, pred_fisico: np.ndarray,
                              target: np.ndarray,
                              sentinela: Optional[np.ndarray] = None) -> dict:
    """A2 (gpu/A2_inv, achado do chefe/dono 26/09/2026): o canal 0
    (path_loss_total) usa `target[:,0] >= PL_TARGET_MAX_VALID` (299 dB) como
    SENTINELA de "sem alvo de PL valido" (mesma convencao do congelado,
    `mod.PL_TARGET_MAX_VALID`/`metricas_particao.pl_valid` -- NAO e um valor
    fisico real, e um marcador fora do clamp de producao [0,200]). R-c0 do
    parecer (P4) exige "fracao de alvos fora da faixa do clamp = 0" antes de
    ler mae_pos_clamp; a populacao sentinela VIOLA R-c0 por construcao (alvo
    ~300, sempre fora de [0,200]), e a rede aprende (huber loss sem mascara
    na perda de producao) a prever perto de 300 tambem ali -- por isso
    `mae_bruto` fica pequeno (raw perto do alvo, sem clamp) mas
    `mae_pos_clamp` explode (clamp trava em 200, |200-300|=100 por no
    sentinela). SEM mascara, o canal 0 misturava as duas populacoes (ate
    ~87% de sentinela em Bauru Q1) e o clamp "piorava o erro" exatamente
    como o proprio parecer avisou (R-c0). Com `sentinela` informado, o
    canal 0 reporta as metricas PRIMARIAS (mae_bruto/mae_pos_clamp/
    fracao_fora_da_faixa/p99_excesso/peso_do_clamp_pct) SO sobre a
    populacao "valido" (~sentinela) -- mesma populacao que
    `mod.metricas_particao`/`mae_pl_db` usa -- e GRAVA TAMBEM, com sufixo
    `_todos_com_sentinela`, os numeros antigos (sem mascara), para
    auditoria/comparacao, nunca escondidos."""
    out = {}
    for canal, (lo, hi) in _CANAL_CLAMP.items():
        raw = pred_afim[:, canal].astype(np.float64)
        clampado = pred_fisico[:, canal].astype(np.float64)
        tgt = target[:, canal].astype(np.float64)

        def _bloco(idx=None):
            r = raw if idx is None else raw[idx]
            c = clampado if idx is None else clampado[idx]
            t = tgt if idx is None else tgt[idx]
            if r.size == 0:
                return {"n": 0, "fracao_fora_da_faixa": None, "p99_excesso_dB_ou_dBm": None,
                        "mae_bruto": None, "mae_pos_clamp": None, "peso_do_clamp_pct": None}
            fora = (r < lo) | (r > hi)
            excesso = np.maximum(lo - r, 0.0) + np.maximum(r - hi, 0.0)
            mb = float(np.mean(np.abs(r - t)))
            mc = float(np.mean(np.abs(c - t)))
            return {
                "n": int(r.size),
                "fracao_fora_da_faixa": float(np.mean(fora)),
                "p99_excesso_dB_ou_dBm": float(np.percentile(excesso, 99)),
                "mae_bruto": mb,
                "mae_pos_clamp": mc,
                "peso_do_clamp_pct": float((mb - mc) / mc * 100.0) if mc > 0 else None,
            }

        if canal == 0 and sentinela is not None:
            valido_idx = ~sentinela
            canal_out = _bloco(valido_idx)  # PRIMARIO: so populacao com PL valido (R-c0)
            canal_out["frac_alvo_sentinela"] = float(np.mean(sentinela))
            canal_out["definicao_sentinela"] = "target[:,0] >= PL_TARGET_MAX_VALID (299 dB)"
            todos = _bloco(None)
            for k, v in todos.items():
                canal_out[f"{k}_todos_com_sentinela"] = v
        else:
            canal_out = _bloco(None)

        if canal == 1:
            mask0 = (tgt == 0.0)
            mask_pos = ~mask0
            canal_out["frac_alvo_exatamente_zero"] = float(np.mean(mask0))
            if mask0.any():
                canal_out["mae_bruto_alvo_zero"] = float(np.mean(np.abs(raw[mask0] - tgt[mask0])))
                canal_out["mae_pos_clamp_alvo_zero"] = float(np.mean(np.abs(clampado[mask0] - tgt[mask0])))
            if mask_pos.any():
                canal_out["mae_bruto_alvo_maior_zero"] = float(np.mean(np.abs(raw[mask_pos] - tgt[mask_pos])))
                canal_out["mae_pos_clamp_alvo_maior_zero"] = float(np.mean(np.abs(clampado[mask_pos] - tgt[mask_pos])))
        out[f"canal_{canal}"] = canal_out
    return out


# --------------------------------------------------------------------------
# A0-ter item 4 (E3, parecer forum-eng-ia_fanout_capacidade.md, secao
# "Regra de avaliacao" (3)): MAE de RSSI (canal 3) por POPULACAO -- "valido"
# (`~sentinela`, o mesmo `sentinela` que `gerar_predicoes_teste_gnn` ja
# devolve: `target[:,0] >= PL_TARGET_MAX_VALID`), "sentinela" e "todos".
# --------------------------------------------------------------------------
def mae_rssi_por_populacao(pred_fisico: np.ndarray, target: np.ndarray,
                            sentinela: np.ndarray) -> dict:
    err = np.abs(pred_fisico[:, 3].astype(np.float64) - target[:, 3].astype(np.float64))
    valido = ~sentinela
    out = {"todos": {"n": int(err.size), "mae_rssi_db": float(err.mean())}}
    for nome, mask in (("valido", valido), ("sentinela", sentinela)):
        if mask.any():
            out[nome] = {"n": int(mask.sum()), "mae_rssi_db": float(err[mask].mean())}
        else:
            out[nome] = {"n": 0, "mae_rssi_db": None}
    return out
