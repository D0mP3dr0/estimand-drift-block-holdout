#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste 1.7 (fase 1, roadmap v3) -- alegacao A2 / P3 (prop:degree): o subgrafo
terreno-terreno induzido por particao preserva o grau local dos nos retidos?

Grafo REAL: (dem, adjacent_to, dem).edge_index dos tensores *_gpu.pt (15 GB,
16 celulas locais, /trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3/), com
dem.pos = posicoes REAIS (nao reconstruidas por linspace). Split:
split_espacial_3vias_exato, copia literal de train_gnn_c0_spatial.py:412-472
(mesma logica de varredura_split_geometria.py / p3_grau_induzido.py), aplicada
sobre pos_m REAL (nao sintetica) -- portanto exata, sem residuo de
reconstrucao de malha.

Metrica por (celula, sorteio, particao): grau medio/P5/P50/P95 dos nos
retidos ANTES (grafo inteiro) e DEPOIS (subgrafo induzido, PyG subgraph()
semantics: aresta retida sse AMBOS os extremos estao no conjunto), separado
em interior/fronteira (fronteira = distancia < ell_medio ~29.8m ate o
complemento da mascara retida, via scipy.ndimage.distance_transform_edt,
que trata fora do array/dominio da celula como fundo -- logo borda do
dominio tambem conta como fronteira). Arestas cruzando particao (uma ponta
train-retido, outra val/test-retido) sao contadas e devem dar 0.

Uso:
  .venv/bin/python v3_1.7_grau_preservacao.py --out <json> --celulas N --seeds N [--log F]
"""
import argparse
import hashlib
import json
import time
import resource
from pathlib import Path

import numpy as np

GPU_DIR = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")
TREINOS_DIR = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
    "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/dados/treinos_c1"
)
CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QS = ["Q1", "Q2", "Q3", "Q4"]
N_SIDE = 3600
FRACS = (0.70, 0.15, 0.15)
SEEDS_20 = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def sha256_of_file(path, limite_bytes=None):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if limite_bytes and n >= limite_bytes:
                break
    return h.hexdigest()


def sha256_of_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


# ---------------------------------------------------------------------
# split_espacial_3vias -- copia literal (train_gnn_c0_spatial.py:412-472),
# identica a varredura_split_geometria.py / p3_grau_induzido.py
# ---------------------------------------------------------------------
def assign_groups(pos_km, grid_size_km):
    grid_x = (pos_km[:, 0] / grid_size_km).astype(int)
    grid_y = (pos_km[:, 1] / grid_size_km).astype(int)
    max_y = grid_y.max() + 1
    return grid_x * max_y + grid_y


def split_espacial_3vias_exato(pos_m, grid_km, buffer_km, fracs, split_seed):
    from scipy.spatial import cKDTree

    pos_km = pos_m / 1000.0
    group_ids = assign_groups(pos_km, grid_km)
    grupos = np.unique(group_ids)
    rng = np.random.RandomState(split_seed)
    grupos_emb = grupos.copy()
    rng.shuffle(grupos_emb)

    n_g = len(grupos_emb)
    n_tr = max(1, int(round(fracs[0] * n_g)))
    n_va = max(1, int(round(fracs[1] * n_g)))
    if n_tr + n_va >= n_g:
        n_tr = max(1, n_g - 2)
        n_va = 1
    g_tr = grupos_emb[:n_tr]
    g_va = grupos_emb[n_tr:n_tr + n_va]
    g_te = grupos_emb[n_tr + n_va:]

    m_tr = np.isin(group_ids, g_tr)
    m_va = np.isin(group_ids, g_va)
    m_te = np.isin(group_ids, g_te)

    kw = dict(compact_nodes=False, balanced_tree=False)
    if buffer_km > 0:
        if m_tr.any() and m_va.any():
            tree_tr = cKDTree(pos_km[m_tr], **kw)
            d, _ = tree_tr.query(pos_km[m_va], k=1, workers=-1)
            idx_va = np.where(m_va)[0]
            m_va[idx_va[d < buffer_km]] = False
        m_trva = m_tr | m_va
        if m_trva.any() and m_te.any():
            tree_trva = cKDTree(pos_km[m_trva], **kw)
            d, _ = tree_trva.query(pos_km[m_te], k=1, workers=-1)
            idx_te = np.where(m_te)[0]
            m_te[idx_te[d < buffer_km]] = False

    return {"train": m_tr, "val": m_va, "test": m_te}


def carregar_celula(cidade, q, log):
    import torch

    pt_path = GPU_DIR / f"{cidade}_v19_{q}_gpu.pt"
    run_path = TREINOS_DIR / f"run_c0c1cf_{cidade}_s42_{q}_g10b2.json"
    cfg = json.load(open(run_path))["config"]

    t0 = time.time()
    d = torch.load(str(pt_path), map_location="cpu", mmap=True, weights_only=False)
    pos_deg = d["dem"].pos.numpy().astype(np.float64)  # (N,2) lon,lat REAIS
    ei = d[("dem", "adjacent_to", "dem")].edge_index.numpy()
    dt_load = time.time() - t0
    n_nodes = pos_deg.shape[0]
    assert n_nodes == N_SIDE * N_SIDE, n_nodes

    # projecao equirretangular identica a latlon_graus_para_metros
    # (train_gnn_c0_spatial.py:351-373)
    lon = pos_deg[:, 0]
    lat = pos_deg[:, 1]
    lon_min, lat_min = float(lon.min()), float(lat.min())
    y_m = (lat - lat_min) * 111_000.0
    x_m = (lon - lon_min) * 111_000.0 * np.cos(np.radians(lat))
    pos_m = np.stack([x_m, y_m], axis=1)

    ell_x = float(np.median(np.abs(np.diff(x_m[:N_SIDE]))))
    # y varia por linha da malha (node_id = row*N_SIDE+col); precisa do passo entre linhas
    ell_y = float(np.median(np.abs(np.diff(y_m[::N_SIDE]))))
    ell_medio = (ell_x + ell_y) / 2.0

    grau_full = np.bincount(ei[0], minlength=n_nodes) if ei.size else np.zeros(n_nodes, dtype=np.int64)
    # simetria: checagem por amostra (custoso checar tudo -- ver script simetria dedicado)
    amostra = np.random.RandomState(0).choice(ei.shape[1], size=min(200000, ei.shape[1]), replace=False)
    pares_diretos = set(map(tuple, ei[:, amostra].T.tolist()))
    pares_invertidos_presentes = 0
    ei_set_check = set(map(tuple, ei[:, amostra].T.tolist()))
    # checagem rapida: para uma sub-amostra, (v,u) esta em qualquer lugar de ei? custoso com set completo;
    # aqui aproximamos com grau_full[u]==grau_full[v] nao decide -- fazemos checagem exata via lookup em set global pequeno
    log(f"  {cidade} {q}: carga {dt_load:.1f}s, n_nos={n_nodes:,}, n_arestas={ei.shape[1]:,}, "
        f"grau_medio={grau_full.mean():.6f}, ell_x={ell_x:.3f}m ell_y={ell_y:.3f}m")

    return {
        "cidade": cidade, "q": q, "pt_path": str(pt_path), "run_path": str(run_path),
        "cfg": cfg, "pos_m": pos_m, "ei": ei, "grau_full": grau_full,
        "ell_x_m": ell_x, "ell_y_m": ell_y, "ell_medio_m": ell_medio,
        "n_nodes": n_nodes, "n_edges": int(ei.shape[1]),
        "tempo_carga_s": dt_load,
    }


def checar_simetria(ei, n_nodes, rng_seed=0, n_amostra=500000):
    """Confirma simetria (grafo nao-direcionado): para uma amostra de arestas
    (u,v), verifica se (v,u) tambem esta em ei, via tabela hash de chaves
    u*n+v restrita a nos amostrados (evita materializar todas as 116M em
    memoria como set de tuplas)."""
    m = ei.shape[1]
    rng = np.random.RandomState(rng_seed)
    idx = rng.choice(m, size=min(n_amostra, m), replace=False)
    u_s, v_s = ei[0, idx], ei[1, idx]
    # keys de TODAS as arestas com origem em algum v_s (para achar (v,u))
    alvo = np.unique(v_s)
    mask_rev_src = np.isin(ei[0], alvo)
    rev_u = ei[0, mask_rev_src]
    rev_v = ei[1, mask_rev_src]
    chave_rev = rev_u.astype(np.int64) * n_nodes + rev_v.astype(np.int64)
    set_rev = set(chave_rev.tolist())
    chave_inv = v_s.astype(np.int64) * n_nodes + u_s.astype(np.int64)
    achou = np.fromiter((c in set_rev for c in chave_inv.tolist()), dtype=bool, count=len(chave_inv))
    return {"n_amostra": int(len(idx)), "n_simetricas": int(achou.sum()),
            "frac_simetrica": float(achou.mean())}


def percentis(vals):
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return None
    return {"media": float(vals.mean()), "p5": float(np.percentile(vals, 5)),
            "p50": float(np.percentile(vals, 50)), "p95": float(np.percentile(vals, 95)),
            "n": int(vals.size)}


def processar_sorteio(celula, seed, log):
    cfg = celula["cfg"]
    fracs = tuple(float(x) for x in cfg["split_frac"].split(","))
    parts = split_espacial_3vias_exato(
        celula["pos_m"], grid_km=cfg["grid_km"], buffer_km=cfg["buffer_km"],
        fracs=fracs, split_seed=seed)

    ei = celula["ei"]
    grau_full = celula["grau_full"]
    n_nodes = celula["n_nodes"]
    ell_medio = celula["ell_medio_m"]

    from scipy import ndimage

    resultado_particoes = {}
    membro = {  # id de particao por no, para checar cruzamento
        0: parts["train"], 1: parts["val"], 2: parts["test"]}
    part_id = np.full(n_nodes, -1, dtype=np.int8)
    part_id[parts["train"]] = 0
    part_id[parts["val"]] = 1
    part_id[parts["test"]] = 2

    # arestas cruzando particao (ambas pontas retidas, mas em particoes != )
    src_pid = part_id[ei[0]]
    dst_pid = part_id[ei[1]]
    ambos_retidos = (src_pid >= 0) & (dst_pid >= 0)
    cruzam = ambos_retidos & (src_pid != dst_pid)
    n_cruzam = int(cruzam.sum())

    for nome, mask in parts.items():
        n_retidos = int(mask.sum())
        if n_retidos == 0:
            resultado_particoes[nome] = {"n_retidos": 0}
            continue
        mask2d = mask.reshape(N_SIDE, N_SIDE)
        # distancia (m) de cada no retido ao complemento da mascara (fronteira
        # do bloco retido, incluindo borda do dominio da celula: EDT trata
        # fora do array como fundo)
        dist_m = ndimage.distance_transform_edt(mask2d, sampling=(celula["ell_y_m"], celula["ell_x_m"]))
        dist_flat = dist_m.ravel()
        fronteira = mask & (dist_flat < ell_medio)
        interior = mask & ~fronteira

        # subgrafo induzido: aresta retida sse ambas pontas em `mask`
        both_in = mask[ei[0]] & mask[ei[1]]
        sub_ei = ei[:, both_in]
        grau_induzido = np.bincount(sub_ei[0], minlength=n_nodes)

        idx_retidos = np.where(mask)[0]
        idx_interior = np.where(interior)[0]
        idx_fronteira = np.where(fronteira)[0]

        antes_retidos = grau_full[idx_retidos]
        depois_retidos = grau_induzido[idx_retidos]
        antes_interior = grau_full[idx_interior]
        depois_interior = grau_induzido[idx_interior]
        antes_fronteira = grau_full[idx_fronteira]
        depois_fronteira = grau_induzido[idx_fronteira]

        with np.errstate(divide="ignore", invalid="ignore"):
            razao_retidos = np.where(antes_retidos > 0, depois_retidos / antes_retidos, np.nan)
            razao_interior = np.where(antes_interior > 0, depois_interior / antes_interior, np.nan)
            razao_fronteira = np.where(antes_fronteira > 0, depois_fronteira / antes_fronteira, np.nan)

        resultado_particoes[nome] = {
            "n_retidos": n_retidos,
            "n_interior": int(interior.sum()),
            "n_fronteira": int(fronteira.sum()),
            "frac_fronteira": float(fronteira.sum() / n_retidos),
            "grau_antes_retidos": percentis(antes_retidos),
            "grau_depois_retidos": percentis(depois_retidos),
            "grau_antes_interior": percentis(antes_interior),
            "grau_depois_interior": percentis(depois_interior),
            "grau_antes_fronteira": percentis(antes_fronteira),
            "grau_depois_fronteira": percentis(depois_fronteira),
            "razao_media_interior": float(np.nanmean(razao_interior)) if razao_interior.size else None,
            "razao_p5_interior": float(np.nanpercentile(razao_interior, 5)) if razao_interior.size else None,
            "frac_interior_razao_abaixo_0995": float(np.nanmean(razao_interior < 0.995)) if razao_interior.size else None,
            "frac_interior_razao_abaixo_099": float(np.nanmean(razao_interior < 0.99)) if razao_interior.size else None,
            "razao_media_fronteira": float(np.nanmean(razao_fronteira)) if razao_fronteira.size else None,
            "razao_p5_fronteira": float(np.nanpercentile(razao_fronteira, 5)) if razao_fronteira.size else None,
            "frac_fronteira_razao_abaixo_099": float(np.nanmean(razao_fronteira < 0.99)) if razao_fronteira.size else None,
            "razao_media_geral_retidos": float(np.nanmean(razao_retidos)) if razao_retidos.size else None,
        }

    return {
        "cidade": celula["cidade"], "q": celula["q"], "split_seed": seed,
        "n_arestas_cruzando_particao_ambas_pontas_retidas": n_cruzam,
        "particoes": resultado_particoes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--celulas", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--log", default=None)
    ap.add_argument("--so-timing", action="store_true", help="carrega 1 celula, roda 1 sorteio, mede tempo e sai")
    args = ap.parse_args()

    logf = open(args.log, "a") if args.log else None

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n"); logf.flush()

    t_inicio = time.time()
    saida = {
        "frente": "matematica-grafos-espectral",
        "teste": "1.7", "alegacao": "A2 / P3 (prop:degree)",
        "criterio": json.load(open(Path(__file__).parent.parent /
                              "_v3_2026-09-25" / "criterios" / "criterio_1.7.json")),
        "script_sha256": sha256_of_text(Path(__file__).read_text()),
        "status": "em_andamento",
    }
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    todas_celulas = [(c, q) for c in CIDADES for q in QS]
    if args.celulas >= 16:
        celulas_alvo = todas_celulas
    elif args.celulas == 4:
        celulas_alvo = [(c, "Q1") for c in CIDADES]  # reducao declarada pelo briefing: uma celula por cidade
    else:
        celulas_alvo = todas_celulas[:args.celulas]
    seeds = SEEDS_20[:args.seeds]

    if args.so_timing:
        cidade, q = celulas_alvo[0]
        t0 = time.time()
        cel = carregar_celula(cidade, q, log)
        t_carga = time.time() - t0
        t1 = time.time()
        r = processar_sorteio(cel, 42, log)
        t_sorteio = time.time() - t1
        t2 = time.time()
        sim = checar_simetria(cel["ei"], cel["n_nodes"])
        t_sim = time.time() - t2
        saida["timing_probe"] = {
            "cidade": cidade, "q": q, "tempo_carga_s": t_carga,
            "tempo_um_sorteio_s": t_sorteio, "tempo_simetria_amostra_s": t_sim,
            "simetria_amostra": sim,
            "amostra_resultado": r,
            "estimativa_16x20_min": (t_carga + t_sorteio * 20) * 16 / 60.0,
            "estimativa_4x5_min": (t_carga + t_sorteio * 5) * 4 / 60.0,
        }
        Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False, default=str))
        log(f"PROBE: carga={t_carga:.1f}s sorteio={t_sorteio:.1f}s sim={t_sim:.1f}s "
            f"est_16x20={saida['timing_probe']['estimativa_16x20_min']:.1f}min "
            f"est_4x5={saida['timing_probe']['estimativa_4x5_min']:.1f}min")
        return

    por_celula = []
    entradas_sha = {}
    for (cidade, q) in celulas_alvo:
        t0c = time.time()
        cel = carregar_celula(cidade, q, log)
        sim = checar_simetria(cel["ei"], cel["n_nodes"])
        resultados_seeds = []
        for seed in seeds:
            r = processar_sorteio(cel, seed, log)
            resultados_seeds.append(r)
            log(f"  {cidade} {q} seed={seed}: cruzam={r['n_arestas_cruzando_particao_ambas_pontas_retidas']} "
                f"razao_interior_test={r['particoes'].get('test', {}).get('razao_media_interior')}")
        entradas_sha[f"{cidade}_{q}_gpu.pt"] = "nao_calculado_por_orcamento (arquivo de 15GB; ver tamanho_bytes)"
        por_celula.append({
            "cidade": cidade, "q": q, "n_nodes": cel["n_nodes"], "n_edges": cel["n_edges"],
            "grau_medio_full": float(cel["grau_full"].mean()),
            "ell_x_m": cel["ell_x_m"], "ell_y_m": cel["ell_y_m"], "ell_medio_m": cel["ell_medio_m"],
            "simetria_amostra": sim,
            "tempo_carga_s": cel["tempo_carga_s"],
            "pt_path": cel["pt_path"], "pt_bytes": Path(cel["pt_path"]).stat().st_size,
            "sorteios": resultados_seeds,
            "tempo_total_celula_s": time.time() - t0c,
        })
        Path(args.out).write_text(json.dumps({**saida, "por_celula": por_celula, "status": "em_andamento"},
                                              indent=1, ensure_ascii=False, default=str))
        log(f"celula {cidade} {q} concluida em {time.time()-t0c:.1f}s")

    # -------- resumo e veredito --------
    razoes_interior_todas = []
    razoes_fronteira_todas = []
    fracs_interior_abaixo_0995 = []
    n_cruzam_total = 0
    piores_interior = []
    for c in por_celula:
        for s in c["sorteios"]:
            n_cruzam_total += s["n_arestas_cruzando_particao_ambas_pontas_retidas"]
            for pnome, p in s["particoes"].items():
                if p.get("n_retidos", 0) == 0:
                    continue
                if p.get("razao_media_interior") is not None:
                    razoes_interior_todas.append(p["razao_media_interior"])
                    fracs_interior_abaixo_0995.append(p["frac_interior_razao_abaixo_0995"])
                    if p["razao_media_interior"] < 0.995:
                        piores_interior.append({"cidade": c["cidade"], "q": c["q"],
                                                 "seed": s["split_seed"], "particao": pnome,
                                                 "razao_media_interior": p["razao_media_interior"]})
                if p.get("razao_media_fronteira") is not None:
                    razoes_fronteira_todas.append(p["razao_media_fronteira"])

    criterio_interior_ok = (len(razoes_interior_todas) > 0 and min(razoes_interior_todas) >= 0.995)
    perda_so_fronteira = (len(piores_interior) == 0)

    veredito = "P3_SUSTENTADA_COM_HIPOTESES" if (criterio_interior_ok and perda_so_fronteira) else "P3_VIRA_REMARK"

    # formula do termo de fronteira sustentada pelos dados: perda_particao ~=
    # frac_fronteira * perda_media_fronteira (perda_media_interior ~ 0)
    formula_fronteira = []
    for c in por_celula:
        for s in c["sorteios"]:
            for pnome, p in s["particoes"].items():
                if p.get("n_retidos", 0) == 0 or p.get("razao_media_geral_retidos") is None:
                    continue
                perda_medida = 1.0 - p["razao_media_geral_retidos"]
                perda_fronteira = 1.0 - (p["razao_media_fronteira"] if p["razao_media_fronteira"] is not None else 1.0)
                termo_previsto = p["frac_fronteira"] * perda_fronteira
                formula_fronteira.append({
                    "cidade": c["cidade"], "q": c["q"], "seed": s["split_seed"], "particao": pnome,
                    "perda_relativa_medida_geral": perda_medida,
                    "frac_fronteira": p["frac_fronteira"],
                    "perda_media_fronteira": perda_fronteira,
                    "termo_frac_x_perda_fronteira": termo_previsto,
                })

    saida["por_celula"] = por_celula
    saida["arestas_cruzando_particao"] = {
        "total_observado_todos_sorteios": n_cruzam_total,
        "esperado": 0,
        "confirma_zero_por_construcao": (n_cruzam_total == 0),
    }
    saida["resumo"] = {
        "n_celulas": len(por_celula), "n_seeds_por_celula": len(seeds),
        "n_amostras_particao_x_sorteio": len(razoes_interior_todas),
        "razao_interior_min": min(razoes_interior_todas) if razoes_interior_todas else None,
        "razao_interior_media": float(np.mean(razoes_interior_todas)) if razoes_interior_todas else None,
        "razao_fronteira_min": min(razoes_fronteira_todas) if razoes_fronteira_todas else None,
        "razao_fronteira_media": float(np.mean(razoes_fronteira_todas)) if razoes_fronteira_todas else None,
        "casos_interior_abaixo_0995": piores_interior,
    }
    saida["formula_termo_fronteira_sustentada_pelos_dados"] = {
        "descricao": ("perda_relativa_da_particao (1 - grau_medio_pos/grau_medio_pre, sobre todos os "
                      "nos retidos) e aproximada por frac_fronteira x perda_media_fronteira, com "
                      "perda_media_interior ~ 0 (ver resumo.razao_interior_media); hipoteses: malha "
                      "regular, arestas terreno-terreno materializadas (nao kNN por feature), "
                      "fronteira retificavel (perimetro finito por bloco)."),
        "detalhe": formula_fronteira,
    }
    saida["veredito_vs_criterio"] = veredito
    saida["nao_verificado"] = [
        {"item": "arestas antena-terreno", "motivo": "P3-1/P3 tratam grau terra-terra; nao incluido por orcamento (ver criterio_1.7.json)"},
        {"item": "simetria exaustiva (100% das arestas)", "motivo": "checada por amostra (checar_simetria, ate 500k arestas por celula); ver campo simetria_amostra por celula"},
        {"item": "16x20 completo" if len(celulas_alvo) < 16 or len(seeds) < 20 else None,
         "motivo": f"executado {len(celulas_alvo)} celulas x {len(seeds)} sorteios por orcamento de CPU (ver escopo)"},
    ]
    saida["nao_verificado"] = [n for n in saida["nao_verificado"] if n.get("item")]
    saida["comando_rodado"] = (
        f".venv/bin/python v3_1.7_grau_preservacao.py --out {args.out} "
        f"--celulas {args.celulas} --seeds {args.seeds}")
    saida["status"] = "concluido_provisorio"
    saida["pico_ram_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    saida["tempo_total_s"] = time.time() - t_inicio
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False, default=str))
    log(f"CONCLUIDO. tempo_total={saida['tempo_total_s']:.1f}s pico_ram={saida['pico_ram_gb']:.2f}GB "
        f"veredito={veredito}")


if __name__ == "__main__":
    main()
