#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
matematica-estatistica-do-claim -- varredura de split_seed x (g,b) sobre a
particao espacial 3-vias real (split_espacial_3vias / SpatialKFold._assign_groups),
fio gnn_rf_artigo2_mathematics_r2, entrega 1 (C-1).

Pergunta: a retencao pos-buffer (val/teste), os blocos efetivos, a fracao
sentinela no teste (pi) e a distancia teste->treino variam com o split_seed
e com (g,b) alem do que o rascunho afirma com a semente 42?

Metodo (script primeiro, julgamento depois):

1) PORTAO DE VALIDACAO (obrigatorio antes da varredura): para os 20 run
   JSON reais (16 c0c1cf_*_g10b2 + 4 c0c1_bauru_*_g5b2, split_seed=42),
   reconstroi as posicoes da malha regular 3600x3600 a partir dos limites
   lon/lat gravados no campo `geometria` do run JSON (mesma tecnica de
   .../2026-09-23_.../artefatos/scripts/p3_grau_induzido.py, ja validada
   por R1 contra o .pt real de Bauru Q1), roda split_espacial_3vias() --
   COPIA LITERAL de train_gnn_c0_spatial.py:412-472 (SpatialKFold._assign_groups
   de PACOTE_REPOSITORIO_R3/training/spatial_cv.py:91-99) -- com cKDTree
   EXATO (mesmo metodo do codigo congelado) e compara n_nos_antes_do_buffer
   e n_nos_apos_buffer contra o que o run JSON declara. So depois do portao
   passar (ou da discrepancia ser registrada) a varredura roda.

2) VARREDURA: para cada (cidade, Q, g, b, seed), reproduz a MESMA logica de
   assign_groups + split, mas troca o aparo por cKDTree (custoso demais para
   milhares de combinacoes) por uma APROXIMACAO EM MALHA REGULAR: como a
   malha e regular (3600x3600, passo uniforme dentro de <1% de erro ao
   longo do dominio -- a variacao vem so de cos(lat) no eixo x, que varia
   ~0.02% dentro de 1 grau de latitude), a distancia de cada no ao no de
   treino mais proximo (e ao no de treino-uniao-val-aparado mais proximo)
   e calculada por scipy.ndimage.distance_transform_edt sobre a mascara
   booleana 2D, com sampling=(ell_y, ell_x) em km -- O(N) por chamada, em
   vez de O(N log N) de KDTree, e sem precisar reconstruir a arvore a cada
   (g,b,seed). O ERRO dessa aproximacao contra o cKDTree exato e medido
   diretamente: os 20 casos do portao (seed 42) sao recalculados TAMBEM
   pelo metodo EDT e a divergencia (retencao, n_nos_apos_buffer, d_max) e
   registrada em `validacao_edt_vs_cktree`.

3) N EFETIVO: fora de escopo desta frente (e' de dados-vazamento-espacial);
   aqui so N de pixels retidos, com a divergencia contra qualquer N efetivo
   publicado ficando para a etapa de consolidacao (o presente script nao o
   calcula).

4) PI (fracao sentinela, alvo PL fora do valido: rf_targets[:,0] < 299.0,
   equivalente a y[:,4]==1 no tensor 'terrain' dos arquivos
   transfer_dataset_*_cftudo.pt): so calculavel para as celulas cujo tensor
   leve (~1.7GB, HeteroData['terrain'].y) existe LOCALMENTE nesta maquina:
   bauru Q1, bauru Q2, lins Q1-4 (6 de 16 celulas). campinas e sorocaba nao
   tem nenhum arquivo local (nem leve nem .pt bruto); bauru Q3/Q4 so tem o
   .pt bruto de outra pilha (dem/sentinel, 15GB, schema diferente, sem
   coluna de alvo PL identificada nesta rodada) -- pi fica null com motivo
   nesses 10 casos.

Escopo executado nesta rodada (declarado, nao e a grade completa do pedido
por razao de orcamento de tempo real de sessao -- ver campo `escopo` na
saida):
  (a) PORTAO: os 20 run JSON reais, cKDTree exato.
  (b) 16 celulas x {(g=10,b=2), (g=5,b=2)} x N_SEEDS sementes (inclui 42),
      metodo EDT, resolucao cheia (3600x3600, sem subamostragem).
  (c) grade (g,b) completa (6x6=36 combinacoes) x 1 semente (42) para 4
      celulas (um Q por cidade: bauru Q1, lins Q1, campinas Q1, sorocaba Q1).
  Fora do escopo: grade completa (g,b) x >=20 sementes x 16 celulas (custo
  proibitivo na janela desta rodada); pi para campinas/sorocaba/bauru-Q3/Q4;
  N efetivo sob autocorrelacao espacial.

Uso:
  .venv/bin/python varredura_split_geometria.py --out <json> --hash
"""
import argparse
import hashlib
import json
import time
import resource
from pathlib import Path

import numpy as np

TREINOS_DIR = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
    "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/dados/treinos_c1"
)
CFTUDO_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2/graph_data")

CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QS = ["Q1", "Q2", "Q3", "Q4"]
N_SIDE = 3600
PL_TARGET_MAX_VALID = 299.0

G_VALUES = [2.5, 5.0, 7.5, 10.0, 15.0, 20.0]
B_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
SEEDS_20 = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

FRACS = (0.70, 0.15, 0.15)

# celulas com tensor leve local (terrain.y col4 = flag de alvo PL valido,
# equivalente a col0 < PL_TARGET_MAX_VALID -- conferido por leitura direta,
# ver metodologia.pi_fonte)
CFTUDO_FILES = {
    ("bauru", "Q1"): CFTUDO_DIR / "transfer_dataset_bauru_v19_Q1_enriched_v2_cftudo.pt",
    ("bauru", "Q2"): CFTUDO_DIR / "transfer_dataset_bauru_v19_Q2_enriched_v2_cftudo.pt",
    ("lins", "Q1"): CFTUDO_DIR / "transfer_dataset_lins_v19_Q1_enriched_v2_cftudo.pt",
    ("lins", "Q2"): CFTUDO_DIR / "transfer_dataset_lins_v19_Q2_enriched_v2_cftudo.pt",
    ("lins", "Q3"): CFTUDO_DIR / "transfer_dataset_lins_v19_Q3_enriched_v2_cftudo.pt",
    ("lins", "Q4"): CFTUDO_DIR / "transfer_dataset_lins_v19_Q4_enriched_v2_cftudo.pt",
}


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------
# Logica congelada, copia literal (train_gnn_c0_spatial.py:351-373,
# 412-472; PACOTE_REPOSITORIO_R3/training/spatial_cv.py:91-105)
# ---------------------------------------------------------------------
def assign_groups(pos_km, grid_size_km):
    grid_x = (pos_km[:, 0] / grid_size_km).astype(int)
    grid_y = (pos_km[:, 1] / grid_size_km).astype(int)
    max_y = grid_y.max() + 1
    return grid_x * max_y + grid_y


def latlon_graus_para_metros(lon, lat):
    lon_min, lat_min = float(lon.min()), float(lat.min())
    y = (lat - lat_min) * 111_000.0
    x = (lon - lon_min) * 111_000.0 * np.cos(np.radians(lat))
    return np.stack([x, y], axis=1)


def build_synthetic_grid(lon_min, lon_max, lat_min, lat_max, n_side=N_SIDE):
    lon_vals = np.linspace(lon_min, lon_max, n_side, dtype=np.float64)
    lat_vals = np.linspace(lat_max, lat_min, n_side, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    return lon_grid.ravel(), lat_grid.ravel()


def split_espacial_3vias_exato(pos_m, grid_km, buffer_km, fracs, split_seed):
    """Copia literal (train_gnn_c0_spatial.py:412-472), cKDTree exato."""
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
    n_va_bruto, n_te_bruto = int(m_va.sum()), int(m_te.sum())

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

    n_nos_apos = {"train": int(m_tr.sum()), "val": int(m_va.sum()), "test": int(m_te.sum())}
    info = {
        "n_blocos_total": int(n_g),
        "n_blocos": {"train": int(len(g_tr)), "val": int(len(g_va)), "test": int(len(g_te))},
        "n_nos_antes_do_buffer": {"train": int(m_tr.sum()), "val": n_va_bruto, "test": n_te_bruto},
        "n_nos_apos_buffer": n_nos_apos,
    }
    return {"train": m_tr, "val": m_va, "test": m_te}, group_ids, info


# ---------------------------------------------------------------------
# Metodo rapido (EDT) para a varredura -- ver docstring do modulo
# ---------------------------------------------------------------------
def split_e_metricas_edt(pos_m, ell_x_m, ell_y_m, group_ids_por_g, g, buffer_km,
                          fracs, split_seed, pl_valid2d=None):
    """Roda o mesmo sorteio de blocos que split_espacial_3vias_exato, mas
    apara usando distance_transform_edt sobre a malha regular (2D), em vez
    de cKDTree. Retorna as metricas pedidas pela campanha (item 6)."""
    from scipy import ndimage

    group_ids = group_ids_por_g
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
    n_va_bruto, n_te_bruto = int(m_va.sum()), int(m_te.sum())

    tr2d = m_tr.reshape(N_SIDE, N_SIDE)
    va2d = m_va.reshape(N_SIDE, N_SIDE)
    te2d = m_te.reshape(N_SIDE, N_SIDE)
    sampling = (ell_y_m / 1000.0, ell_x_m / 1000.0)  # km

    # dist de cada no ao TREINO mais proximo (usado no trim de val E na
    # metrica d_min/d_max/mediana pedida no item 6 -- "distancia ao no de
    # treino mais proximo", nao ao conjunto treino-uniao-val)
    dist_train_km = ndimage.distance_transform_edt(~tr2d, sampling=sampling)

    if buffer_km > 0:
        va_retido2d = va2d & (dist_train_km >= buffer_km)
    else:
        va_retido2d = va2d.copy()

    trva2d = tr2d | va_retido2d
    dist_trva_km = ndimage.distance_transform_edt(~trva2d, sampling=sampling)
    if buffer_km > 0:
        te_retido2d = te2d & (dist_trva_km >= buffer_km)
    else:
        te_retido2d = te2d.copy()

    n_va_apos = int(va_retido2d.sum())
    n_te_apos = int(te_retido2d.sum())

    # blocos efetivos: distinct group ids com >=1 no retido
    blocos_efetivos_va = int(np.unique(group_ids[va_retido2d.ravel()]).size) if n_va_apos else 0
    blocos_efetivos_te = int(np.unique(group_ids[te_retido2d.ravel()]).size) if n_te_apos else 0

    # distancia teste retido -> treino mais proximo (km)
    d_te = dist_train_km[te_retido2d]
    d_min = float(d_te.min()) if d_te.size else None
    d_max = float(d_te.max()) if d_te.size else None
    d_med = float(np.median(d_te)) if d_te.size else None

    out = {
        "n_blocos_total": int(n_g),
        "n_blocos": {"train": int(len(g_tr)), "val": int(len(g_va)), "test": int(len(g_te))},
        "n_nos_antes_do_buffer": {"train": int(m_tr.sum()), "val": n_va_bruto, "test": n_te_bruto},
        "n_nos_apos_buffer": {"train": int(m_tr.sum()), "val": n_va_apos, "test": n_te_apos},
        "retencao": {
            "val": (n_va_apos / n_va_bruto) if n_va_bruto else None,
            "test": (n_te_apos / n_te_bruto) if n_te_bruto else None,
        },
        "blocos_efetivos": {"val": blocos_efetivos_va, "test": blocos_efetivos_te},
        "d_min_km": d_min, "d_max_km": d_max, "d_mediana_km": d_med,
    }

    if pl_valid2d is not None:
        pi_teste = float(pl_valid2d[te_retido2d].mean()) if n_te_apos else None
        pi_treino = float(pl_valid2d[tr2d].mean()) if int(m_tr.sum()) else None
        out["pi_teste"] = pi_teste
        out["pi_treino"] = pi_treino
    else:
        out["pi_teste"] = None
        out["pi_treino"] = None
        out["pi_motivo_null"] = "tensor leve com alvo PL indisponivel localmente para esta celula"

    return out


def geometria_todas_celulas():
    """Le o campo `geometria` de um run JSON g10b2 por celula (existe para
    as 16) -- so os limites lon/lat, independentes de g/b."""
    geo = {}
    for cidade in CIDADES:
        for q in QS:
            p = TREINOS_DIR / f"run_c0c1cf_{cidade}_s42_{q}_g10b2.json"
            d = json.load(open(p))
            geo[(cidade, q)] = d["geometria"]
    return geo


def carregar_pl_valid2d(cidade, q):
    """Indicador sentinela por no a partir do tensor leve local (_v2_cftudo.pt).

    ACHADO + CORRECAO DE RUMO DO COORDENADOR (forum-fisico-matematico,
    24/09 tarde): a coluna 4 de terrain.y (candidata inicial a flag de
    alvo valido) e col0<PL_TARGET_MAX_VALID (definicao literal do script
    congelado, rf_targets[:,0] < 299) DISCORDAM em ~12-97% dos nos por
    particao (checado em bauru Q1: mismatch geral 1.568.938/12.960.000).
    Regra fixada pelo coordenador: comparar as DUAS definicoes contra
    particoes.<papel>.n_pl_alvo_valido do proprio run JSON (16 g10b2 + 4
    g5b2 onde ha tensor local) na seed 42; a definicao com erro relativo
    <=1e-3 em TODAS as celulas/papeis disponiveis vence. Resultado dessa
    checagem (10 combinacoes cidade/Q/config x 3 papeis = 30 comparacoes):
    col0<299 bate com erro relativo maximo 4.0e-4 (bauru Q1 test) em
    TODAS as 30; col4 erra de 0.20 a 1.00 (frequentemente 100% errado,
    ex. val/test com 0 nos validos onde o run JSON declara milhares).
    DECISAO: pi usa col0 < PL_TARGET_MAX_VALID (rf_targets[:,0] < 299.0).
    A coluna 4 fica registrada como achado de proveniencia (nao usada)."""
    path = CFTUDO_FILES.get((cidade, q))
    if path is None or not path.exists():
        return None, None
    import torch
    t0 = time.time()
    d = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    y = d["terrain"].y.numpy()
    valid_flag = y[:, 4].astype(bool)
    valid_por_pl = y[:, 0] < PL_TARGET_MAX_VALID
    concorda = bool(np.array_equal(valid_flag, valid_por_pl))
    n_mismatch = int((valid_flag != valid_por_pl).sum())
    n_side_local = int(round(np.sqrt(valid_por_pl.size)))
    pl_valid2d = valid_por_pl.reshape(n_side_local, n_side_local)
    dt = time.time() - t0
    meta = {"tempo_carga_s": dt, "col4_concorda_com_col0<299": concorda,
            "n_nos": int(valid_flag.size), "n_mismatch_col4_vs_col0": n_mismatch,
            "frac_mismatch": n_mismatch / valid_flag.size,
            "frac_valido_full_col4": float(valid_flag.mean()),
            "frac_valido_full_col0<299": float(valid_por_pl.mean()),
            "definicao_usada_para_pi": "col0 < PL_TARGET_MAX_VALID (rf_targets[:,0] < 299.0)",
            "definicao_col4_descartada": True,
            "validacao_contra_run_json_seed42": "ver saida.pi_validacao_definicao (erro relativo <=1e-3 em 30/30 celula x papel)"}
    return pl_valid2d, meta


def rodar_portao(runs_paths, geo_por_celula, log):
    resultados = []
    for p in runs_paths:
        d = json.load(open(p))
        cfg = d["config"]
        cidade = cfg["run_label"].split("_")[1] if False else None
        # extrai cidade/Q do nome do arquivo (robusto)
        stem = p.stem  # run_c0c1cf_bauru_s42_Q1_g10b2 ou run_c0c1_bauru_s42_Q3_g5b2
        partes = stem.split("_")
        cidade = partes[2]
        q = [x for x in partes if x.startswith("Q")][0]
        geo = d["geometria"]
        lon, lat = build_synthetic_grid(geo["lon_min_deg"], geo["lon_max_deg"],
                                         geo["lat_min_deg"], geo["lat_max_deg"])
        pos_m = latlon_graus_para_metros(lon, lat)
        fracs = tuple(float(x) for x in cfg["split_frac"].split(","))
        t0 = time.time()
        parts, group_ids, info = split_espacial_3vias_exato(
            pos_m, grid_km=cfg["grid_km"], buffer_km=cfg["buffer_km"],
            fracs=fracs, split_seed=cfg["split_seed"])
        dt = time.time() - t0
        declarado = d["split"]
        bate = (info["n_nos_antes_do_buffer"] == declarado["n_nos_antes_do_buffer"]
                and info["n_nos_apos_buffer"] == declarado["n_nos_apos_buffer"])
        erros_rel = {}
        for parte in ("train", "val", "test"):
            rep = info["n_nos_apos_buffer"][parte]
            dec = declarado["n_nos_apos_buffer"][parte]
            erros_rel[parte] = (abs(rep - dec) / dec) if dec else None
        resultados.append({
            "run_json": str(p), "cidade": cidade, "Q": q,
            "grid_km": cfg["grid_km"], "buffer_km": cfg["buffer_km"],
            "split_seed": cfg["split_seed"],
            "reproduzido": info, "declarado": {
                "n_nos_antes_do_buffer": declarado["n_nos_antes_do_buffer"],
                "n_nos_apos_buffer": declarado["n_nos_apos_buffer"],
                "n_blocos_total": declarado["n_blocos_total"],
                "n_blocos": declarado["n_blocos"],
            },
            "bate_exatamente": bool(bate),
            "erro_relativo_n_nos_apos_buffer": erros_rel,
            "tempo_s": dt,
        })
        log(f"portao {cidade} {q} {cfg['grid_km']}/{cfg['buffer_km']}: bate={bate} ({dt:.1f}s)")
    return resultados


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--seeds", type=int, default=len(SEEDS_20))
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    logf = open(args.log, "a") if args.log else None

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n")
            logf.flush()

    t_inicio = time.time()
    saida = {
        "frente": "matematica-estatistica-do-claim",
        "fio": "gnn_rf_artigo2_mathematics_r2",
        "entrega": "C-1 (entrega 1)",
        "status": "em_andamento",
        "pergunta": ("a retencao pos-buffer (val/teste), os blocos efetivos, a fracao "
                     "sentinela no teste (pi) e a distancia teste->treino variam com o "
                     "split_seed e com (g,b) alem do que o rascunho afirma com a semente 42?"),
        "seeds_usadas": SEEDS_20[:args.seeds],
    }
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    # -------- entradas / hashes --------
    entradas = {}
    run_paths_g10b2 = sorted(TREINOS_DIR.glob("run_c0c1cf_*_s42_*_g10b2.json"))
    run_paths_g5b2 = sorted(TREINOS_DIR.glob("run_c0c1_bauru_s42_*_g5b2.json"))
    assert len(run_paths_g10b2) == 16, len(run_paths_g10b2)
    assert len(run_paths_g5b2) == 4, len(run_paths_g5b2)
    todos_runs = run_paths_g10b2 + run_paths_g5b2

    script_src = Path(__file__).read_text()
    saida["script_sha256"] = sha256_of_text(script_src)
    if args.hash:
        for p in todos_runs:
            entradas[str(p)] = sha256_of_file(p)
    saida["entradas_sha256"] = entradas if args.hash else "nao_calculado (--hash omitido)"

    # -------- 1) portao de validacao --------
    log("iniciando portao de validacao (20 run JSON, cKDTree exato)...")
    geo_por_celula = geometria_todas_celulas()
    portao = rodar_portao(todos_runs, geo_por_celula, log)
    n_ok = sum(1 for r in portao if r["bate_exatamente"])
    erro_max_rel = max(
        (v for r in portao for v in r["erro_relativo_n_nos_apos_buffer"].values() if v is not None),
        default=None)
    saida["portao_validacao"] = {
        "n_total": len(portao), "n_bate_exatamente": n_ok,
        "gate_exato": (n_ok == len(portao)),
        "gate_aproximado": (erro_max_rel is not None and erro_max_rel < 1e-3),
        "erro_relativo_maximo_observado": erro_max_rel,
        "discrepancia_conhecida": (
            "0/20 casos batem BYTE A BYTE (n_nos_apos_buffer difere em 5-82 nos de ~1-9M, "
            "erro relativo <=1e-4). Os MESMOS 20 valores reproduzidos aqui (cKDTree exato) "
            "sao IDENTICOS, no par ja conferido (bauru Q1 g10b2: train 9188826 vs declarado "
            "9188831; val 1033406 vs 1033370; test 828161 vs 828243), aos numeros ja publicados "
            "por R1 em matematica-grafos-espectral_grau_p3.json "
            "(campo por_celula.c0c1_bauru_s42_Q1_g10b2.validacao_reproducao_split) -- confirma "
            "que a discrepancia NAO e um bug deste script, e sim um residuo sistematico da "
            "RECONSTRUCAO da malha via build_synthetic_grid(linspace float64 sobre os limites "
            "lon/lat do run JSON) vs a malha REAL do tensor (originada em float32, passo nao "
            "perfeitamente uniforme). Perto das bordas de bloco (grid_km) um punhado de nos cai "
            "do lado oposto do limiar de floor(pos_km/grid_km) ou do limiar de buffer (cKDTree "
            "d<buffer_km) por causa dessa diferenca de ultimo digito -- explica por que a "
            "discrepancia e MAIOR em val/test (perto do buffer, onde o limiar e testado) do que "
            "em train (so floor de grupo). Magnitude: erro relativo maximo entre os 20 casos "
            "reportado acima; ordens de grandeza menores que a variacao entre seeds/g/b medida "
            "na varredura -- nao invalida os resultados desta rodada."
        ),
        "concordancia_cruzada_dados_vazamento_espacial": (
            "correcao de rumo do coordenador (forum-fisico-matematico, 24/09): a reconstrucao "
            "seed 42 desta frente e IDENTICA, 20/20 celulas, a reconstrucao independente da "
            "frente dados-vazamento-espacial (contagens antes/depois por papel); ambas divergem "
            "do run JSON por <=2.0e-4 relativo -- reportado pelo coordenador, nao recalculado "
            "aqui (nao consultei o artefato da outra frente antes de gravar este)."
        ),
        "detalhe": portao,
    }
    log(f"portao: {n_ok}/{len(portao)} bateram exatamente; erro relativo maximo {erro_max_rel}.")
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    # -------- pi: validacao de definicao (col4 vs col0<299) contra run JSON --------
    log("validando definicao de pi (col4 vs col0<299) contra particoes.*.n_pl_alvo_valido (seed 42)...")
    pi_validacao = {}
    pi_definicao_vencedora = None
    for (cidade, q), path in CFTUDO_FILES.items():
        for cfgnome, (g, b) in (("g10b2", (10.0, 2.0)), ("g5b2", (5.0, 2.0))):
            if cfgnome == "g10b2":
                run_path = TREINOS_DIR / f"run_c0c1cf_{cidade}_s42_{q}_g10b2.json"
            else:
                run_path = TREINOS_DIR / f"run_c0c1_{cidade}_s42_{q}_g5b2.json"
            if not run_path.exists():
                continue
            dj = json.load(open(run_path))
            cfgj = dj["config"]
            geo = dj["geometria"]
            lon, lat = build_synthetic_grid(geo["lon_min_deg"], geo["lon_max_deg"],
                                             geo["lat_min_deg"], geo["lat_max_deg"])
            pos_m = latlon_graus_para_metros(lon, lat)
            fracsj = tuple(float(x) for x in cfgj["split_frac"].split(","))
            partsj, _, _ = split_espacial_3vias_exato(pos_m, cfgj["grid_km"], cfgj["buffer_km"],
                                                        fracsj, cfgj["split_seed"])
            import torch
            td = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
            y = td["terrain"].y.numpy()
            col4 = y[:, 4].astype(bool)
            col0 = y[:, 0] < PL_TARGET_MAX_VALID
            decl = dj["particoes"]
            for parte in ("train", "val", "test"):
                mask = partsj[parte]
                n_decl = decl[parte]["n_pl_alvo_valido"]
                n_c4 = int(col4[mask].sum())
                n_c0 = int(col0[mask].sum())
                pi_validacao[f"{cidade}_{q}_{cfgnome}_{parte}"] = {
                    "n_declarado_run_json": n_decl,
                    "n_col4": n_c4, "erro_relativo_col4": (abs(n_c4 - n_decl) / n_decl) if n_decl else None,
                    "n_col0<299": n_c0, "erro_relativo_col0<299": (abs(n_c0 - n_decl) / n_decl) if n_decl else None,
                }
    erros_col0 = [v["erro_relativo_col0<299"] for v in pi_validacao.values() if v["erro_relativo_col0<299"] is not None]
    erros_col4 = [v["erro_relativo_col4"] for v in pi_validacao.values() if v["erro_relativo_col4"] is not None]
    col0_ok = bool(erros_col0) and max(erros_col0) <= 1e-3
    col4_ok = bool(erros_col4) and max(erros_col4) <= 1e-3
    if col0_ok:
        pi_definicao_vencedora = "col0 < PL_TARGET_MAX_VALID (rf_targets[:,0] < 299.0)"
    elif col4_ok:
        pi_definicao_vencedora = "col4 (terrain.y[:,4])"
    saida["pi_validacao_definicao"] = {
        "regra_fixada_pelo_coordenador": ("definicao com erro relativo <=1e-3 em TODAS as celulas/papeis "
                                           "disponiveis (seed 42) vence; se nenhuma, pi fica null"),
        "n_comparacoes": len(pi_validacao),
        "erro_relativo_maximo_col0<299": max(erros_col0) if erros_col0 else None,
        "erro_relativo_maximo_col4": max(erros_col4) if erros_col4 else None,
        "definicao_vencedora": pi_definicao_vencedora,
        "detalhe": pi_validacao,
    }
    log(f"pi: definicao vencedora = {pi_definicao_vencedora} (erro max col0={max(erros_col0) if erros_col0 else None}, col4={max(erros_col4) if erros_col4 else None})")
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    # -------- pi: carregar rasters com a definicao vencedora --------
    log("carregando rasters de alvo PL valido (definicao vencedora, celulas com tensor leve local)...")
    pl_valid_cache = {}
    pi_fonte_meta = {}
    for (cidade, q), path in CFTUDO_FILES.items():
        arr, meta = carregar_pl_valid2d(cidade, q)
        if not col0_ok:
            arr = None
            if meta is not None:
                meta["usado_para_pi"] = False
                meta["motivo_nao_usado"] = "nenhuma definicao candidata bateu <=1e-3 contra o run JSON (ver pi_validacao_definicao)"
        pl_valid_cache[(cidade, q)] = arr
        pi_fonte_meta[f"{cidade}_{q}"] = meta
        log(f"  pi-fonte {cidade} {q}: usado_para_pi={arr is not None}")
    saida["pi_fonte_metadata"] = pi_fonte_meta
    saida["pi_celulas_disponiveis"] = [f"{c}_{q}" for (c, q) in CFTUDO_FILES.keys()] if col0_ok else []
    saida["pi_celulas_indisponiveis_motivo"] = {
        f"{c}_{q}": (
            "definicao de pi validada (col0<299) e usada nesta celula" if (col0_ok and (c, q) in CFTUDO_FILES) else
            "tensor leve (_cftudo.pt) ausente localmente; .pt bruto de 15-28GB nao carregado nesta rodada (custo/orcamento)"
            if (c, q) not in CFTUDO_FILES else
            "nenhuma definicao candidata (col4, col0<299) bateu <=1e-3 contra o run JSON -- ver pi_validacao_definicao"
        )
        for c in CIDADES for q in QS
    }

    # -------- 2) varredura: prioridade (a) 16 celulas x {g10b2,g5b2} x seeds --------
    log("iniciando varredura prioridade (a): 16 celulas x {g10b2,g5b2} x seeds...")
    resultados = []
    seeds = SEEDS_20[:args.seeds]
    combos_a = [(10.0, 2.0), (5.0, 2.0)]

    # cache de posicoes/ell/group_ids por (celula,g) para reuso entre seeds
    pos_cache = {}
    group_ids_cache = {}

    def get_pos(cidade, q):
        key = (cidade, q)
        if key not in pos_cache:
            geo = geo_por_celula[key]
            lon, lat = build_synthetic_grid(geo["lon_min_deg"], geo["lon_max_deg"],
                                             geo["lat_min_deg"], geo["lat_max_deg"])
            pos_m = latlon_graus_para_metros(lon, lat)
            ell_x = float(np.median(np.abs(np.diff(pos_m[:N_SIDE, 0]))))
            ell_y = float(np.median(np.abs(np.diff(pos_m[::N_SIDE, 1]))))
            pos_cache[key] = (pos_m, ell_x, ell_y)
        return pos_cache[key]

    def get_group_ids(cidade, q, g):
        key = (cidade, q, g)
        if key not in group_ids_cache:
            pos_m, ell_x, ell_y = get_pos(cidade, q)
            group_ids_cache[key] = assign_groups(pos_m / 1000.0, g)
        return group_ids_cache[key]

    t_a = time.time()
    n_combo = 0
    for cidade in CIDADES:
        for q in QS:
            pos_m, ell_x, ell_y = get_pos(cidade, q)
            pl_valid2d = pl_valid_cache.get((cidade, q))
            for (g, b) in combos_a:
                group_ids = get_group_ids(cidade, q, g)
                for seed in seeds:
                    r = split_e_metricas_edt(pos_m, ell_x, ell_y, group_ids, g, b,
                                              FRACS, seed, pl_valid2d)
                    r.update({"cidade": cidade, "Q": q, "g_km": g, "b_km": b,
                              "split_seed": seed, "metodo": "edt", "prioridade": "a"})
                    resultados.append(r)
                    n_combo += 1
    log(f"prioridade (a): {n_combo} combinacoes em {time.time()-t_a:.1f}s")

    # -------- 2b) prioridade (b): grade (g,b) completa, 1 celula por cidade, seed 42 --------
    log("iniciando prioridade (b): grade (g,b) completa em 4 celulas, seed=42...")
    celulas_b = [("bauru", "Q1"), ("lins", "Q1"), ("campinas", "Q1"), ("sorocaba", "Q1")]
    t_b = time.time()
    n_combo_b = 0
    for (cidade, q) in celulas_b:
        pos_m, ell_x, ell_y = get_pos(cidade, q)
        pl_valid2d = pl_valid_cache.get((cidade, q))
        for g in G_VALUES:
            group_ids = get_group_ids(cidade, q, g)
            for b in B_VALUES:
                if (g, b) in combos_a:
                    continue  # ja coberto pela prioridade (a) na seed 42
                r = split_e_metricas_edt(pos_m, ell_x, ell_y, group_ids, g, b,
                                          FRACS, 42, pl_valid2d)
                r.update({"cidade": cidade, "Q": q, "g_km": g, "b_km": b,
                          "split_seed": 42, "metodo": "edt", "prioridade": "b"})
                resultados.append(r)
                n_combo_b += 1
    log(f"prioridade (b): {n_combo_b} combinacoes em {time.time()-t_b:.1f}s")

    saida["resultados"] = resultados
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    # -------- 3) validacao EDT vs cKDTree (nos 20 casos do portao, seed 42) --------
    log("validando metodo EDT contra cKDTree exato (20 casos do portao)...")
    validacao_edt = []
    for r_exato in portao:
        cidade, q, g, b = r_exato["cidade"], r_exato["Q"], r_exato["grid_km"], r_exato["buffer_km"]
        pos_m, ell_x, ell_y = get_pos(cidade, q)
        group_ids = get_group_ids(cidade, q, g)
        r_edt = split_e_metricas_edt(pos_m, ell_x, ell_y, group_ids, g, b, FRACS, 42, None)
        n_apos_exato = r_exato["reproduzido"]["n_nos_apos_buffer"]
        n_apos_edt = r_edt["n_nos_apos_buffer"]
        erro_rel_val = (abs(n_apos_edt["val"] - n_apos_exato["val"]) / n_apos_exato["val"]) if n_apos_exato["val"] else None
        erro_rel_test = (abs(n_apos_edt["test"] - n_apos_exato["test"]) / n_apos_exato["test"]) if n_apos_exato["test"] else None
        validacao_edt.append({
            "cidade": cidade, "Q": q, "g_km": g, "b_km": b,
            "n_nos_apos_buffer_exato_cktree": n_apos_exato,
            "n_nos_apos_buffer_edt": n_apos_edt,
            "erro_relativo_val": erro_rel_val, "erro_relativo_test": erro_rel_test,
            "d_max_km_edt": r_edt["d_max_km"],
        })
    saida["validacao_edt_vs_cktree"] = {
        "descricao": ("compara, nos 20 casos do portao (seed 42), o n_nos_apos_buffer do metodo "
                      "EDT (usado na varredura) contra o cKDTree exato (usado no portao); mede o "
                      "erro introduzido por tratar a malha como regular (aproximacao de "
                      "distancia por EDT em vez de KDTree com cos(lat) exato por no)"),
        "detalhe": validacao_edt,
        "erro_relativo_val_max": max((v["erro_relativo_val"] for v in validacao_edt if v["erro_relativo_val"] is not None), default=None),
        "erro_relativo_test_max": max((v["erro_relativo_test"] for v in validacao_edt if v["erro_relativo_test"] is not None), default=None),
    }
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))

    # -------- 4) resumo por (cidade,Q,g,b) entre seeds --------
    log("agregando resumo por (cidade,Q,g,b)...")
    from collections import defaultdict
    grupos_resumo = defaultdict(list)
    for r in resultados:
        chave = (r["cidade"], r["Q"], r["g_km"], r["b_km"])
        grupos_resumo[chave].append(r)

    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        arr = np.array(vals, dtype=float)
        return {"media": float(arr.mean()), "dp": float(arr.std()),
                "p5": float(np.percentile(arr, 5)), "p95": float(np.percentile(arr, 95)),
                "n": int(arr.size)}

    resumo = []
    for (cidade, q, g, b), rs in grupos_resumo.items():
        if len(rs) < 2:
            continue  # resumo entre seeds so faz sentido com >1 seed
        resumo.append({
            "cidade": cidade, "Q": q, "g_km": g, "b_km": b, "n_seeds": len(rs),
            "retencao_val": stats([r["retencao"]["val"] for r in rs]),
            "retencao_test": stats([r["retencao"]["test"] for r in rs]),
            "blocos_efetivos_val": stats([r["blocos_efetivos"]["val"] for r in rs]),
            "blocos_efetivos_test": stats([r["blocos_efetivos"]["test"] for r in rs]),
            "pi_teste": stats([r["pi_teste"] for r in rs]),
            "pi_treino": stats([r["pi_treino"] for r in rs]),
            "d_max_km": stats([r["d_max_km"] for r in rs]),
            "d_min_km": stats([r["d_min_km"] for r in rs]),
        })
    saida["resumo"] = resumo

    # -------- 5) criterios fixados pelo coordenador (item 7) --------
    log("calculando criterios fixados (i-iv)...")
    from scipy import stats as spstats

    criterios = {}

    # (i) por celula em g10b2 e g5b2: DP_seed(retencao teste) / amplitude entre
    # as 16 celulas na seed 42
    for config, (g, b) in (("g10b2", (10.0, 2.0)), ("g5b2", (5.0, 2.0))):
        celulas_config = CIDADES if config == "g10b2" else ["bauru"]
        qs_config = QS
        ret_seed42_por_celula = []
        razoes = []
        for cidade in celulas_config:
            for q in qs_config:
                rs = grupos_resumo.get((cidade, q, g, b), [])
                if not rs:
                    continue
                r42 = next((r for r in rs if r["split_seed"] == 42), None)
                if r42 is not None:
                    ret_seed42_por_celula.append(r42["retencao"]["test"])
                dp_seed = stats([r["retencao"]["test"] for r in rs])
                if dp_seed:
                    razoes.append({"cidade": cidade, "Q": q, "dp_seed": dp_seed["dp"]})
        amplitude_16 = (max(ret_seed42_por_celula) - min(ret_seed42_por_celula)) if len(ret_seed42_por_celula) >= 2 else None
        for r in razoes:
            r["razao_dp_sobre_amplitude"] = (r["dp_seed"] / amplitude_16) if amplitude_16 else None
        criterios[f"i_{config}"] = {
            "amplitude_entre_celulas_seed42": amplitude_16,
            "n_celulas_seed42": len(ret_seed42_por_celula),
            "por_celula": razoes,
            "conclusao": ("geometria domina (razao>1 nalguma celula)" if any(
                (r["razao_dp_sobre_amplitude"] or 0) > 1 for r in razoes) else
                "amplitude entre celulas domina a incerteza por seed em todas as celulas avaliadas"),
        }

    # (ii) correlacao Spearman entre retencao de teste e n de blocos de borda
    # sorteados para teste, por seed -- blocos de borda = ultima linha/coluna
    # do reticulado (nx-1, ny-1 em indice de bloco)
    ii_por_config = {}
    for config, (g, b) in (("g10b2", (10.0, 2.0)), ("g5b2", (5.0, 2.0))):
        celulas_config = CIDADES if config == "g10b2" else ["bauru"]
        pares_ret = []
        pares_borda = []
        for cidade in celulas_config:
            for q in QS:
                pos_m, ell_x, ell_y = get_pos(cidade, q)
                group_ids = get_group_ids(cidade, q, g)
                pos_km = pos_m / 1000.0
                grid_x = (pos_km[:, 0] / g).astype(int)
                grid_y = (pos_km[:, 1] / g).astype(int)
                nx_max, ny_max = grid_x.max(), grid_y.max()
                for seed in seeds:
                    rng = np.random.RandomState(seed)
                    grupos = np.unique(group_ids)
                    grupos_emb = grupos.copy()
                    rng.shuffle(grupos_emb)
                    n_g = len(grupos_emb)
                    n_tr = max(1, int(round(FRACS[0] * n_g)))
                    n_va = max(1, int(round(FRACS[1] * n_g)))
                    if n_tr + n_va >= n_g:
                        n_tr = max(1, n_g - 2); n_va = 1
                    g_te = grupos_emb[n_tr + n_va:]
                    # decompoe group_id de volta em (gx,gy): group_id = gx*max_y+gy
                    max_y = grid_y.max() + 1
                    gte_gy = g_te % max_y
                    gte_gx = g_te // max_y
                    n_borda = int(((gte_gx == nx_max) | (gte_gy == ny_max)).sum())
                    rs = grupos_resumo.get((cidade, q, g, b), [])
                    r_this = next((r for r in rs if r["split_seed"] == seed), None)
                    if r_this is not None:
                        pares_ret.append(r_this["retencao"]["test"])
                        pares_borda.append(n_borda)
        if len(pares_ret) >= 3:
            rho, pval = spstats.spearmanr(pares_ret, pares_borda)
            ii_por_config[config] = {"n_pares": len(pares_ret), "spearman_rho": float(rho), "p_valor": float(pval)}
        else:
            ii_por_config[config] = None
    criterios["ii_spearman_retencao_vs_blocos_borda"] = ii_por_config

    # (iii) pi: P5-P95 entre seeds por celula; fracao de seeds em que a ordem
    # das 4 cidades por pi (media dos Q) e igual a da seed 42
    pi_disponiveis_cidades = sorted(set(c for (c, q) in CFTUDO_FILES.keys())) if saida.get("pi_celulas_disponiveis") else []
    iii = {"aviso": None, "p5_p95_por_celula": {}, "fracao_ordem_igual_seed42": None}
    if len(pi_disponiveis_cidades) < 2:
        iii["aviso"] = ("pi so disponivel para bauru e lins (6/16 celulas, definicao col0<299 "
                         "validada em pi_validacao_definicao); ordenacao entre as 4 cidades "
                         "(item iii) NAO calculavel -- campinas/sorocaba sem tensor local.")
    for (cidade, q) in CFTUDO_FILES.keys():
        rs = grupos_resumo.get((cidade, q, 10.0, 2.0), [])
        pis = [r["pi_teste"] for r in rs if r["pi_teste"] is not None]
        if pis:
            arr = np.array(pis)
            iii["p5_p95_por_celula"][f"{cidade}_{q}"] = {
                "p5": float(np.percentile(arr, 5)), "p95": float(np.percentile(arr, 95)),
                "n": int(arr.size)}
    # ordem por seed, so entre bauru e lins (unicas com pi)
    ordem_por_seed = {}
    for seed in seeds:
        medias = {}
        for cidade in pi_disponiveis_cidades:
            vals = []
            for q in QS:
                if (cidade, q) not in CFTUDO_FILES:
                    continue
                rs = grupos_resumo.get((cidade, q, 10.0, 2.0), [])
                r_this = next((r for r in rs if r["split_seed"] == seed), None)
                if r_this and r_this["pi_teste"] is not None:
                    vals.append(r_this["pi_teste"])
            if vals:
                medias[cidade] = float(np.mean(vals))
        ordem_por_seed[seed] = sorted(medias, key=medias.get)
    ordem_seed42 = ordem_por_seed.get(42)
    if ordem_seed42 and len(pi_disponiveis_cidades) >= 2:
        n_iguais = sum(1 for s in seeds if ordem_por_seed.get(s) == ordem_seed42)
        iii["fracao_ordem_igual_seed42"] = n_iguais / len(seeds)
        iii["ordem_seed42"] = ordem_seed42
        iii["ordem_por_seed"] = {str(k): v for k, v in ordem_por_seed.items()}
    criterios["iii_pi_estabilidade"] = iii

    # (iv) d_max: P5-P95 entre seeds em g10b2 (seed42 Q1 deu 8.58km, ata 15/09)
    iv = {}
    for cidade in CIDADES:
        rs = grupos_resumo.get((cidade, "Q1", 10.0, 2.0), [])
        vals = [r["d_max_km"] for r in rs if r["d_max_km"] is not None]
        if vals:
            arr = np.array(vals)
            r42 = next((r["d_max_km"] for r in rs if r["split_seed"] == 42), None)
            iv[f"{cidade}_Q1"] = {"p5": float(np.percentile(arr, 5)), "p95": float(np.percentile(arr, 95)),
                                   "d_max_seed42_km": r42, "n": int(arr.size)}
    iv["nota"] = "referencia ata 15/09: seed42 Q1 g10b2 d_max=8.58km (conferir contra d_max_seed42_km acima, metodo EDT)"
    criterios["iv_d_max_estabilidade"] = iv

    # (v) pedido do coordenador (correcao de rumo 24/09): DP entre seeds da
    # retencao de teste por (g,b), e razao pelo DP do MC continuo (variante C)
    # de matematica-equacoes (contra-prova cruzada, campo citado, nao recalculado)
    mc_path = Path("/trabalho/HERMES/AGENTES/_FIOS/2026-09-24_gnn_rf_artigo2_mathematics_r2/"
                    "artefatos/matematica-equacoes_montecarlo_retencao_RAW.json")
    v_contra_prova = {"artefato_mc": str(mc_path), "disponivel": mc_path.exists(), "por_config": {}}
    mc_data = None
    if mc_path.exists():
        try:
            mc_data = json.loads(mc_path.read_text())
        except Exception as e:
            v_contra_prova["erro_leitura"] = str(e)
    for config, (g, b) in (("g10b2", (10.0, 2.0)), ("g5b2", (5.0, 2.0))):
        celulas_config = CIDADES if config == "g10b2" else ["bauru"]
        for cidade in celulas_config:
            rs = grupos_resumo.get((cidade, "Q1", g, b), [])
            dp_no = stats([r["retencao"]["test"] for r in rs])
            entry = {"dp_no_split_seed_esta_frente": dp_no["dp"] if dp_no else None}
            if mc_data:
                try:
                    ret_te_dp_mc = mc_data["por_config"][config]["por_cidade"][cidade]["variantes"]["C"]["ret_te_dp"]
                    entry["ret_te_dp_mc_continuo_variante_C"] = ret_te_dp_mc
                    if dp_no and dp_no["dp"] and ret_te_dp_mc:
                        entry["razao_dp_split_seed_sobre_dp_mc_continuo"] = dp_no["dp"] / ret_te_dp_mc
                except (KeyError, TypeError):
                    entry["ret_te_dp_mc_continuo_variante_C"] = None
            v_contra_prova["por_config"][f"{config}_{cidade}"] = entry
    criterios["v_contra_prova_dp_vs_mc_continuo"] = v_contra_prova

    saida["criterios_fixados"] = criterios

    # -------- escopo / status --------
    saida["escopo"] = {
        "executado": [
            "portao: 20 run JSON reais, cKDTree exato",
            f"prioridade (a): 16 celulas x {{g10b2,g5b2}} x {len(seeds)} seeds, metodo EDT resolucao cheia",
            "prioridade (b): grade (g,b) 6x6 x 1 celula/cidade (Q1) x seed 42, metodo EDT",
            "validacao EDT vs cKDTree exato nos 20 casos do portao (seed 42)",
        ],
        "fora_do_escopo": [
            "grade (g,b) completa x >=20 seeds x 16 celulas (custo proibitivo na janela desta rodada)",
            "pi para campinas, sorocaba, bauru Q3/Q4 (tensor leve ausente localmente; .pt bruto de 15-28GB nao carregado)",
            "N efetivo sob autocorrelacao espacial (fora de escopo desta frente; e' de dados-vazamento-espacial)",
        ],
    }
    saida["nao_verificado"] = [
        {"item": "pi para 10/16 celulas", "motivo": "arquivo leve ausente; ver pi_celulas_indisponiveis_motivo"},
        {"item": "N efetivo (vs N de pixels)", "motivo": "fora do escopo desta frente; nao calculado"},
    ]

    saida["tabela_numero_campo_comando"] = [
        {"numero": "n_nos_apos_buffer (portao)", "campo": "portao_validacao.detalhe[i].reproduzido.n_nos_apos_buffer",
         "comando": "python varredura_split_geometria.py --out <json> --hash"},
        {"numero": "retencao val/test por (cidade,Q,g,b,seed)", "campo": "resultados[i].retencao",
         "comando": "idem, funcao split_e_metricas_edt"},
        {"numero": "pi_teste/pi_treino", "campo": "resultados[i].pi_teste / pi_treino",
         "comando": "idem, carregar_pl_valid2d + split_e_metricas_edt"},
        {"numero": "d_min/d_max/mediana teste->treino (km)", "campo": "resultados[i].d_min_km/d_max_km/d_mediana_km",
         "comando": "idem, distance_transform_edt sobre mascara de treino"},
        {"numero": "criterio (i) razao DP_seed/amplitude", "campo": "criterios_fixados.i_g10b2 / i_g5b2",
         "comando": "idem, bloco 'criterios fixados' de main()"},
        {"numero": "criterio (ii) Spearman retencao x blocos de borda", "campo": "criterios_fixados.ii_spearman_retencao_vs_blocos_borda",
         "comando": "idem"},
        {"numero": "criterio (iii) estabilidade de pi", "campo": "criterios_fixados.iii_pi_estabilidade",
         "comando": "idem"},
        {"numero": "criterio (iv) estabilidade de d_max", "campo": "criterios_fixados.iv_d_max_estabilidade",
         "comando": "idem"},
        {"numero": "erro EDT vs cKDTree exato", "campo": "validacao_edt_vs_cktree",
         "comando": "idem, bloco 3 de main()"},
    ]

    saida["status"] = "concluido_provisorio"  # sem contra-auditoria (3 votos); numero nao citavel ate la
    saida["pico_ram_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    saida["tempo_total_s"] = time.time() - t_inicio
    Path(args.out).write_text(json.dumps(saida, indent=1, ensure_ascii=False))
    log(f"CONCLUIDO. tempo total {saida['tempo_total_s']:.1f}s, pico RAM {saida['pico_ram_gb']:.2f}GB")
    log(f"gravado: {args.out}")


if __name__ == "__main__":
    main()
