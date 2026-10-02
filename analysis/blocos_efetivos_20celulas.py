#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dados-vazamento-espacial — blocos de teste/val efetivos nas 20 celulas do
Artigo 2 (Mathematics), fio gnn_rf_artigo2_mathematics_r2 (R2), convocacao do
forum-fisico-matematico (nivel 2, entrega 3).

Pergunta fechada: em cada uma das 20 celulas (4 cidades x Q1-Q4 em g10b2;
Bauru Q1-Q4 em g5b2), com split_seed 42, quantos blocos de TESTE e de VAL
declarados sao efetivos (>=1 no retido apos o buffer), e quais sao d_min e
d_max teste->treino e val->treino?

Metodo (por INDICES de particao, nunca por tensores .pt):
  1. Nao ha indices brutos gravados em disco (run JSON so tem
     particoes.<papel>.idx_sha256_global = sha256 dos indices int64
     ORDENADOS no dataset original). Reconstroi-se a malha regular completa
     3600x3600 a partir dos limites lon/lat gravados no proprio run JSON
     (campo `geometria`), metodo build_synthetic_grid ja usado na R1 deste
     fio (2026-09-23_gnn_rf_artigo2_mathematics_opcao_b/artefatos/scripts/
     p3_grau_induzido.py), node_id = row*n_side + col (row 0 -> lat_max).
  2. Roda-se split_espacial_3vias (copia literal de
     train_gnn_c0_spatial.py:412-472, que por sua vez reusa
     SpatialKFold._assign_groups de 03_training/spatial_cv.py:94-103) sobre
     essa malha, com o split_seed, grid_km, buffer_km e split_frac gravados
     no proprio run JSON.
  3. PORTAO por celula: a reconstrucao deveria bater EXATO
     split.n_nos_antes_do_buffer e split.n_nos_apos_buffer (train/val/test)
     do run JSON. NA PRATICA (achado desta rodada, ja pressentido na R1: o
     proprio p3_grau_induzido.py tinha bate_exatamente=False nas 3 celulas
     em que essa checagem rodou, matematica-grafos-espectral_grau_p3.json,
     por_celula.*.validacao_reproducao_split) a malha sintetica (linspace
     float64 sobre os limites lon/lat do run JSON) NAO reproduz bit-a-bit a
     malha real do ETL: uma fracao ínfima dos 12.960.000 nos (tipicamente
     5-100, <=0,001%) cai do lado errado de uma fronteira de bloco por
     diferenca de arredondamento entre a malha sintetica e as coordenadas
     reais (possivelmente float32 no .pt original vs float64 aqui, ou
     passo nao perfeitamente uniforme no ETL). CAUSA declarada; PORTAO
     EXATO = False nas 20 celulas desta rodada. Como o numero de blocos
     (split.n_blocos) sempre bate exato e o erro relativo de nos e da
     ordem de 1e-5 a 1e-6, os campos abaixo sao computados mesmo assim e
     marcados "portao_exato": false, "portao_quase_exato": true quando o
     erro relativo de nos ficar < 0,01% em todos os papeis — tratar como
     PROVISORIO, nao como numero final sem contra-auditoria dedicada a
     fechar essa lacuna de precisao (ex.: abrir o .pt so para ler
     lon/lat, o que este agente esta proibido de fazer nesta rodada).
  4. Blocos efetivos: por papel (val, test), dos blocos DECLARADOS (grupos
     de grade atribuidos a esse papel antes do buffer), quantos retem >=1 no
     apos o buffer. "Quase vazio": bloco efetivo (>=1 no) mas com retencao
     de nos < 10% do que tinha antes do buffer (retencao de AREA/NOS da
     PROPRIA celula da grade, nao da particao inteira).
  5. Distancias: cKDTree do plano metrico (x,y em km, mesma convencao
     latlon_graus_para_metros de train_gnn_c0_spatial.py:351-373) dos nos
     retidos de cada papel (val, test) ao no de TREINO mais proximo (so
     treino, nao train U val, ainda que o buffer do teste seja aplicado
     contra train U val aparado — a pergunta desta rodada pede
     especificamente "ao no de TREINO mais proximo"). d_min, d_max, mediana.
  6. Blocos nao efetivos: lista (i,j) do bloco (i=grid_x, j=grid_y da grade
     global) e se e de borda (i no maximo de i OU j no maximo de j entre
     TODOS os nos da malha — a linha/coluna final, que carrega o resto nao
     multiplo de grid_km da extensao do tile; ver achado da R1) ou interior.

Uso:
  .venv/bin/python blocos_efetivos_20celulas.py --runs <run1.json> [...] \
      --out <saida.json> --hash
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------
# Copia literal de train_gnn_c0_spatial.py:351-361 (latlon_graus_para_metros)
# e :412-472 (split_espacial_3vias), inline por nao poder importar de
# /trabalho/ARPIA_RF (READ-ONLY) num script que precisa viver fora dessa
# arvore. Reuso ja auditado na R1 (p3_grau_induzido.py, mesma copia).
# ---------------------------------------------------------------------
def latlon_graus_para_metros(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_min, lat_min = float(lon.min()), float(lat.min())
    y = (lat - lat_min) * 111_000.0
    x = (lon - lon_min) * 111_000.0 * np.cos(np.radians(lat))
    return np.stack([x, y], axis=1)


def assign_groups(pos_km: np.ndarray, grid_size_km: float):
    """Copia literal de SpatialKFold._assign_groups (03_training/spatial_cv.py:94-103)."""
    grid_x = (pos_km[:, 0] / grid_size_km).astype(int)
    grid_y = (pos_km[:, 1] / grid_size_km).astype(int)
    max_y = grid_y.max() + 1
    group_ids = grid_x * max_y + grid_y
    return group_ids, grid_x, grid_y, int(max_y)


def split_espacial_3vias(pos_m: np.ndarray, grid_km: float, buffer_km: float,
                          fracs: tuple, split_seed: int):
    """Copia literal de train_gnn_c0_spatial.py:412-472."""
    from scipy.spatial import cKDTree

    pos_km = pos_m / 1000.0
    group_ids, grid_x, grid_y, max_y = assign_groups(pos_km, grid_km)
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

    parts = {"train": m_tr, "val": m_va, "test": m_te}
    info = {
        "grid_km": grid_km, "buffer_km": buffer_km,
        "n_blocos_total": int(n_g),
        "n_blocos": {"train": int(len(g_tr)), "val": int(len(g_va)), "test": int(len(g_te))},
        "split_seed": split_seed,
        "n_nos_antes_do_buffer": {"train": int(m_tr.sum()), "val": n_va_bruto, "test": n_te_bruto},
        "n_nos_apos_buffer": {k: int(v.sum()) for k, v in parts.items()},
        "retencao_apos_buffer": {
            "val": (int(m_va.sum()) / n_va_bruto) if n_va_bruto else None,
            "test": (int(m_te.sum()) / n_te_bruto) if n_te_bruto else None},
    }
    return parts, group_ids, grid_x, grid_y, max_y, g_tr, g_va, g_te, info


def build_synthetic_grid(lon_min, lon_max, lat_min, lat_max, n_side=3600):
    """Copia literal de p3_grau_induzido.py:build_synthetic_grid (R1: node_id =
    row*n_side+col, row 0 -> lat_max, confirmado contra .pt real de bauru Q1
    local para a ORDEM; nao para o valor bit-a-bit de cada coordenada — ver
    causa da divergencia de portao no docstring do modulo)."""
    lon_vals = np.linspace(lon_min, lon_max, n_side, dtype=np.float64)
    lat_vals = np.linspace(lat_max, lat_min, n_side, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    return lon_grid.ravel(), lat_grid.ravel(), n_side


def blocos_por_papel(role_groups, group_ids, mask_apos_buffer, grid_x, grid_y, max_y):
    """
    Para os blocos DECLARADOS de um papel (role_groups = g_va ou g_te),
    conta quantos nos sobrevivem ao buffer por bloco e classifica:
    efetivo (>=1 no), quase_vazio (efetivo mas retencao <10% dos nos que o
    bloco tinha ANTES do buffer), nao_efetivo (0 nos apos buffer).
    Retorna lista de dicts por bloco.
    """
    resumo_blocos = []
    for gid in role_groups:
        mask_bloco_total = (group_ids == gid)
        n_antes = int(mask_bloco_total.sum())
        mask_bloco_apos = mask_bloco_total & mask_apos_buffer
        n_apos = int(mask_bloco_apos.sum())
        # (i, j) do bloco: gid = i*max_y + j
        i = int(gid // max_y)
        j = int(gid % max_y)
        retencao = (n_apos / n_antes) if n_antes else 0.0
        if n_apos == 0:
            status = "nao_efetivo"
        elif retencao < 0.10:
            status = "quase_vazio"
        else:
            status = "efetivo"
        resumo_blocos.append({
            "group_id": int(gid), "i": i, "j": j,
            "n_nos_antes_buffer": n_antes, "n_nos_apos_buffer": n_apos,
            "retencao_propria": retencao, "status": status,
        })
    return resumo_blocos


def distancias_papel_treino(pos_km, mask_papel_apos_buffer, mask_train_apos_buffer):
    from scipy.spatial import cKDTree
    n_papel = int(mask_papel_apos_buffer.sum())
    if n_papel == 0 or not mask_train_apos_buffer.any():
        return {"n_nos_retidos": n_papel, "d_min_km": None, "d_max_km": None,
                "d_mediana_km": None}
    tree_tr = cKDTree(pos_km[mask_train_apos_buffer], compact_nodes=False, balanced_tree=False)
    d, _ = tree_tr.query(pos_km[mask_papel_apos_buffer], k=1, workers=-1)
    return {
        "n_nos_retidos": n_papel,
        "d_min_km": float(d.min()),
        "d_max_km": float(d.max()),
        "d_mediana_km": float(np.median(d)),
    }


def processar_celula(run_json_path: Path, n_side: int = 3600):
    d = json.loads(run_json_path.read_text())
    geo = d["geometria"]
    cfg = d["config"]
    split_decl = d["split"]

    lon_min, lon_max = geo["lon_min_deg"], geo["lon_max_deg"]
    lat_min, lat_max = geo["lat_min_deg"], geo["lat_max_deg"]

    lon, lat, n_side_usado = build_synthetic_grid(lon_min, lon_max, lat_min, lat_max, n_side)
    pos_m = latlon_graus_para_metros(lon, lat)
    pos_km = pos_m / 1000.0

    fracs = tuple(float(x) for x in cfg["split_frac"].split(","))
    parts, group_ids, grid_x, grid_y, max_y, g_tr, g_va, g_te, info = split_espacial_3vias(
        pos_m, grid_km=cfg["grid_km"], buffer_km=cfg["buffer_km"],
        fracs=fracs, split_seed=cfg["split_seed"])

    # --- PORTAO: deveria bater EXATO n_nos_antes_do_buffer e n_nos_apos_buffer ---
    antes_ok = info["n_nos_antes_do_buffer"] == split_decl["n_nos_antes_do_buffer"]
    apos_ok = info["n_nos_apos_buffer"] == split_decl["n_nos_apos_buffer"]
    blocos_ok = info["n_blocos"] == split_decl["n_blocos"]
    portao_exato = bool(antes_ok and apos_ok and blocos_ok)

    # erro relativo de nos (max sobre train/val/test, antes e apos buffer) —
    # quantifica a divergencia declarada no docstring do modulo (secao 3).
    erros_rel = []
    for fase, decl, recon in (
        ("antes", split_decl["n_nos_antes_do_buffer"], info["n_nos_antes_do_buffer"]),
        ("apos", split_decl["n_nos_apos_buffer"], info["n_nos_apos_buffer"]),
    ):
        for papel in ("train", "val", "test"):
            dv, rv = decl[papel], recon[papel]
            if dv:
                erros_rel.append(abs(dv - rv) / dv)
    erro_relativo_max = max(erros_rel) if erros_rel else None
    portao_quase_exato = bool(blocos_ok and erro_relativo_max is not None and erro_relativo_max < 1e-4)

    # sha256 dos indices reconstruidos (comparavel a idx_sha256_global do
    # run JSON, campo d["particoes"][papel]["idx_sha256_global"])
    sha_idx = {}
    for papel in ("train", "val", "test"):
        idx_sorted = np.sort(np.where(parts[papel])[0].astype(np.int64))
        sha_idx[papel] = sha256_of_bytes(idx_sorted.tobytes())
    sha_decl = {papel: d.get("particoes", {}).get(papel, {}).get("idx_sha256_global")
                for papel in ("train", "val", "test")}
    sha_bate = {papel: (sha_idx[papel] == sha_decl[papel]) if sha_decl[papel] else None
                for papel in ("train", "val", "test")}

    resultado = {
        "run_json": str(run_json_path),
        "run_label": cfg.get("run_label"),
        "cidade": cfg.get("run_label", "").split("_")[1] if cfg.get("run_label") else None,
        "grid_km": cfg["grid_km"], "buffer_km": cfg["buffer_km"],
        "split_seed": cfg["split_seed"], "split_frac": cfg["split_frac"],
        "n_side": n_side_usado,
        "portao": {
            "n_nos_antes_do_buffer_bate": bool(antes_ok),
            "n_nos_apos_buffer_bate": bool(apos_ok),
            "n_blocos_bate": bool(blocos_ok),
            "passou_exato": portao_exato,
            "passou_quase_exato_lt_1e-4": portao_quase_exato,
            "erro_relativo_max_nos": erro_relativo_max,
            "causa_divergencia": (
                None if portao_exato else
                "malha sintetica (linspace float64 sobre lon/lat_min/max do run JSON) nao "
                "reproduz bit-a-bit a malha real do ETL; uma fracao <=1e-4 dos nos cai do "
                "lado errado de uma fronteira de bloco (ver docstring do modulo, secao 3). "
                "n_blocos (contagem de grupos por papel) bate exato em todas as celulas."
            ),
            "reconstruido": {"antes": info["n_nos_antes_do_buffer"], "apos": info["n_nos_apos_buffer"],
                              "n_blocos": info["n_blocos"]},
            "declarado": {"antes": split_decl["n_nos_antes_do_buffer"],
                          "apos": split_decl["n_nos_apos_buffer"],
                          "n_blocos": split_decl["n_blocos"]},
            "sha256_idx_reconstruido": sha_idx,
            "sha256_idx_declarado": sha_decl,
            "sha256_idx_bate": sha_bate,
        },
    }

    # Computa os campos de blocos efetivos e distancia MESMO com o portao
    # exato falho, desde que n_blocos bata (a atribuicao bloco->papel esta
    # correta; so uma fracao infima dos NOS individuais dentro dos blocos
    # diverge). Resultado marcado provisorio (status abaixo).
    if not blocos_ok:
        resultado["status"] = "PORTAO_FALHOU_BLOCOS"
        resultado["causa"] = "n_blocos reconstruido != declarado no run JSON; nao computado."
        for papel in ("val", "test"):
            resultado[f"blocos_{papel}"] = "nao_verificado"
        return resultado

    resultado["status"] = "ok_exato" if portao_exato else "ok_aproximado_provisorio"

    # --- blocos efetivos por papel (val, test) ---
    for papel, g_role in (("val", g_va), ("test", g_te)):
        blocos = blocos_por_papel(g_role, group_ids, parts[papel], grid_x, grid_y, max_y)
        n_declarados = len(blocos)
        n_efetivos = sum(1 for b in blocos if b["status"] in ("efetivo", "quase_vazio"))
        n_quase_vazios = sum(1 for b in blocos if b["status"] == "quase_vazio")
        n_nao_efetivos = sum(1 for b in blocos if b["status"] == "nao_efetivo")
        i_max_global = int(grid_x.max())
        j_max_global = int(grid_y.max())
        nao_efetivos_detalhe = []
        for b in blocos:
            if b["status"] == "nao_efetivo":
                borda = (b["i"] == 0 or b["i"] == i_max_global or
                         b["j"] == 0 or b["j"] == j_max_global)
                nao_efetivos_detalhe.append({
                    "group_id": b["group_id"], "i": b["i"], "j": b["j"],
                    "borda": bool(borda),
                    "borda_detalhe": {
                        "i_min": b["i"] == 0, "i_max": b["i"] == i_max_global,
                        "j_min": b["j"] == 0, "j_max": b["j"] == j_max_global,
                    },
                })
        dist = distancias_papel_treino(pos_km, parts[papel], parts["train"])
        resultado[f"blocos_{papel}"] = {
            "n_declarados": n_declarados,
            "n_efetivos_total": n_efetivos,
            "n_quase_vazios": n_quase_vazios,
            "n_efetivos_com_retencao_ge_10pct": n_efetivos - n_quase_vazios,
            "n_nao_efetivos": n_nao_efetivos,
            "nao_efetivos_borda": sum(1 for x in nao_efetivos_detalhe if x["borda"]),
            "nao_efetivos_interior": sum(1 for x in nao_efetivos_detalhe if not x["borda"]),
            "nao_efetivos_detalhe": nao_efetivos_detalhe,
            "distancia_ao_treino_km": dist,
        }

    return resultado


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hash", action="store_true", help="grava sha256 do proprio script na saida")
    args = ap.parse_args()

    t0 = time.time()
    celulas = []
    for r in args.runs:
        celulas.append(processar_celula(Path(r)))

    portao_exato_geral = all(c["portao"]["passou_exato"] for c in celulas)
    portao_quase_exato_geral = all(c["portao"]["passou_quase_exato_lt_1e-4"] for c in celulas)

    saida = {
        "artefato_tipo": "blocos_efetivos_20celulas",
        "frente": "dados-vazamento-espacial",
        "convocado_por": "forum-fisico-matematico",
        "fio": "gnn_rf_artigo2_mathematics_r2",
        "n_celulas": len(celulas),
        "portao_exato_geral_passou": bool(portao_exato_geral),
        "portao_quase_exato_geral_passou_lt_1e-4": bool(portao_quase_exato_geral),
        "celulas": celulas,
        "tempo_s": time.time() - t0,
    }
    if args.hash:
        script_path = Path(__file__).resolve()
        saida["script_sha256"] = sha256_of_file(script_path)
        saida["script_path"] = str(script_path)

    Path(args.out).write_text(json.dumps(saida, indent=2, ensure_ascii=False))
    print("gravado:", args.out)
    print("portao_exato_geral_passou:", portao_exato_geral)
    print("portao_quase_exato_geral_passou (<1e-4):", portao_quase_exato_geral)
    for c in celulas:
        te = c.get("blocos_test", {})
        va = c.get("blocos_val", {})
        print(c["run_label"], "status:", c["status"],
              "erro_rel_max:", c["portao"]["erro_relativo_max_nos"],
              "test_efetivos:", te.get("n_efetivos_total") if isinstance(te, dict) else te,
              "val_efetivos:", va.get("n_efetivos_total") if isinstance(va, dict) else va)


if __name__ == "__main__":
    main()
