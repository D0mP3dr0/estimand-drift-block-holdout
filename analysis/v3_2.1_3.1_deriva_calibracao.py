#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
matematica-estatistica-do-claim -- testes 2.1 (deriva do erro, alegacao A3) e
3.1 (calibracao validos vs mediana contaminada, alegacao A4), roadmap v3,
fio 2026-09-25_gnn_rf_artigo2_v3_fase1_cpu.

Uma passada por celula pelo tensor `_enriched_cftudo.pt` (28 GB, mmap),
retomavel (grava JSON parcial apos cada celula). 16 celulas x N split_seeds
(default: 20, [42,0..18]; ou --seeds explicita/aleatoria -- ver
`--seeds`), split_espacial_3vias (COPIA LITERAL de
train_gnn_c0_spatial.py:412-487 via spatial_cv.SpatialKFold, reusada por
import do proprio r3_alvo_pos_correcao_16celulas.py -- ver
`_metodologia.origem_split`), config g10b2 (grid_km=10, buffer_km=2,
fracs=0.70/0.15/0.15), frequencia 900 MHz (config dos run JSON).

DIVERGENCIA DE SEEDS REGISTRADA (achado, nao corrigido silenciosamente): a
lista default SEEDS_20 = [42,0,1,...,18] usada nas rodadas de 20 sementes
deste script DIFERE em um elemento da lista de sementes da C-1
(`varredura_fracao_valida_r3.py` / `varredura_split_geometria.py`), que usa
[42,1,2,...,19] -- este script tem seed 0 onde a C-1 tem seed 19. O
verificador 3 (raster do chefe em
`_v3_2026-09-25/votos/chefe_teste_seeds_pixel.json`) mostrou que os 20 seeds
da C-1 tem retencao media 2,6% ACIMA de seeds aleatorios -- ou seja, a lista
de 20 nao e uma amostra aleatoria de sementes, e a divergencia de 1 seed
entre este script e a C-1 nao e ravina: os dois conjuntos de 20 podem ter
vies de retencao diferente entre si e frente a sementes aleatorias. Por
isso esta rodada tambem aceita `--seeds aleatorios:N:semente` para gerar um
conjunto de sementes verdadeiramente aleatorio e comparar dp/fracao de
inversao contra os 20 originais (robustez de A3/A4 fora da lista fixa).

Formulas FSPL/Hata rural/COST-231 subarbano: import direto (sem copia, sem
edicao) de dados/scripts_congelados/baselines_v2_por_particao.py (codigo
congelado da IEEE Access), linhas 245-284.

Saida: dois JSON (2.1 e 3.1) com por_celula, resumo, veredito_vs_criterio.
`--sufixo-saida` troca o nome dos artefatos de saida (e do cache parcial)
para NAO sobrescrever a rodada de 20 sementes ao rodar com outra lista.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SCRIPT_PATH = Path(__file__).resolve()
TENSOR_DIR = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")
RUNDIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
              "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/treinos")
MANIFEST = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/"
                 "manifest_mathematics_v3.jsonl")
SPATIAL_CV_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2/03_training")
BASELINES_CONGELADO = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
    "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/dados/"
    "scripts_congelados/baselines_v2_por_particao.py")

OUT_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
OUT_21_DEFAULT = OUT_DIR / "fase2" / "2.1_deriva_erro_baselines_16x20.json"
OUT_31_DEFAULT = OUT_DIR / "fase3" / "3.1_calibracao_validos_vs_mediana.json"
CRIT_21 = OUT_DIR / "criterios" / "criterio_2.1.json"
CRIT_31 = OUT_DIR / "criterios" / "criterio_3.1.json"

CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QUADRANTES = ["Q1", "Q2", "Q3", "Q4"]
PL_TARGET_MAX_VALID = 299.0
FREQ_MHZ = 900.0
GRID_KM = 10.0
BUFFER_KM = 2.0
FRACS = (0.70, 0.15, 0.15)
SEEDS_20 = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
SEEDS_20_C1 = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
assert len(SEEDS_20) == 20

sys.path.append(str(SPATIAL_CV_DIR))
from spatial_cv import SpatialKFold  # noqa: E402

_spec = importlib.util.spec_from_file_location("baselines_congelado", BASELINES_CONGELADO)
_bl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bl)
free_space_path_loss = _bl.free_space_path_loss
okumura_hata_rural = _bl.okumura_hata_rural
cost231_hata = _bl.cost231_hata
BASELINES_SCRIPT_SHA256 = hashlib.sha256(BASELINES_CONGELADO.read_bytes()).hexdigest()

MODELOS = {
    "fspl": lambda d: free_space_path_loss(d, FREQ_MHZ),
    "hata_rural": lambda d: okumura_hata_rural(d, FREQ_MHZ),
    "cost231_sub": lambda d: cost231_hata(d, FREQ_MHZ, environment="suburban"),
}


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def parse_seeds(spec: str) -> tuple:
    """Retorna (lista_de_seeds, descricao). spec vazio -> SEEDS_20 default.
    spec 'aleatorios:N:semente' -> RandomState(semente).randint(0,1_000_000,size=N).
    spec 'a,b,c' -> lista explicita de inteiros."""
    if not spec:
        return list(SEEDS_20), "default_SEEDS_20_[42,0..18]"
    if spec.startswith("aleatorios:"):
        _, n_str, semente_str = spec.split(":")
        n, semente = int(n_str), int(semente_str)
        rng = np.random.RandomState(semente)
        seeds = rng.randint(0, 1_000_000, size=n).tolist()
        return seeds, f"aleatorios_N={n}_semente={semente}"
    seeds = [int(x) for x in spec.split(",")]
    return seeds, f"explicita_{len(seeds)}_seeds"


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_idx(idx: np.ndarray) -> str:
    a = np.ascontiguousarray(np.sort(np.asarray(idx, dtype=np.int64)))
    return hashlib.sha256(a.tobytes()).hexdigest()


def carregar_manifest_sha(basename: str) -> str:
    with open(MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("grupo") == "tensores_cftudo" and Path(r.get("caminho", "")).name == basename:
                return r["sha256"]
    raise RuntimeError(f"basename {basename} nao encontrado no manifest tensores_cftudo")


def latlon_graus_para_metros(pos: torch.Tensor) -> np.ndarray:
    lon = pos[:, 0].double().numpy()
    lat = pos[:, 1].double().numpy()
    lon_min, lat_min = float(lon.min()), float(lat.min())
    y = (lat - lat_min) * 111_000.0
    x = (lon - lon_min) * 111_000.0 * np.cos(np.radians(lat))
    return np.stack([x, y], axis=1)


def split_espacial_3vias(pos_m: np.ndarray, grid_km: float, buffer_km: float,
                          fracs: tuple, split_seed: int) -> dict:
    """COPIA de train_gnn_c0_spatial.py:412-487, reusada por import em
    r3_alvo_pos_correcao_16celulas.py (mesma logica, apenas embutida aqui
    para nao depender de sys.path de outro script de frente)."""
    from scipy.spatial import cKDTree

    pos_km = pos_m / 1000.0
    skf = SpatialKFold(n_splits=3, buffer_km=buffer_km,
                        grid_size_km=grid_km, random_state=split_seed)
    group_ids = skf._assign_groups(pos_km)
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

    return {"train": np.where(m_tr)[0].astype(np.int64),
            "val": np.where(m_va)[0].astype(np.int64),
            "test": np.where(m_te)[0].astype(np.int64)}


def carregar_tensor(path: Path):
    load_kw = dict(map_location="cpu", weights_only=False)
    try:
        rf = torch.load(path, mmap=True, **load_kw)
        return rf, "mmap"
    except Exception as e:
        log(f"  mmap falhou ({e!r}); tentando sem mmap")
        rf = torch.load(path, **load_kw)
        return rf, "sem_mmap"


def mae(pred: np.ndarray, alvo: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - alvo))) if alvo.size else None


def processar_celula(cidade: str, quad: str, verificar_sha: bool, seeds: list) -> dict:
    basename = f"transfer_dataset_{cidade}_v19_{quad}_enriched_cftudo.pt"
    tensor_path = TENSOR_DIR / basename
    run_path = RUNDIR / f"c0c1cf_{cidade}_s42_{quad}_g10b2" / f"run_c0c1cf_{cidade}_s42_{quad}_g10b2.json"

    rec = {"cidade": cidade, "quadrante": quad, "celula": f"{cidade}_{quad}",
           "tensor_path": str(tensor_path), "run_path": str(run_path),
           "timestamp_utc_inicio": datetime.now(timezone.utc).isoformat()}

    if not tensor_path.exists():
        rec["status"] = "erro_tensor_ausente"
        return rec
    if not run_path.exists():
        rec["status"] = "erro_run_json_ausente"
        return rec

    run_rec = json.loads(run_path.read_text(encoding="utf-8"))
    freq_run = run_rec.get("config", {}).get("freq_mhz")
    if freq_run is not None and float(freq_run) != FREQ_MHZ:
        rec["aviso_freq"] = f"config freq_mhz={freq_run} != {FREQ_MHZ} usado no script"

    sha_manifest = carregar_manifest_sha(basename)
    rec["sha256_manifest"] = sha_manifest
    if verificar_sha:
        t0 = time.perf_counter()
        sha_calc = sha256_file(tensor_path)
        rec["sha256_calculado"] = sha_calc
        rec["sha256_ok"] = (sha_calc == sha_manifest)
        rec["tempo_sha_s"] = time.perf_counter() - t0
        if not rec["sha256_ok"]:
            rec["status"] = "erro_sha_tensor_divergente"
            return rec
    else:
        rec["sha256_ok"] = "nao_recalculado_nesta_rodada"

    t1 = time.perf_counter()
    rf, modo_carga = carregar_tensor(tensor_path)
    rec["modo_carga"] = modo_carga
    ty = torch.as_tensor(rf["terrain"].y).float().clone().numpy()
    dist_all = torch.as_tensor(rf["terrain"].dist_nearest_m).float().clone().numpy()
    pos_deg = torch.as_tensor(rf["terrain"].pos).float().clone()
    n_total = int(ty.shape[0])
    del rf
    gc.collect()
    rec["tempo_load_s"] = time.perf_counter() - t1
    rec["n_nodes_total"] = n_total
    log(f"  {cidade}_{quad}: carregado ({modo_carga}) em {rec['tempo_load_s']:.1f}s, n={n_total:,}")

    pos_m = latlon_graus_para_metros(pos_deg)
    rssi = ty[:, 3].astype(np.float64)
    pl = ty[:, 0].astype(np.float64)
    sentinela_all = pl >= PL_TARGET_MAX_VALID

    por_sorteio = []
    for seed in seeds:
        parts = split_espacial_3vias(pos_m, GRID_KM, BUFFER_KM, FRACS, seed)
        tr, te = parts["train"], parts["test"]
        if tr.size == 0 or te.size == 0:
            por_sorteio.append({"split_seed": seed, "status": "erro_particao_vazia",
                                 "n_train": int(tr.size), "n_test": int(te.size)})
            continue

        rssi_tr, rssi_te = rssi[tr], rssi[te]
        pl_tr, pl_te = pl[tr], pl[te]
        dist_tr, dist_te = dist_all[tr], dist_all[te]
        sent_tr, sent_te = sentinela_all[tr], sentinela_all[te]

        pop_te = {"validos": ~sent_te, "sentinela": sent_te,
                  "todos": np.ones_like(sent_te, dtype=bool)}

        # ---- 2.1: constante = mediana(RSSI) do treino inteiro ----
        constante = float(np.median(rssi_tr))
        mae_constante = {p: mae(constante, rssi_te[m]) for p, m in pop_te.items()}
        n_pop_te = {p: int(m.sum()) for p, m in pop_te.items()}

        # ---- calibracao (a): offset no treino inteiro (contaminado) ----
        offset_a = {}
        mae_modelo_a = {}
        for nome, fmodel in MODELOS.items():
            pl_pred_tr = fmodel(dist_tr)
            p_tx_eff = float(np.median(rssi_tr + pl_pred_tr))
            offset_a[nome] = p_tx_eff
            pl_pred_te = fmodel(dist_te)
            rssi_bl_te = p_tx_eff - pl_pred_te
            mae_modelo_a[nome] = {p: mae(rssi_bl_te[m], rssi_te[m]) for p, m in pop_te.items()}

        # ---- calibracao (b): offset so nos validos do treino ----
        offset_b = {}
        mae_modelo_b = {}
        mask_tr_valido = ~sent_tr
        for nome, fmodel in MODELOS.items():
            if mask_tr_valido.sum() > 0:
                pl_pred_tr_v = fmodel(dist_tr[mask_tr_valido])
                p_tx_eff_b = float(np.median(rssi_tr[mask_tr_valido] + pl_pred_tr_v))
            else:
                p_tx_eff_b = None
            offset_b[nome] = p_tx_eff_b
            if p_tx_eff_b is None:
                mae_modelo_b[nome] = {p: None for p in pop_te}
                continue
            pl_pred_te = fmodel(dist_te)
            rssi_bl_te = p_tx_eff_b - pl_pred_te
            mae_modelo_b[nome] = {p: mae(rssi_bl_te[m], rssi_te[m]) for p, m in pop_te.items()}

        # ---- inversao constante > FSPL (por populacao, cada calibracao) ----
        inversao = {
            "a_contaminada": {p: (mae_constante[p] is not None and mae_modelo_a["fspl"][p] is not None
                                    and mae_constante[p] > mae_modelo_a["fspl"][p]) for p in pop_te},
            "b_validos": {p: (mae_constante[p] is not None and mae_modelo_b["fspl"][p] is not None
                               and mae_constante[p] > mae_modelo_b["fspl"][p]) for p in pop_te},
        }

        por_sorteio.append({
            "split_seed": seed, "status": "ok",
            "n_train": int(tr.size), "n_test": int(te.size),
            "n_pop_teste": n_pop_te,
            "constante_treino_mediana_rssi": constante,
            "mae_constante_teste": mae_constante,
            "offset_p_tx_eff_a_contaminado": offset_a,
            "mae_modelo_a_contaminado_teste": mae_modelo_a,
            "offset_p_tx_eff_b_validos": offset_b,
            "mae_modelo_b_validos_teste": mae_modelo_b,
            "inversao_constante_gt_fspl": inversao,
        })

    del ty, rssi, pl, dist_all, pos_deg, pos_m, sentinela_all
    gc.collect()

    rec["status"] = "ok"
    rec["por_sorteio"] = por_sorteio
    rec["timestamp_utc_fim"] = datetime.now(timezone.utc).isoformat()
    return rec


def _dp_ponderado(vals: list, pesos: list) -> float:
    """Desvio-padrao ponderado (pesos de reliabilidade, estimador nao
    viesado): var = sum(w*(x-xbar)^2) / (V1 - V2/V1), V1=sum(w), V2=sum(w^2).
    Peso = n de nos validos no teste do sorteio (sorteios com teste maior
    pesam mais no dp entre sorteios)."""
    w = np.asarray(pesos, dtype=np.float64)
    x = np.asarray(vals, dtype=np.float64)
    if w.sum() <= 0 or len(x) < 2:
        return None
    xbar = float(np.sum(w * x) / np.sum(w))
    V1, V2 = float(np.sum(w)), float(np.sum(w * w))
    denom = V1 - V2 / V1
    if denom <= 0:
        return None
    var = float(np.sum(w * (x - xbar) ** 2) / denom)
    return float(np.sqrt(var)) if var >= 0 else None


def resumir_2_1(celulas: dict) -> dict:
    """dp entre sorteios (por celula, populacao=validos, metrica=constante),
    dp SIMPLES (ddof=1) e dp PONDERADO (peso = n de nos validos no teste do
    sorteio), vs dp entre celulas (media entre sorteios), dp x dp."""
    dp_entre_sorteios = {}
    dp_ponderado_por_celula = {}
    medias_por_celula = {}
    n_celulas_dp_ge_0_1 = 0
    detalhe_por_celula = {}
    for chave, rec in celulas.items():
        if rec.get("status") != "ok":
            continue
        sorteios_ok = [s for s in rec["por_sorteio"] if s.get("status") == "ok"
                       and s["mae_constante_teste"]["validos"] is not None]
        vals_validos = [s["mae_constante_teste"]["validos"] for s in sorteios_ok]
        pesos_validos = [s["n_pop_teste"]["validos"] for s in sorteios_ok]
        vals_todos = [s["mae_constante_teste"]["todos"] for s in rec["por_sorteio"]
                      if s.get("status") == "ok" and s["mae_constante_teste"]["todos"] is not None]
        vals_fspl_validos = [s["mae_modelo_a_contaminado_teste"]["fspl"]["validos"] for s in rec["por_sorteio"]
                             if s.get("status") == "ok" and s["mae_modelo_a_contaminado_teste"]["fspl"]["validos"] is not None]
        if not vals_validos:
            continue
        dp_v = float(np.std(vals_validos, ddof=1)) if len(vals_validos) > 1 else 0.0
        dp_v_ponderado = _dp_ponderado(vals_validos, pesos_validos)
        dp_t = float(np.std(vals_todos, ddof=1)) if len(vals_todos) > 1 else 0.0
        dp_f = float(np.std(vals_fspl_validos, ddof=1)) if len(vals_fspl_validos) > 1 else 0.0
        detalhe_por_celula[chave] = {
            "n_sorteios_ok": len(vals_validos),
            "constante_validos": {"mediana": float(np.median(vals_validos)), "min": float(np.min(vals_validos)),
                                   "max": float(np.max(vals_validos)), "dp_simples": dp_v,
                                   "dp_ponderado_por_n_validos_teste": dp_v_ponderado},
            "constante_todos": {"mediana": float(np.median(vals_todos)), "min": float(np.min(vals_todos)),
                                 "max": float(np.max(vals_todos)), "dp_simples": dp_t},
            "fspl_a_validos": {"mediana": float(np.median(vals_fspl_validos)), "min": float(np.min(vals_fspl_validos)),
                                "max": float(np.max(vals_fspl_validos)), "dp_simples": dp_f},
        }
        dp_entre_sorteios[chave] = dp_v
        if dp_v_ponderado is not None:
            dp_ponderado_por_celula[chave] = dp_v_ponderado
        medias_por_celula[chave] = float(np.mean(vals_validos))
        if dp_v >= 0.1:
            n_celulas_dp_ge_0_1 += 1

    dp_entre_celulas = float(np.std(list(medias_por_celula.values()), ddof=1)) if len(medias_por_celula) > 1 else None
    return {
        "detalhe_por_celula": detalhe_por_celula,
        "n_celulas_com_dp_entre_sorteios_ge_0_1_dB": n_celulas_dp_ge_0_1,
        "n_celulas_total_ok": len(dp_entre_sorteios),
        "dp_entre_celulas_das_medias_validos_dB": dp_entre_celulas,
        "dp_medio_entre_sorteios_validos_dB_simples": float(np.mean(list(dp_entre_sorteios.values()))) if dp_entre_sorteios else None,
        "dp_medio_entre_sorteios_validos_dB_ponderado": float(np.mean(list(dp_ponderado_por_celula.values()))) if dp_ponderado_por_celula else None,
        "definicao_dp_ponderado": "peso = n de nos validos no teste retido do sorteio; estimador de reliabilidade (nao viesado): var=sum(w*(x-xbar)^2)/(sum(w)-sum(w^2)/sum(w))",
    }


def resumir_3_1(celulas: dict) -> dict:
    total_combos = 0
    venceu_a = {"validos": 0, "todos": 0}
    venceu_b = {"validos": 0, "todos": 0}
    offsets_a_por_modelo = {"fspl": [], "hata_rural": [], "cost231_sub": []}
    offsets_b_por_modelo = {"fspl": [], "hata_rural": [], "cost231_sub": []}
    lins_q1_seed42 = None
    for chave, rec in celulas.items():
        if rec.get("status") != "ok":
            continue
        for s in rec["por_sorteio"]:
            if s.get("status") != "ok":
                continue
            total_combos += 1
            for pop in ("validos", "todos"):
                if s["inversao_constante_gt_fspl"]["a_contaminada"].get(pop):
                    venceu_a[pop] += 1
                if s["inversao_constante_gt_fspl"]["b_validos"].get(pop):
                    venceu_b[pop] += 1
            for m in offsets_a_por_modelo:
                if s["offset_p_tx_eff_a_contaminado"].get(m) is not None:
                    offsets_a_por_modelo[m].append(s["offset_p_tx_eff_a_contaminado"][m])
                if s["offset_p_tx_eff_b_validos"].get(m) is not None:
                    offsets_b_por_modelo[m].append(s["offset_p_tx_eff_b_validos"][m])
            if chave == "lins_Q1" and s["split_seed"] == 42:
                lins_q1_seed42 = {
                    "offset_a_contaminado": s["offset_p_tx_eff_a_contaminado"],
                    "offset_b_validos": s["offset_p_tx_eff_b_validos"],
                }
    frac_a_validos = venceu_a["validos"] / total_combos if total_combos else None
    frac_b_validos = venceu_b["validos"] / total_combos if total_combos else None
    return {
        "n_combinacoes_celula_x_sorteio": total_combos,
        "constante_vence_fspl_calibracao_a_contaminada": venceu_a,
        "constante_vence_fspl_calibracao_b_validos": venceu_b,
        "fracao_inversao_validos_calibracao_a": frac_a_validos,
        "fracao_inversao_validos_calibracao_b": frac_b_validos,
        "offset_p_tx_eff_a_contaminado_estatisticas": {
            m: {"mediana": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}
            for m, v in offsets_a_por_modelo.items() if v},
        "offset_p_tx_eff_b_validos_estatisticas": {
            m: {"mediana": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}
            for m, v in offsets_b_por_modelo.items() if v},
        "lins_Q1_seed42_offsets": lins_q1_seed42,
        "texto_atual_citado_lins_q1_s42_dbm": {"fspl": 20.4, "hata_rural": 55.2, "cost231_sub": 83.3},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celula", default="", help="ex.: lins_Q1 (fumaca de uma celula)")
    ap.add_argument("--out-parcial", default="", help="default depende de --sufixo-saida")
    ap.add_argument("--sem-sha", action="store_true", help="pula recalculo de sha256 (ja verificado em R2)")
    ap.add_argument("--seeds", default="", help="'' = SEEDS_20 default; 'aleatorios:N:semente'; ou 'a,b,c' explicita")
    ap.add_argument("--sufixo-saida", default="", help="ex.: '_16x60rnd' -- troca nome dos JSON de saida e do cache parcial")
    args = ap.parse_args()

    seeds, seeds_desc = parse_seeds(args.seeds)
    sufixo = args.sufixo_saida
    if sufixo:
        out_21 = OUT_DIR / "fase2" / f"2.1_deriva_erro_baselines{sufixo}.json"
        out_31 = OUT_DIR / "fase3" / f"3.1_calibracao_validos_vs_mediana{sufixo}.json"
        parcial_default = OUT_DIR / "fase2" / f"_v3_2.1_3.1_parcial{sufixo}.json"
    else:
        out_21 = OUT_21_DEFAULT
        out_31 = OUT_31_DEFAULT
        parcial_default = OUT_DIR / "fase2" / "_v3_2.1_3.1_parcial.json"
    parcial_path = Path(args.out_parcial) if args.out_parcial else parcial_default

    parcial_path.parent.mkdir(parents=True, exist_ok=True)
    if parcial_path.exists():
        estado = json.loads(parcial_path.read_text(encoding="utf-8"))
    else:
        estado = {"script_caminho": str(SCRIPT_PATH),
                  "script_sha256": hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
                  "baselines_congelado_path": str(BASELINES_CONGELADO),
                  "baselines_congelado_sha256": BASELINES_SCRIPT_SHA256,
                  "seeds_usados": seeds, "seeds_descricao": seeds_desc,
                  "timestamp_utc_inicio": datetime.now(timezone.utc).isoformat(),
                  "celulas": {}}

    if args.celula:
        cidade, quad = args.celula.split("_")
        pares = [(cidade, quad)]
    else:
        pares = [(c, q) for c in CIDADES for q in QUADRANTES]

    for cidade, quad in pares:
        chave = f"{cidade}_{quad}"
        ja = estado["celulas"].get(chave)
        if ja is not None and ja.get("status") == "ok":
            log(f"{chave}: ja concluida, pulando")
            continue
        log(f"=== {chave} (seeds: {seeds_desc}, n={len(seeds)}) ===")
        t0 = time.perf_counter()
        try:
            rec = processar_celula(cidade, quad, verificar_sha=not args.sem_sha, seeds=seeds)
        except Exception as e:
            rec = {"cidade": cidade, "quadrante": quad, "status": "erro_excecao",
                   "erro": repr(e), "traceback": traceback.format_exc()}
        rec["tempo_total_s"] = time.perf_counter() - t0
        estado["celulas"][chave] = rec
        estado["timestamp_utc_ultima_gravacao"] = datetime.now(timezone.utc).isoformat()
        parcial_path.write_text(json.dumps(estado, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"{chave}: status={rec.get('status')} em {rec['tempo_total_s']:.1f}s (gravado em {parcial_path})")

    if args.celula:
        log("fumaca concluida, sem consolidacao final")
        return

    # ---- consolidacao: 2.1 e 3.1 ----
    celulas = estado["celulas"]
    n_ok = sum(1 for r in celulas.values() if r.get("status") == "ok")

    nota_divergencia_seeds = {
        "seeds_usados_nesta_rodada": seeds, "descricao": seeds_desc,
        "seeds_20_default_deste_script": SEEDS_20,
        "seeds_20_da_c1": SEEDS_20_C1,
        "divergencia": "quando seeds_descricao == 'default_SEEDS_20_[42,0..18]', este script usa seed 0 onde a C-1 "
                       "(varredura_fracao_valida_r3.py) usa seed 19 -- 19 de 20 sementes coincidem. O verificador 3 "
                       "mostrou que os 20 seeds da C-1 tem retencao media 2,6% acima de seeds aleatorios (raster do "
                       "chefe, _v3_2026-09-25/votos/chefe_teste_seeds_pixel.json); a lista de 20 NAO e amostra "
                       "aleatoria de sementes, e por isso a comparacao contra seeds aleatorios (--seeds aleatorios:N:S) "
                       "e o teste de robustez desta rodada.",
    }

    criterio_21 = json.loads(CRIT_21.read_text(encoding="utf-8"))
    resumo_21 = resumir_2_1(celulas)
    veredito_21 = ("A3_sustentada_deriva_nao_pequena"
                   if resumo_21["n_celulas_com_dp_entre_sorteios_ge_0_1_dB"] >= 8
                   else "A3_deriva_pequena_titulo_reconsiderar")
    saida_21 = {
        "id": "2.1", "pergunta": criterio_21["pergunta"], "alegacao": criterio_21["alegacao"],
        "criterio": criterio_21,
        "metodo": f"16 celulas x {len(seeds)} split_seeds ({seeds_desc}), split_espacial_3vias g10b2 (grid=10km,buffer=2km,fracs 0.70/0.15/0.15), "
                  "constante=mediana(RSSI treino inteiro do sorteio), FSPL calibrado por offset em p_tx_eff=mediana(rssi_tr+FSPL(dist_tr,900MHz)) no treino inteiro; "
                  "MAE no teste retido por populacao (validos: PL<299, sentinela: PL>=299, todos). dp simples (ddof=1) e dp ponderado "
                  "(peso=n_validos do teste por sorteio) reportados.",
        "nota_divergencia_seeds": nota_divergencia_seeds,
        "insumos": [{"caminho": str(TENSOR_DIR / f"transfer_dataset_{c}_v19_{q}_enriched_cftudo.pt"),
                     "sha256": celulas.get(f"{c}_{q}", {}).get("sha256_manifest")}
                    for c in CIDADES for q in QUADRANTES],
        "formulas_congeladas": {"caminho": str(BASELINES_CONGELADO), "sha256": BASELINES_SCRIPT_SHA256,
                                 "linhas": "245-284 (free_space_path_loss, okumura_hata_rural, cost231_hata)"},
        "por_celula": resumo_21["detalhe_por_celula"],
        "resumo": {k: v for k, v in resumo_21.items() if k != "detalhe_por_celula"},
        "n_celulas_ok": n_ok, "n_celulas_total": 16,
        "veredito_vs_criterio": veredito_21,
        "nao_verificado": [],
        "comando_rodado": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_21.parent.mkdir(parents=True, exist_ok=True)
    out_21.write_text(json.dumps(saida_21, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"gravado {out_21}")

    criterio_31 = json.loads(CRIT_31.read_text(encoding="utf-8"))
    resumo_31 = resumir_3_1(celulas)
    frac_b = resumo_31["fracao_inversao_validos_calibracao_b"]
    frac_a = resumo_31["fracao_inversao_validos_calibracao_a"]
    if frac_b is not None and frac_b >= 0.5:
        veredito_31 = "A4_sustentada_com_baseline_calibrada_corretamente"
    elif frac_a is not None and frac_a >= 0.5:
        veredito_31 = "A4_reformular_inversao_induzida_pela_calibracao_contaminada"
    else:
        veredito_31 = "A4_nao_sustentada_em_nenhuma_calibracao"
    saida_31 = {
        "id": "3.1", "pergunta": criterio_31["pergunta"], "alegacao": criterio_31["alegacao"],
        "criterio": criterio_31,
        "metodo": f"16 celulas x {len(seeds)} split_seeds ({seeds_desc}, mesmos splits de 2.1); FSPL/Hata_rural/COST231_suburbano calibrados por offset "
                  "p_tx_eff=mediana(rssi_tr+PL_modelo(dist_tr,900MHz)) em (a) treino inteiro contaminado e (b) so validos do treino; "
                  "MAE por populacao no teste retido; contagem constante>FSPL por calibracao.",
        "nota_divergencia_seeds": nota_divergencia_seeds,
        "insumos": [{"caminho": str(TENSOR_DIR / f"transfer_dataset_{c}_v19_{q}_enriched_cftudo.pt"),
                     "sha256": celulas.get(f"{c}_{q}", {}).get("sha256_manifest")}
                    for c in CIDADES for q in QUADRANTES],
        "formulas_congeladas": {"caminho": str(BASELINES_CONGELADO), "sha256": BASELINES_SCRIPT_SHA256,
                                 "linhas": "245-284 (free_space_path_loss, okumura_hata_rural, cost231_hata)"},
        "por_celula": {chave: [{"split_seed": s["split_seed"],
                                  "mae_constante_validos": s["mae_constante_teste"]["validos"],
                                  "mae_fspl_a_validos": s["mae_modelo_a_contaminado_teste"]["fspl"]["validos"],
                                  "mae_fspl_b_validos": s["mae_modelo_b_validos_teste"]["fspl"]["validos"],
                                  "inversao_a": s["inversao_constante_gt_fspl"]["a_contaminada"]["validos"],
                                  "inversao_b": s["inversao_constante_gt_fspl"]["b_validos"]["validos"]}
                                 for s in rec["por_sorteio"] if s.get("status") == "ok"]
                        for chave, rec in celulas.items() if rec.get("status") == "ok"},
        "resumo": resumo_31,
        "n_celulas_ok": n_ok, "n_celulas_total": 16,
        "veredito_vs_criterio": veredito_31,
        "nao_verificado": [],
        "comando_rodado": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_31.parent.mkdir(parents=True, exist_ok=True)
    out_31.write_text(json.dumps(saida_31, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"gravado {out_31}")
    log("concluido")


if __name__ == "__main__":
    main()
