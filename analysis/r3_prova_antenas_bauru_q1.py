"""D-e (prova por script, CPU): dist_nearest_m de bauru_Q1 pos-correcao vem
de que conjunto de antenas?

Hipotese do adendo v3 (forum-eng-dados, artefatos/forum-eng-dados_parecer_R3.md):
- POS-correcao (transfer_dataset_bauru_v19_Q1_enriched_cftudo.pt): dist_nearest_m
  eh a Haversine minima contra `antenna.pos` do PROPRIO grafo (fallback de
  enrich_rf_targets.py:475-478, porque graph_data_v3 nao tem o CSV de antenas).
- PRE-correcao (transfer_dataset_bauru_v19_Q1_enriched.pt, sem _cftudo): dist_nearest_m
  foi medida contra o CSV `antenas_interior_sp_final.csv` filtrado a +/-200 MHz de
  1800 MHz e ao bbox do terreno com 0.2 graus de folga (enrich_rf_targets.py:94-134,
  157-199, 458-472), 175 torres unicas (alvo_stats_bauru_s42_Q1_v2.json).

Metodo: 1 leitura de cada tensor sob flock (map_location='cpu', mmap=True,
CUDA_VISIBLE_DEVICES=''), amostra de 1e6 nos terrain (seed 20260925, sem
reposicao) por tensor. Recalcula Haversine minima contra (a) antenna.pos do
proprio grafo e (b) as antenas do CSV filtradas, compara com dist_nearest_m
gravado no tensor (erro maximo e mediano em metros). Reporta tambem contagem
de antenas em antenna.pos e no CSV filtrado, e se pos esta em graus (lat/lon,
EPSG:4326) ou metros (checado por |x|<=180 e |y|<=90 -- mesmo teste que
enrich_rf_targets.py:_dist_nearest_chunked).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_PATH = Path(__file__).resolve()
R = 6371000.0  # raio da Terra em metros (mesma constante de enrich_rf_targets.py)

PRE_PT = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2/graph_data/transfer_dataset_bauru_v19_Q1_enriched.pt")
POST_PT = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3/transfer_dataset_bauru_v19_Q1_enriched_cftudo.pt")
POST_SHA_MANIFEST = "33e8d57368920241fc289f8c939de35ece7178c316d2bf5488c3cdf98ff5fb5c"
CSV_PATH = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2/data_raw/antenas_interior_sp_final.csv")
ALVO_STATS_PRE = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
    "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/dados/alvo/"
    "alvo_stats_bauru_s42_Q1_v2.json"
)
OUT_JSON = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/artefatos/"
    "dados-proveniencia_r3_prova_antenas_bauru_q1.json"
)

FREQ_MHZ_TARGET = 1800.0
FREQ_TOL_MHZ = 200.0
BBOX_BUFFER_DEG = 0.2
SEED = 20260925
N_SAMPLE = 1_000_000


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_self() -> str:
    return sha256_file(SCRIPT_PATH)


def haversine_min_m(terrain_pos: np.ndarray, antenna_pos: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """COPIA da definicao de enrich_rf_targets.py:_haversine_cdist_m /
    _haversine_nearest_chunked (mesma formula, mesmo R), em numpy puro."""
    lon_a = np.radians(terrain_pos[:, 0].astype(np.float64))
    lat_a = np.radians(terrain_pos[:, 1].astype(np.float64))
    lon_b = np.radians(antenna_pos[:, 0].astype(np.float64))
    lat_b = np.radians(antenna_pos[:, 1].astype(np.float64))
    n = terrain_pos.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dlon = lon_b[None, :] - lon_a[start:end, None]
        dlat = lat_b[None, :] - lat_a[start:end, None]
        a = (np.sin(dlat / 2) ** 2
             + np.cos(lat_a[start:end, None]) * np.cos(lat_b[None, :])
             * np.sin(dlon / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        out[start:end] = (R * c).min(axis=1)
    return out


def load_antenna_pos_from_csv(csv_path: Path, terrain_pos: np.ndarray,
                               target_freq_mhz: float, tol_mhz: float,
                               buffer_deg: float):
    """COPIA da logica de enrich_rf_targets.py:_load_antenna_pos_from_csv
    (filtro por bbox do terreno + freq +/- tol), reimplementada em pandas puro
    para nao depender de torch aqui."""
    df = pd.read_csv(csv_path, usecols=["Latitude", "Longitude", "FreqTxMHz"], low_memory=False)
    df = df.dropna(subset=["Latitude", "Longitude"])

    lon_min = float(terrain_pos[:, 0].min()) - buffer_deg
    lon_max = float(terrain_pos[:, 0].max()) + buffer_deg
    lat_min = float(terrain_pos[:, 1].min()) - buffer_deg
    lat_max = float(terrain_pos[:, 1].max()) + buffer_deg

    mask_bbox = (
        (df["Longitude"] >= lon_min) & (df["Longitude"] <= lon_max)
        & (df["Latitude"] >= lat_min) & (df["Latitude"] <= lat_max)
    )
    df_bbox = df[mask_bbox]

    freq_mask = (df_bbox["FreqTxMHz"] - target_freq_mhz).abs() <= tol_mhz
    df_freq = df_bbox[freq_mask]

    pos = df_freq[["Longitude", "Latitude"]].drop_duplicates().to_numpy(dtype=np.float64)
    return pos, len(df_bbox), len(df_freq)


def carregar_tensor(path: Path):
    load_kw = dict(map_location="cpu", weights_only=False)
    try:
        rf = torch.load(path, mmap=True, **load_kw)
        return rf, "mmap"
    except Exception as e:
        log(f"  mmap falhou ({e!r}); tentando sem mmap")
        rf = torch.load(path, **load_kw)
        return rf, "sem_mmap"


def is_geographic(pos: np.ndarray) -> bool:
    """Mesmo teste de enrich_rf_targets.py:_dist_nearest_chunked."""
    return float(np.abs(pos[:, 0]).max()) <= 181.0


def processar_tensor(path: Path, label: str, sha_esperado: str | None,
                      csv_pos: np.ndarray, csv_freq_count: int, csv_bbox_count: int,
                      resultado: dict) -> None:
    t0 = time.time()
    sha = sha256_file(path)
    resultado[f"{label}_sha256"] = sha
    resultado[f"{label}_tamanho_bytes"] = path.stat().st_size
    if sha_esperado is not None:
        resultado[f"{label}_sha256_confere_manifest"] = (sha == sha_esperado)
    log(f"{label}: sha256={sha[:16]}... ({time.time()-t0:.1f}s)")

    rf, modo = carregar_tensor(path)
    resultado[f"{label}_modo_load"] = modo

    terrain_pos_t = rf["terrain"].pos
    if not isinstance(terrain_pos_t, torch.Tensor):
        terrain_pos_t = torch.as_tensor(terrain_pos_t)
    terrain_pos = terrain_pos_t.float().numpy()
    n_terrain = terrain_pos.shape[0]

    dist_stored_t = rf["terrain"].dist_nearest_m
    if not isinstance(dist_stored_t, torch.Tensor):
        dist_stored_t = torch.as_tensor(dist_stored_t)
    dist_stored = dist_stored_t.float().numpy()

    antenna_pos = None
    n_antenna_pos = None
    if hasattr(rf["antenna"], "pos"):
        ap = rf["antenna"].pos
        if not isinstance(ap, torch.Tensor):
            ap = torch.as_tensor(ap)
        antenna_pos = ap.float().numpy()
        n_antenna_pos = antenna_pos.shape[0]

    resultado[f"{label}_n_terrain"] = int(n_terrain)
    resultado[f"{label}_n_antenna_pos"] = int(n_antenna_pos) if n_antenna_pos is not None else None
    resultado[f"{label}_terrain_pos_geografico"] = bool(is_geographic(terrain_pos))
    if antenna_pos is not None:
        resultado[f"{label}_antenna_pos_geografico"] = bool(is_geographic(antenna_pos))

    rng = np.random.RandomState(SEED)
    n_sample = min(N_SAMPLE, n_terrain)
    idx = rng.choice(n_terrain, size=n_sample, replace=False)
    idx.sort()
    resultado[f"{label}_n_amostra"] = int(n_sample)
    resultado[f"{label}_seed_amostra"] = SEED

    pos_sample = terrain_pos[idx]
    dist_stored_sample = dist_stored[idx].astype(np.float64)

    if antenna_pos is not None:
        t1 = time.time()
        dist_hip_antennapos = haversine_min_m(pos_sample, antenna_pos)
        erro_a = np.abs(dist_hip_antennapos - dist_stored_sample)
        resultado[f"{label}_hipA_antenna_pos"] = {
            "erro_max_m": float(erro_a.max()),
            "erro_mediano_m": float(np.median(erro_a)),
            "n_antenas": int(antenna_pos.shape[0]),
            "tempo_s": round(time.time() - t1, 1),
        }
        log(f"{label} hipA (antenna.pos, n={antenna_pos.shape[0]}): "
            f"erro_max={erro_a.max():.3f}m erro_mediano={np.median(erro_a):.3f}m")

    if csv_pos is not None and len(csv_pos) > 0:
        t1 = time.time()
        dist_hip_csv = haversine_min_m(pos_sample, csv_pos)
        erro_b = np.abs(dist_hip_csv - dist_stored_sample)
        resultado[f"{label}_hipB_csv_filtrado"] = {
            "erro_max_m": float(erro_b.max()),
            "erro_mediano_m": float(np.median(erro_b)),
            "n_antenas": int(len(csv_pos)),
            "n_apos_bbox": int(csv_bbox_count),
            "n_apos_freq": int(csv_freq_count),
            "tempo_s": round(time.time() - t1, 1),
        }
        log(f"{label} hipB (CSV filtrado, n={len(csv_pos)}): "
            f"erro_max={erro_b.max():.3f}m erro_mediano={np.median(erro_b):.3f}m")

    del rf
    import gc
    gc.collect()
    log(f"{label}: processado em {time.time()-t0:.1f}s total")


def main() -> int:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    resultado: dict = {
        "script": str(SCRIPT_PATH),
        "script_sha256": None,  # preenchido no fim (self-hash pos-escrita nao circular; hash do arquivo tal como esta no disco ao rodar)
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "seed_amostra": SEED,
        "n_amostra_alvo": N_SAMPLE,
        "csv_path": str(CSV_PATH),
        "freq_mhz_alvo": FREQ_MHZ_TARGET,
        "freq_tol_mhz": FREQ_TOL_MHZ,
        "bbox_buffer_deg": BBOX_BUFFER_DEG,
    }
    resultado["script_sha256"] = sha256_self()

    resultado["csv_sha256"] = sha256_file(CSV_PATH)

    # alvo_stats pre-correcao: registra fonte_antenas e freq usada (conferencia cruzada)
    if ALVO_STATS_PRE.exists():
        alvo_pre = json.loads(ALVO_STATS_PRE.read_text(encoding="utf-8"))
        resultado["alvo_stats_pre_fonte_antenas"] = alvo_pre.get("dist_nearest", {}).get("fonte_antenas")
        resultado["alvo_stats_pre_freq_mhz"] = alvo_pre.get("reuso", {}).get("freq_mhz")
        resultado["alvo_stats_pre_dist_identica_v1"] = alvo_pre.get(
            "comparacao_v2_vs_v1_publicado", {}
        ).get("dist_nearest_identica_bit_a_bit")

    log("Carregando terrain.pos do tensor POS-correcao para montar bbox do CSV...")
    rf_post_probe, _ = carregar_tensor(POST_PT)
    terrain_pos_probe = rf_post_probe["terrain"].pos
    if not isinstance(terrain_pos_probe, torch.Tensor):
        terrain_pos_probe = torch.as_tensor(terrain_pos_probe)
    terrain_pos_probe = terrain_pos_probe.float().numpy()
    del rf_post_probe
    import gc
    gc.collect()

    csv_pos, n_bbox, n_freq = load_antenna_pos_from_csv(
        CSV_PATH, terrain_pos_probe, FREQ_MHZ_TARGET, FREQ_TOL_MHZ, BBOX_BUFFER_DEG
    )
    resultado["csv_n_apos_bbox"] = int(n_bbox)
    resultado["csv_n_apos_freq_unico"] = int(len(csv_pos))
    log(f"CSV: {n_bbox} antenas no bbox, {n_freq} apos filtro de freq, {len(csv_pos)} posicoes unicas")

    lock_path = Path(
        "/trabalho/HERMES/AGENTES/_FIOS/2026-09-24_gnn_rf_artigo2_mathematics_r2/tensor28gb.lock"
    )
    import fcntl

    log(f"Adquirindo flock em {lock_path} para processar POST ({POST_PT.name})...")
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            processar_tensor(POST_PT, "post", POST_SHA_MANIFEST, csv_pos, n_freq, n_bbox, resultado)
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)

    if PRE_PT.exists():
        log(f"Adquirindo flock em {lock_path} para processar PRE ({PRE_PT.name})...")
        with open(lock_path, "w") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            try:
                processar_tensor(PRE_PT, "pre", None, csv_pos, n_freq, n_bbox, resultado)
            finally:
                fcntl.flock(lockf, fcntl.LOCK_UN)
    else:
        resultado["pre_tensor_ausente"] = True

    # Veredito da hipotese
    def melhor_hip(prefixo: str):
        a = resultado.get(f"{prefixo}_hipA_antenna_pos", {}).get("erro_mediano_m")
        b = resultado.get(f"{prefixo}_hipB_csv_filtrado", {}).get("erro_mediano_m")
        if a is None and b is None:
            return None
        if b is None or (a is not None and a <= b):
            return "hipA_antenna_pos"
        return "hipB_csv_filtrado"

    resultado["veredito_post"] = melhor_hip("post")
    resultado["veredito_pre"] = melhor_hip("pre") if PRE_PT.exists() else None

    hipotese_confirmada = (
        resultado["veredito_post"] == "hipA_antenna_pos"
        and (not PRE_PT.exists() or resultado["veredito_pre"] == "hipB_csv_filtrado")
    )
    resultado["hipotese_confirmada"] = bool(hipotese_confirmada)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Gravado {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
