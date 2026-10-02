"""
TerrainGNN V19 — Graph Generator por Cidade com Sentinel-2 REAL
================================================================
Ambiente : F:\\S33_dslm\\s33_amb_virtual\\.venv\\Scripts\\python.exe
GPU      : RTX 5070 Ti  |  sm_120  |  15.9 GB VRAM
Stack    : PyTorch 2.10+cu128  |  PyG 2.8.0  |  CUDA 12.8

Diferenças V18.3 → V19
-----------------------
  - Cada bioma (V18.3) → cada cidade (V19): Lins, Campinas, Bauru, Sorocaba
  - Sentinel sintético/tiles degradados → APENAS TIFs reais 10m por cidade
  - Bbox 1°×1° (V18.5) → 2°×2° centrado na cidade (mesmo padrão V18.3)
  - DEM recortado do mosaico consolidado ao bbox 2°×2°
  - Derivativos: CuPy (V18.3) mantido → PyTorch CUDA nativo (V18.5/V19, mesmo GPU)
  - KNN LiDAR: cKDTree (V18.3) → gpu_knn torch.cdist tiled (V18.5/V19)
  - Output: {city}_v19_gpu.pt  (7200×7200 DEM = 12.96M nós, padrão V18.3)
  - Normalization dict V18.4 completo: elev_mean, elev_std, slope_max, canopy_max,
    is_normalized, version

Sentinel real disponível por cidade (10m):
  sentinel2/lins/interior_sp_s2_lins_10m_real.tif        — 25% da área 2°×2°
  sentinel2/campinas/interior_sp_s2_campinas_10m_real.tif — 25%
  sentinel2/bauru/interior_sp_s2_bauru_10m_real.tif       — 25%
  sentinel2/sorocaba/interior_sp_s2_sorocaba_10m_real.tif — 25%
  → TIFs adjacentes são carregados se cobrirem a área da cidade
  → Áreas sem TIF real → zeros (honestos)

Uso:
  python generate_graph_v19.py --city lins
  python generate_graph_v19.py --city campinas
  python generate_graph_v19.py --city bauru
  python generate_graph_v19.py --city sorocaba
  python generate_graph_v19.py --all
  python generate_graph_v19.py --city lins --dry-run
"""

import gc
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from scipy.spatial import cKDTree
from rasterio.transform import Affine
from rasterio.warp import reproject
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import grid as pyg_grid
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# DEVICE
# ============================================================================
if not torch.cuda.is_available():
    logger.error("CUDA não disponível. Use F:\\S33_dslm\\s33_amb_virtual\\.venv")
    sys.exit(1)

DEVICE = torch.device('cuda')
_props = torch.cuda.get_device_properties(0)
logger.info(f"GPU : {_props.name}  |  {_props.total_memory / 1024**3:.1f} GB  |  "
            f"sm_{_props.major}{_props.minor}")

# ============================================================================
# CAMINHOS
# ============================================================================
BASE_DIR      = Path(r"F:\arpia_topo_refinado\TOPO_RF\GNN_RF_V2\data_raw\satelites_raw")
DEM_MOSAIC    = BASE_DIR / "dem" / "interior_sp_dem_consolidated.tif"
DEM_TILES_DIR = BASE_DIR / "dem" / "tiles"
LIDAR_FILE    = BASE_DIR / "lidar" / "lidar_interior_sp.csv"
S2_BASE_DIR   = BASE_DIR / "sentinel2"
OUTPUT_DIR    = Path(r"F:\arpia_topo_refinado\TOPO_RF\GNN_RF_V2\graph_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# TIFs reais 10m disponíveis.
# Prioridade: TIF 2°×2° (download_sentinel2_v19_2deg.py → cobertura 100%)
# Fallback  : TIF 1°×1° original (cobertura ~40% da área 2°×2°)
# Regra     : se o TIF 2deg existir, ele é listado SOZINHO para a cidade (cobre 100%).
#             Se não existir, usa o TIF 1deg (e vizinhos que sobreponham).

def _build_real_s2_list() -> list:
    cities = ["lins", "campinas", "bauru", "sorocaba"]
    result = []
    for city in cities:
        tif_2deg = S2_BASE_DIR / city / f"interior_sp_s2_{city}_10m_real_2deg.tif"
        tif_1deg = S2_BASE_DIR / city / f"interior_sp_s2_{city}_10m_real.tif"
        if tif_2deg.exists():
            result.append(tif_2deg)
        elif tif_1deg.exists():
            result.append(tif_1deg)
    return result

REAL_S2_TIFS = _build_real_s2_list()

# ============================================================================
# CONFIGURAÇÃO POR CIDADE
# Centro urbano (IBGE) → bbox 2°×2° calculado automaticamente
# ============================================================================
DEM_RESOLUTION      = 30.0
SENTINEL_RESOLUTION = 10.0
LIDAR_K_NEIGHBORS   = 8
LIDAR_K_LL          = 32
KNN_TILE_SIZE       = 50_000
SLOPE_MAX           = 90.0
CANOPY_MAX          = 50.0

CITY_CENTERS = {
    "lins":     {"lon": -49.7451, "lat": -21.6796, "label": "Lins SP"},
    "campinas": {"lon": -47.0608, "lat": -22.9056, "label": "Campinas SP"},
    "bauru":    {"lon": -49.0636, "lat": -22.3246, "label": "Bauru SP"},
    "sorocaba": {"lon": -47.4526, "lat": -23.5015, "label": "Sorocaba SP"},
}


# ============================================================================
# BBOX 2°×2° CENTRADO — com ajuste automático aos limites do DEM
# ============================================================================
def compute_city_bbox(city: str) -> tuple:
    """
    Retorna (lon_min, lat_min, lon_max, lat_max) do bbox 2°×2° centrado no
    centro urbano da cidade. Se o bbox extrapolar os limites do DEM consolidado,
    desloca minimamente para caber dentro (sem reduzir o tamanho de 2°×2°).
    """
    c = CITY_CENTERS[city]
    lon_min = c["lon"] - 1.0
    lat_min = c["lat"] - 1.0
    lon_max = c["lon"] + 1.0
    lat_max = c["lat"] + 1.0

    with rasterio.open(DEM_MOSAIC) as src:
        d = src.bounds

    # Deslocar para caber dentro do DEM sem reduzir tamanho
    if lon_max > d.right:
        shift = lon_max - d.right
        lon_min -= shift
        lon_max -= shift
    if lon_min < d.left:
        shift = d.left - lon_min
        lon_min += shift
        lon_max += shift
    if lat_max > d.top:
        shift = lat_max - d.top
        lat_min -= shift
        lat_max -= shift
    if lat_min < d.bottom:
        shift = d.bottom - lat_min
        lat_min += shift
        lat_max += shift

    return (lon_min, lat_min, lon_max, lat_max)


# ============================================================================
# POSIÇÕES AFINS NA GPU
# ============================================================================
def affine_positions_gpu(transform: Affine, H: int, W: int) -> torch.Tensor:
    """Retorna (H*W, 2) [lon, lat] calculado 100% na GPU."""
    rows = torch.arange(H, device=DEVICE, dtype=torch.float32).repeat_interleave(W)
    cols = torch.arange(W, device=DEVICE, dtype=torch.float32).repeat(H)
    lons = transform.c + cols * transform.a
    lats = transform.f + rows * transform.e
    return torch.stack([lons, lats], dim=1)


# ============================================================================
# MOSAICO DEM (cria se não existir)
# ============================================================================
def ensure_dem_mosaic() -> Path:
    if DEM_MOSAIC.exists():
        return DEM_MOSAIC
    logger.info(f"Criando mosaico DEM: {DEM_TILES_DIR} → {DEM_MOSAIC}")
    tif_files = sorted(DEM_TILES_DIR.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"Nenhum .tif em {DEM_TILES_DIR}")
    srcs = [rasterio.open(fp) for fp in tif_files]
    min_x = min(s.bounds.left for s in srcs)
    max_x = max(s.bounds.right for s in srcs)
    min_y = min(s.bounds.bottom for s in srcs)
    max_y = max(s.bounds.top for s in srcs)
    for s in srcs:
        s.close()
    with rasterio.open(tif_files[0]) as ref:
        res_x, res_y = ref.res
        meta = ref.meta.copy()
    out_w = int(round((max_x - min_x) / res_x))
    out_h = int(round((max_y - min_y) / res_y))
    out_trans = Affine.translation(min_x, max_y) * Affine.scale(res_x, -res_y)
    meta.update(driver="GTiff", height=out_h, width=out_w, transform=out_trans,
                compress="lzw", tiled=True, blockxsize=256, blockysize=256, bigtiff="YES")
    with rasterio.open(DEM_MOSAIC, "w", **meta) as dst:
        for fp in tqdm(tif_files, desc="Mesclando tiles DEM"):
            with rasterio.open(fp) as src:
                col_off = int(round((src.bounds.left - min_x) / res_x))
                row_off = int(round((max_y - src.bounds.top) / res_y))
                win = rasterio.windows.Window(col_off, row_off, src.width, src.height)
                dst.write(src.read(1), 1, window=win)
    logger.info(f"Mosaico DEM criado: {DEM_MOSAIC}")
    return DEM_MOSAIC


# ============================================================================
# DERIVATIVOS TOPOGRÁFICOS — 100% PyTorch CUDA (mantém precisão V18.5)
# ============================================================================
@torch.no_grad()
def calculate_derivatives_gpu(dem_np: np.ndarray, cell_size: float) -> dict:
    logger.info("Derivativos topográficos (PyTorch CUDA)...")
    px = cell_size * 111_320.0 if cell_size < 1.0 else cell_size
    dem = torch.tensor(dem_np, device=DEVICE, dtype=torch.float32)

    g_y, g_x = torch.gradient(dem, spacing=px)

    slope_rad = torch.atan(torch.hypot(g_x, g_y))
    slope_deg = torch.nan_to_num(torch.rad2deg(slope_rad))

    aspect_rad = torch.remainder(torch.atan2(-g_x, g_y) + 2 * torch.pi, 2 * torch.pi)
    aspect_cos = torch.cos(aspect_rad)
    aspect_sin = torch.sin(aspect_rad)

    g_xx = torch.gradient(g_x, spacing=px)[1]
    g_yy = torch.gradient(g_y, spacing=px)[0]
    curvature = g_xx + g_yy

    kernel    = torch.ones(1, 1, 3, 3, device=DEVICE, dtype=torch.float32) / 9.0
    dem_4d    = dem.unsqueeze(0).unsqueeze(0)
    mean_elev = F.conv2d(dem_4d, kernel, padding=1).squeeze()
    tpi       = dem - mean_elev

    local_max = F.max_pool2d(dem_4d, kernel_size=3, stride=1, padding=1).squeeze()
    local_min = -F.max_pool2d(-dem_4d, kernel_size=3, stride=1, padding=1).squeeze()
    tri       = local_max - local_min

    mean_sq   = F.conv2d(dem_4d ** 2, kernel, padding=1).squeeze()
    roughness = torch.sqrt(torch.clamp(mean_sq - mean_elev ** 2, min=0.0))

    flow_accum = torch.log1p(torch.abs(curvature) * 100.0 + 1.0)

    logger.info("   Derivativos OK")
    return dict(
        slope=slope_deg, aspect_cos=aspect_cos, aspect_sin=aspect_sin,
        curvature=curvature, tpi=tpi, tri=tri, roughness=roughness,
        flow_accum=flow_accum, dem_tensor=dem,
    )


# ============================================================================
# KNN GPU — tiled para controlar VRAM
# ============================================================================
@torch.no_grad()
def gpu_knn(query: torch.Tensor, ref: torch.Tensor, k: int) -> tuple:
    N = query.shape[0]
    M = ref.shape[0]
    free_mem  = torch.cuda.mem_get_info()[0]
    tile_size = max(1, int(free_mem * 0.35 / (M * 4)))
    tile_size = min(tile_size, KNN_TILE_SIZE)
    logger.info(f"   gpu_knn N={N:,} M={M:,} k={k} tile={tile_size:,} "
                f"VRAM={free_mem/1024**3:.1f}GB")
    all_dist = torch.full((N, k), float('inf'), device=DEVICE, dtype=torch.float32)
    all_idx  = torch.zeros((N, k), device=DEVICE, dtype=torch.long)
    for start in range(0, N, tile_size):
        end   = min(start + tile_size, N)
        dists = torch.cdist(query[start:end], ref)
        td, ti = torch.topk(dists, k, largest=False, dim=1)
        all_dist[start:end] = td
        all_idx[start:end]  = ti
        del dists
        torch.cuda.empty_cache()
    return all_dist, all_idx


# ============================================================================
# CARREGAR SENTINEL-2 REAL — acumula TIFs reais que cobrem a área
# Exatamente como V18.3: reproject→accumulate→average→normalize
# Apenas TIFs reais 10m (REAL_S2_TIFS). Zeros onde não há dado.
# ============================================================================
def load_sentinel2_real(
    actual_bounds: tuple,
    dem_crs,
    H_sent: int,
    W_sent: int,
    sent_transform: Affine,
) -> tuple:
    lon_min, lat_min, lon_max, lat_max = actual_bounds

    b02 = np.zeros((H_sent, W_sent), dtype=np.float32)
    b03 = np.zeros((H_sent, W_sent), dtype=np.float32)
    b04 = np.zeros((H_sent, W_sent), dtype=np.float32)
    b08 = np.zeros((H_sent, W_sent), dtype=np.float32)
    cnt = np.zeros((H_sent, W_sent), dtype=np.float32)
    tmp = np.zeros((H_sent, W_sent), dtype=np.float32)

    loaded = 0
    for tif_path in REAL_S2_TIFS:
        if not tif_path.exists():
            logger.warning(f"   S2 TIF não encontrado: {tif_path}")
            continue
        try:
            with rasterio.open(tif_path) as s2:
                tb = s2.bounds
                # Verificar sobreposição real com o bbox da área
                if (tb.right <= lon_min or tb.left >= lon_max or
                        tb.top <= lat_min or tb.bottom >= lat_max):
                    continue
                if s2.count < 4:
                    logger.warning(f"   {tif_path.name}: apenas {s2.count} bandas, esperado 4")
                    continue

                logger.info(f"   Carregando: {tif_path.name} "
                            f"lon[{tb.left:.3f},{tb.right:.3f}] "
                            f"lat[{tb.bottom:.3f},{tb.top:.3f}]")

                for bi, buf in enumerate([b02, b03, b04, b08]):
                    tmp.fill(0.0)
                    reproject(
                        source=rasterio.band(s2, bi + 1),
                        destination=tmp,
                        src_transform=s2.transform,
                        src_crs=s2.crs,
                        dst_transform=sent_transform,
                        dst_crs=dem_crs,
                        resampling=Resampling.nearest,
                        num_threads=8,
                    )
                    buf += tmp
                    if bi == 0:
                        cnt += (tmp > 0).astype(np.float32)

                loaded += 1
        except Exception as exc:
            logger.warning(f"   Erro ao carregar {tif_path.name}: {exc}")

    del tmp
    logger.info(f"   TIFs S2 reais carregados: {loaded}")

    # Média onde há sobreposição de TIFs; zeros onde não há dado
    cnt_safe = np.maximum(cnt, 1.0)
    b02 /= cnt_safe
    b03 /= cnt_safe
    b04 /= cnt_safe
    b08 /= cnt_safe

    # Reflectância S2-L2A: [0, 10000] → [0, 1]
    b02 = np.clip(b02 / 10_000.0, 0.0, 1.0)
    b03 = np.clip(b03 / 10_000.0, 0.0, 1.0)
    b04 = np.clip(b04 / 10_000.0, 0.0, 1.0)
    b08 = np.clip(b08 / 10_000.0, 0.0, 1.0)

    real_pct = 100.0 * (cnt > 0).sum() / (H_sent * W_sent)
    logger.info(f"   Cobertura S2 real: {real_pct:.1f}% da área 2°×2°")
    logger.info(f"   B04 mean={b04[cnt>0].mean():.4f}  "
                f"B08 mean={b08[cnt>0].mean():.4f}" if (cnt > 0).any() else "   (sem pixels reais)")

    return b02, b03, b04, b08, cnt


# ============================================================================
# GERAÇÃO DO GRAFO HETEROGÊNEO V19
# Padrão V18.3: 7200×7200 DEM, 21600×21600 Sentinel, normalization V18.4
# ============================================================================
def generate_graph_v19(city: str, dry_run: bool = False) -> Path:
    logger.info(f"\n{'='*70}")
    logger.info(f"V19 HeteroGraph Sentinel REAL: {city.upper()}")
    logger.info(f"{'='*70}")
    torch.cuda.reset_peak_memory_stats()

    ensure_dem_mosaic()

    # ── 0. BBOX 2°×2° CENTRADO ───────────────────────────────────────────────
    lon_min, lat_min, lon_max, lat_max = compute_city_bbox(city)
    logger.info(f"   Cidade : {CITY_CENTERS[city]['label']}")
    logger.info(f"   Centro : lon={CITY_CENTERS[city]['lon']:.4f}  "
                f"lat={CITY_CENTERS[city]['lat']:.4f}")
    logger.info(f"   Bbox 2°×2°: lon[{lon_min:.4f}, {lon_max:.4f}]  "
                f"lat[{lat_min:.4f}, {lat_max:.4f}]")

    # ── 1. CARGA DEM ─────────────────────────────────────────────────────────
    logger.info("DEM: recortando bbox 2°×2° do mosaico...")
    with rasterio.open(DEM_MOSAIC) as src:
        row_s, col_s = src.index(lon_min, lat_max)   # canto NW
        row_e, col_e = src.index(lon_max, lat_min)   # canto SE
        row_s = max(0, row_s); col_s = max(0, col_s)
        row_e = min(src.height, row_e); col_e = min(src.width, col_e)

        if dry_run:
            row_e = min(row_s + 1024, row_e)
            col_e = min(col_s + 1024, col_e)
            logger.info("   DRY RUN: janela 1024×1024")

        win = rasterio.windows.Window(col_s, row_s, col_e - col_s, row_e - row_s)
        dem_np        = src.read(1, window=win).astype(np.float32)
        dem_transform = src.window_transform(win)
        l, b, r, t    = src.window_bounds(win)
        dem_crs       = src.crs
        cell_size     = abs(dem_transform[0])

    H_dem, W_dem  = dem_np.shape
    N_dem         = H_dem * W_dem
    actual_bounds = (l, b, r, t)
    logger.info(f"   DEM: {H_dem}×{W_dem} = {N_dem:,} nós | "
                f"cell={cell_size:.6f}° ({cell_size*111320:.1f}m)")
    logger.info(f"   Bounds: lon[{l:.4f}, {r:.4f}]  lat[{b:.4f}, {t:.4f}]")

    nodata = -9999
    valid  = dem_np != nodata
    if not valid.all():
        dem_np = np.where(valid, dem_np, float(np.nanmean(dem_np[valid])))

    # ── 2. ESTATÍSTICAS DE NORMALIZAÇÃO (padrão V18.4) ───────────────────────
    elev_mean = float(np.nanmean(dem_np))
    elev_std  = float(np.nanstd(dem_np))
    if elev_std < 1e-6:
        elev_std = 1.0
        logger.warning("   elev_std próximo de zero, usando 1.0")
    logger.info(f"   elev_mean={elev_mean:.2f}m  elev_std={elev_std:.2f}m")

    # ── 3. DERIVATIVOS TOPOGRÁFICOS (GPU) ────────────────────────────────────
    derivs = calculate_derivatives_gpu(dem_np, cell_size)

    # ── 4. SENTINEL-2 REAL (CPU load → numpy) ────────────────────────────────
    H_sent, W_sent = H_dem * 3, W_dem * 3
    sent_transform = dem_transform * dem_transform.scale(1 / 3, 1 / 3)
    N_sent         = H_sent * W_sent
    logger.info(f"Sentinel: grade {H_sent}×{W_sent} = {N_sent:,} nós")

    b02_np, b03_np, b04_np, b08_np, cnt_np = load_sentinel2_real(
        actual_bounds, dem_crs, H_sent, W_sent, sent_transform)

    # Índices espectrais — numpy (arrays grandes, GPU depois)
    ndvi_np = np.clip((b08_np - b04_np) / (b08_np + b04_np + 1e-6), -1.0, 1.0)
    ndwi_np = np.clip((b03_np - b08_np) / (b03_np + b08_np + 1e-6), -1.0, 1.0)
    bsi_n   = (b04_np + b02_np) - (b08_np + b03_np)
    bsi_np  = np.clip(bsi_n / ((b04_np + b02_np) + (b08_np + b03_np) + 1e-6), -1.0, 1.0)
    shd_np  = np.clip(1.0 - np.sqrt((b02_np**2 + b03_np**2 + b04_np**2) / 3.0), 0.0, 1.0)

    logger.info(f"   NDVI mean={ndvi_np[cnt_np>0].mean():.3f}  "
                f"NDWI mean={ndwi_np[cnt_np>0].mean():.3f}" if (cnt_np > 0).any() else "")

    # ── 5. LIDAR CSV ──────────────────────────────────────────────────────────
    logger.info("LiDAR...")
    if LIDAR_FILE.exists():
        df   = pd.read_csv(LIDAR_FILE)
        mask = ((df['lat'] >= actual_bounds[1]) & (df['lat'] <= actual_bounds[3]) &
                (df['lon'] >= actual_bounds[0]) & (df['lon'] <= actual_bounds[2]))
        df      = df[mask].reset_index(drop=True)
        N_lidar = len(df)
        logger.info(f"   LiDAR: {N_lidar:,} pontos na região")
    else:
        df      = pd.DataFrame()
        N_lidar = 0
        logger.warning(f"   LiDAR não encontrado: {LIDAR_FILE}")

    # ── 6. CONSTRUÇÃO HETERODATA ──────────────────────────────────────────────
    logger.info("Construindo HeteroData...")
    data = HeteroData()

    # ── 6.1 NÓS DEM (17 features, normalização V18.4) ───────────────────────
    dem_elev_flat = dem_np.flatten()
    dem_elev_norm = (dem_elev_flat - elev_mean) / elev_std

    dem_x = torch.zeros((N_dem, 17), dtype=torch.float32)
    dem_x[:, 0] = torch.tensor(dem_elev_norm)
    dem_x[:, 1] = torch.tensor(derivs['slope'].flatten().cpu().numpy() / SLOPE_MAX)
    dem_x[:, 2] = torch.tensor(derivs['aspect_cos'].flatten().cpu().numpy())
    dem_x[:, 3] = torch.tensor(derivs['aspect_sin'].flatten().cpu().numpy())
    dem_x[:, 4] = torch.tensor(derivs['curvature'].flatten().cpu().numpy())
    dem_x[:, 5] = torch.tensor(derivs['tpi'].flatten().cpu().numpy() / elev_std)
    dem_x[:, 6] = torch.tensor(derivs['tri'].flatten().cpu().numpy() / elev_std)
    dem_x[:, 7] = torch.tensor(derivs['roughness'].flatten().cpu().numpy() / elev_std)

    # Agregar Sentinel → DEM (media 3×3)
    logger.info("   Agregando Sentinel → resolução DEM...")
    for col, arr in enumerate([b02_np, b03_np, b04_np, b08_np, ndvi_np, ndwi_np], start=8):
        aggregated = arr.reshape(H_dem, 3, W_dem, 3).mean(axis=(1, 3)).flatten()
        dem_x[:, col] = torch.tensor(aggregated)
    # cols 14–16: has_lidar, z_lidar, confidence → preenchidos em §6.3

    dem_y = torch.zeros((N_dem, 14), dtype=torch.float32)
    dem_y[:, 0] = dem_x[:, 0]   # elev normalizada
    dem_y[:, 1] = dem_x[:, 1]   # slope
    dem_y[:, 2] = dem_x[:, 2]   # aspect_cos
    dem_y[:, 3] = dem_x[:, 3]   # aspect_sin
    dem_y[:, 4] = dem_x[:, 4]   # curvature
    dem_y[:, 5] = dem_x[:, 5]   # tpi
    dem_y[:, 6] = dem_x[:, 6]   # tri
    dem_y[:, 7] = dem_x[:, 7]   # roughness
    dem_y[:, 8] = torch.tensor(derivs['flow_accum'].flatten().cpu().numpy())
    # col 9: canopy → preenchido após LiDAR
    for col, arr in enumerate([ndvi_np, ndwi_np, bsi_np, shd_np], start=10):
        dem_y[:, col] = torch.tensor(arr.reshape(H_dem, 3, W_dem, 3).mean(axis=(1, 3)).flatten())

    # Posições DEM na GPU → salvar CPU
    dem_pos_gpu = affine_positions_gpu(dem_transform, H_dem, W_dem)

    dem_r   = torch.arange(H_dem, device=DEVICE).repeat_interleave(W_dem)
    dem_c   = torch.arange(W_dem, device=DEVICE).repeat(H_dem)
    dem_pix = dem_r * W_dem + dem_c

    data['dem'].x          = dem_x
    data['dem'].y          = dem_y
    data['dem'].pos        = dem_pos_gpu.cpu()
    data['dem'].pixel_idx  = dem_pix.cpu()
    data['dem'].node_id    = torch.arange(N_dem, dtype=torch.long)
    data['dem'].grid_shape = (H_dem, W_dem)

    # ── 6.2 NÓS SENTINEL (8 features) ────────────────────────────────────────
    logger.info("   Nós Sentinel...")
    # Processar em chunks para manter RAM controlada (padrão V18.3 chunk_rows=1024)
    data['sentinel'].x          = torch.zeros((N_sent, 8), dtype=torch.float32)
    data['sentinel'].pos        = torch.zeros((N_sent, 2), dtype=torch.float32)
    data['sentinel'].parent_idx = torch.zeros((N_sent,),  dtype=torch.long)
    data['sentinel'].pixel_idx  = torch.zeros((N_sent,),  dtype=torch.long)
    data['sentinel'].node_id    = torch.arange(N_sent, dtype=torch.long)
    data['sentinel'].grid_shape = torch.tensor([H_sent, W_sent], dtype=torch.long)

    CHUNK = 1024
    cur   = 0
    for r_start in tqdm(range(0, H_sent, CHUNK), desc="Sent nodes"):
        r_end      = min(r_start + CHUNK, H_sent)
        rows_chunk = r_end - r_start
        n_chunk    = rows_chunk * W_sent

        r_idx = np.repeat(np.arange(r_start, r_end, dtype=np.int64), W_sent)
        c_idx = np.tile(np.arange(W_sent, dtype=np.int64), rows_chunk)

        cx = torch.zeros((n_chunk, 8), dtype=torch.float32)
        cx[:, 0] = torch.tensor(b02_np[r_start:r_end].flatten())
        cx[:, 1] = torch.tensor(b03_np[r_start:r_end].flatten())
        cx[:, 2] = torch.tensor(b04_np[r_start:r_end].flatten())
        cx[:, 3] = torch.tensor(b08_np[r_start:r_end].flatten())
        cx[:, 4] = torch.tensor(ndvi_np[r_start:r_end].flatten())
        cx[:, 5] = torch.tensor(ndwi_np[r_start:r_end].flatten())
        cx[:, 6] = torch.tensor((r_idx % 3).astype(np.float32) / 2.0)
        cx[:, 7] = torch.tensor((c_idx % 3).astype(np.float32) / 2.0)

        # Posições via transform afim
        lons_c = sent_transform.c + c_idx * sent_transform.a
        lats_c = sent_transform.f + r_idx * sent_transform.e
        cp_pos = torch.tensor(np.column_stack([lons_c, lats_c]), dtype=torch.float32)

        dem_r_c = torch.tensor(r_idx // 3, dtype=torch.long)
        dem_c_c = torch.tensor(c_idx // 3, dtype=torch.long)
        parent  = dem_r_c * W_dem + dem_c_c
        pix     = torch.tensor(r_idx * W_sent + c_idx, dtype=torch.long)

        end = cur + n_chunk
        data['sentinel'].x[cur:end]          = cx
        data['sentinel'].pos[cur:end]        = cp_pos
        data['sentinel'].parent_idx[cur:end] = parent
        data['sentinel'].pixel_idx[cur:end]  = pix
        cur = end
        del cx, cp_pos, r_idx, c_idx

    # Liberar arrays Sentinel numpy
    del b02_np, b03_np, b04_np, b08_np, ndvi_np, ndwi_np, bsi_np, shd_np, cnt_np
    gc.collect()

    # ── 6.3 NÓS LIDAR (3 features, normalização V18.4) ───────────────────────
    if N_lidar > 0:
        lidar_z_raw  = df['elevation'].values.astype(np.float32)
        lidar_z_norm = (lidar_z_raw - elev_mean) / elev_std

        lidar_x = torch.zeros((N_lidar, 3), dtype=torch.float32)
        lidar_x[:, 0] = torch.tensor(lidar_z_norm)
        lidar_x[:, 1] = 1.0   # confidence
        lidar_x[:, 2] = 0.0   # source_id

        lidar_pos = torch.tensor(
            np.column_stack([df['lon'].values, df['lat'].values]), dtype=torch.float32)

        lidar_rows, lidar_cols = rasterio.transform.rowcol(
            dem_transform, df['lon'].values, df['lat'].values)
        lidar_rows   = np.clip(np.array(lidar_rows), 0, H_dem - 1)
        lidar_cols   = np.clip(np.array(lidar_cols), 0, W_dem - 1)
        lidar_parent = torch.tensor(lidar_rows * W_dem + lidar_cols, dtype=torch.long)

        data['lidar'].x          = lidar_x
        data['lidar'].pos        = lidar_pos
        data['lidar'].parent_idx = lidar_parent
        data['lidar'].node_id    = torch.arange(N_lidar, dtype=torch.long)

        # Agregação LiDAR → DEM (has_lidar, z_lidar, canopy)
        agg = (pd.DataFrame({'p': lidar_parent.numpy(), 'zn': lidar_z_norm, 'zr': lidar_z_raw})
               .groupby('p').mean())
        pi   = torch.tensor(agg.index.values, dtype=torch.long)
        data['dem'].x[pi, 14] = 1.0
        data['dem'].x[pi, 15] = torch.tensor(agg['zn'].values, dtype=torch.float32)

        canopy_raw  = dem_elev_flat[agg.index.values] - agg['zr'].values
        canopy_norm = np.clip(canopy_raw / CANOPY_MAX, 0.0, 1.0)
        data['dem'].y[pi, 9] = torch.tensor(canopy_norm, dtype=torch.float32)
        logger.info(f"   Canopy: mean={canopy_raw.mean():.1f}m  max={canopy_raw.max():.1f}m")

    else:
        data['lidar'].x          = torch.zeros((0, 3), dtype=torch.float32)
        data['lidar'].pos        = torch.zeros((0, 2), dtype=torch.float32)
        data['lidar'].parent_idx = torch.zeros((0,),  dtype=torch.long)
        data['lidar'].node_id    = torch.zeros((0,),  dtype=torch.long)

    # ── 7. ARESTAS ────────────────────────────────────────────────────────────
    logger.info("Criando arestas...")

    # 7.1 DEM ↔ DEM 8-vizinhança
    edge_idx_dem, _ = pyg_grid(H_dem, W_dem)
    src_e, dst_e    = edge_idx_dem
    d_row = ((dst_e // W_dem) - (src_e // W_dem)).float()
    d_col = ((dst_e %  W_dem) - (src_e %  W_dem)).float()
    dist_dem = torch.sqrt(d_row**2 + d_col**2) * DEM_RESOLUTION
    dz_dem   = data['dem'].x[dst_e, 0] - data['dem'].x[src_e, 0]

    data['dem', 'adjacent_to', 'dem'].edge_index = edge_idx_dem
    data['dem', 'adjacent_to', 'dem'].edge_attr  = torch.stack([dist_dem, dz_dem], dim=1)
    logger.info(f"   DEM↔DEM       : {edge_idx_dem.shape[1]:,}")

    # 7.2 Sentinel → DEM
    s_idx       = torch.arange(N_sent, dtype=torch.long)
    local_row_s = data['sentinel'].x[:, 6] * 2.0
    local_col_s = data['sentinel'].x[:, 7] * 2.0
    offset_x    = (local_col_s - 1) * SENTINEL_RESOLUTION
    offset_y    = (local_row_s - 1) * SENTINEL_RESOLUTION

    data['sentinel', 'belongs_to', 'dem'].edge_index = torch.stack(
        [s_idx, data['sentinel'].parent_idx])
    data['sentinel', 'belongs_to', 'dem'].edge_attr  = torch.stack(
        [offset_x, offset_y, local_row_s * 3 + local_col_s], dim=1)
    logger.info(f"   Sentinel→DEM  : {N_sent:,}")

    # 7.3 LiDAR → DEM (belongs_to + near_to via KNN GPU)
    if N_lidar > 0:
        l_idx    = torch.arange(N_lidar, dtype=torch.long)
        par_pos  = data['dem'].pos[lidar_parent]
        dist_ctr = torch.norm(lidar_pos - par_pos, dim=1)

        data['lidar', 'belongs_to', 'dem'].edge_index = torch.stack([l_idx, lidar_parent])
        data['lidar', 'belongs_to', 'dem'].edge_attr  = torch.stack(
            [dist_ctr, lidar_x[:, 1]], dim=1)

        logger.info("   KNN LiDAR→DEM (cKDTree CPU)...")
        dem_pos_np   = data['dem'].pos.numpy().astype(np.float64)
        lidar_pos_np = lidar_pos.numpy().astype(np.float64)
        tree_dem     = cKDTree(dem_pos_np)
        knn_d_np, knn_i_np = tree_dem.query(
            lidar_pos_np, k=LIDAR_K_NEIGHBORS, workers=os.cpu_count()
        )
        knn_d = torch.from_numpy(knn_d_np).float()
        knn_i = torch.from_numpy(knn_i_np).long()

        src_near  = torch.arange(N_lidar).repeat_interleave(LIDAR_K_NEIGHBORS)
        dst_near  = knn_i.flatten()
        dist_near = knn_d.flatten()
        w_near    = 1.0 / (dist_near**2 + 1e-6)

        data['lidar', 'near_to', 'dem'].edge_index = torch.stack([src_near, dst_near])
        data['lidar', 'near_to', 'dem'].edge_attr  = torch.stack([dist_near, w_near], dim=1)
        logger.info(f"   LiDAR→DEM knn : {src_near.shape[0]:,}")

        # 7.4 LiDAR ↔ LiDAR (raio adaptativo, cKDTree CPU)
        if N_lidar > 1:
            logger.info("   LiDAR↔LiDAR (cKDTree CPU, raio adaptativo)...")
            k_ll     = min(LIDAR_K_LL, N_lidar - 1)
            tree_ll  = cKDTree(lidar_pos_np)
            ll_d_np, ll_i_np = tree_ll.query(
                lidar_pos_np, k=k_ll + 1, workers=os.cpu_count()
            )
            ll_d = torch.from_numpy(ll_d_np[:, 1:]).float()
            ll_i = torch.from_numpy(ll_i_np[:, 1:]).long()

            slope_flat  = derivs['slope'].flatten().cpu()
            lidar_slope = slope_flat[lidar_parent]

            src_ll  = torch.arange(N_lidar).repeat_interleave(k_ll)
            dst_ll  = ll_i.flatten()
            dist_ll = ll_d.flatten() * 111_320.0    # graus → metros

            def _radius(s):
                r = torch.where(s < 5.0,  torch.tensor(120.0), torch.tensor(90.0))
                return torch.where(s < 15.0, r, torch.tensor(60.0))

            max_r = torch.minimum(_radius(lidar_slope[src_ll]),
                                  _radius(lidar_slope[dst_ll]))
            ok = dist_ll <= max_r

            src_ll  = src_ll[ok];  dst_ll  = dst_ll[ok]
            dist_ll = dist_ll[ok]
            ss_bi   = lidar_slope[torch.cat([src_ll, dst_ll])]
            sd_bi   = lidar_slope[torch.cat([dst_ll, src_ll])]
            src_bi  = torch.cat([src_ll, dst_ll])
            dst_bi  = torch.cat([dst_ll, src_ll])
            dist_bi = torch.cat([dist_ll, dist_ll])
            zdiff   = torch.abs(lidar_x[src_bi, 0] - lidar_x[dst_bi, 0])

            data['lidar', 'near_to', 'lidar'].edge_index = torch.stack([src_bi, dst_bi])
            data['lidar', 'near_to', 'lidar'].edge_attr  = torch.stack(
                [dist_bi, zdiff, ss_bi, sd_bi], dim=1)
            logger.info(f"   LiDAR↔LiDAR   : {src_bi.shape[0]:,}")
        else:
            data['lidar', 'near_to', 'lidar'].edge_index = torch.empty((2, 0), dtype=torch.long)
            data['lidar', 'near_to', 'lidar'].edge_attr  = torch.empty((0, 4), dtype=torch.float32)
    else:
        data['lidar', 'belongs_to', 'dem'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['lidar', 'belongs_to', 'dem'].edge_attr  = torch.empty((0, 2), dtype=torch.float32)
        data['lidar', 'near_to', 'dem'].edge_index    = torch.empty((2, 0), dtype=torch.long)
        data['lidar', 'near_to', 'dem'].edge_attr     = torch.empty((0, 2), dtype=torch.float32)
        data['lidar', 'near_to', 'lidar'].edge_index  = torch.empty((2, 0), dtype=torch.long)
        data['lidar', 'near_to', 'lidar'].edge_attr   = torch.empty((0, 4), dtype=torch.float32)

    # ── 8. METADATA — normalization V18.4 completo ───────────────────────────
    data.region       = city
    data.city_label   = CITY_CENTERS[city]["label"]
    data.sentinel_src = "real"

    data.raster_meta = {
        'dem_transform':  dem_transform,
        'sent_transform': sent_transform,
        'crs':            str(dem_crs),
        'dem_shape':      (H_dem, W_dem),
        'sent_shape':     (H_sent, W_sent),
        'bounds':         actual_bounds,
    }

    # Idêntico ao V18.4 — compatível com checkpoint GNN_TOPO
    data.normalization = {
        'elev_mean':     elev_mean,
        'elev_std':      elev_std,
        'slope_max':     SLOPE_MAX,
        'canopy_max':    CANOPY_MAX,
        'is_normalized': True,
        'version':       'V18.4',
    }

    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    # ── 9. RESUMO E SALVAR ────────────────────────────────────────────────────
    suffix   = "_dryrun" if dry_run else ""
    out_path = OUTPUT_DIR / f"{city}_v19{suffix}_gpu.pt"

    logger.info(f"\n{'─'*60}")
    logger.info(f"Resumo V19 {city.upper()}:")
    logger.info(f"  DEM nós          : {N_dem:>15,}  ({H_dem}×{W_dem})")
    logger.info(f"  Sentinel nós     : {N_sent:>15,}  ({H_sent}×{W_sent})")
    logger.info(f"  LiDAR nós        : {N_lidar:>15,}")
    logger.info(f"  DEM↔DEM arestas  : {edge_idx_dem.shape[1]:>15,}")
    logger.info(f"  Sentinel→DEM     : {N_sent:>15,}")
    logger.info(f"  S2 fonte         : REAL 10m (zeros onde sem dado)")
    logger.info(f"  Norm version     : V18.4 (compatível GNN_TOPO)")
    logger.info(f"  Pico VRAM        : {peak_vram:.2f} GB")
    logger.info(f"{'─'*60}")

    logger.info(f"Salvando: {out_path}")
    torch.save(data, out_path)
    size_mb = out_path.stat().st_size / (1024**2)
    logger.info(f"Concluído: {out_path.name}  ({size_mb:.0f} MB)")
    return out_path


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TerrainGNN V19 — Sentinel REAL 10m | 7200×7200 | Padrão V18.3")
    parser.add_argument("--city", type=str, choices=list(CITY_CENTERS.keys()),
                        help="Cidade: lins | campinas | bauru | sorocaba")
    parser.add_argument("--all",     action="store_true",
                        help="Gerar as 4 cidades em sequência")
    parser.add_argument("--dry-run", action="store_true",
                        help="Janela 1024×1024 para teste rápido")
    args = parser.parse_args()

    if args.all:
        cities  = list(CITY_CENTERS.keys())
        results = {}
        logger.info(f"Modo --all: {len(cities)} cidades")
        for city in cities:
            try:
                path = generate_graph_v19(city=city, dry_run=args.dry_run)
                results[city] = {"status": "ok", "path": str(path)}
            except Exception as exc:
                import traceback
                logger.error(f"Falha {city}: {exc}")
                traceback.print_exc()
                results[city] = {"status": "error", "error": str(exc)}
            torch.cuda.empty_cache()
            gc.collect()

        logger.info("\n" + "="*60 + "\nRESUMO FINAL:")
        for city, res in results.items():
            s = "OK" if res["status"] == "ok" else "ERRO"
            logger.info(f"  [{s}] {city}: {res.get('path', res.get('error'))}")

    elif args.city:
        generate_graph_v19(city=args.city, dry_run=args.dry_run)

    else:
        parser.print_help()
        print("\nExemplos:")
        print("  python generate_graph_v19.py --city lins")
        print("  python generate_graph_v19.py --all")
        print("  python generate_graph_v19.py --city campinas --dry-run")
