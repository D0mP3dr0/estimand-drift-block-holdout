"""
Split {city}_v19_gpu.pt (7200x7200) em 4 quadrantes de 3600x3600
=================================================================
Compativel com SpatialHeteroGraphDataset e extract_embeddings pipeline.
Suporta qualquer cidade do projeto V19 via --city.

Quadrantes:
  Q1 (NW): rows[0:3600,    cols  0:3600]
  Q2 (NE): rows[0:3600,    cols 3600:7200]
  Q3 (SW): rows[3600:7200, cols  0:3600]
  Q4 (SE): rows[3600:7200, cols 3600:7200]

Saida (ex. campinas):
  graph_data/campinas_v19_Q1_gpu.pt  (~15 GB cada)
  graph_data/campinas_v19_Q2_gpu.pt
  graph_data/campinas_v19_Q3_gpu.pt
  graph_data/campinas_v19_Q4_gpu.pt

Uso:
  python split_v19_quadrants.py                          # lins (padrao)
  python split_v19_quadrants.py --city campinas
  python split_v19_quadrants.py --city sorocaba --quadrants Q1 Q2
  python split_v19_quadrants.py --city lins --output-dir E:\\GRAPH_V19
  python split_v19_quadrants.py --quadrants Q1 Q2        # apenas dois (lins)
"""

import argparse
import gc
import logging
import time
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# ─── Caminhos base ────────────────────────────────────────────────────────────
_GRAPH_DATA = Path(r"F:\arpia_topo_refinado\TOPO_RF\GNN_RF_V2\graph_data")

# Grade original
H_FULL = 7200
W_FULL = 7200
H_Q    = 3600   # metade da grade
W_Q    = 3600

QUADRANT_DEFS = {
    "Q1": (0,    H_Q,    0,    W_Q),    # NW
    "Q2": (0,    H_Q,    W_Q,  W_FULL), # NE
    "Q3": (H_Q,  H_FULL, 0,    W_Q),    # SW
    "Q4": (H_Q,  H_FULL, W_Q,  W_FULL), # SE
}


# ─── Utilitarios ─────────────────────────────────────────────────────────────
def _make_g2l(glob_idx: torch.Tensor, total: int) -> torch.Tensor:
    """Cria mapeamento global→local como int32 (economiza memoria vs int64)."""
    g2l = torch.full((total,), -1, dtype=torch.int32)
    g2l[glob_idx] = torch.arange(glob_idx.shape[0], dtype=torch.int32)
    return g2l


def _filter_edges(ei: torch.Tensor, ea: torch.Tensor,
                  src_mask: torch.Tensor, dst_mask: torch.Tensor,
                  src_g2l: torch.Tensor, dst_g2l: torch.Tensor,
                  label: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Filtra e re-indexa arestas. src_g2l e dst_g2l sao int32."""
    s, d = ei
    mask = src_mask[s] & dst_mask[d]
    new_s = src_g2l[s[mask]].long()
    new_d = dst_g2l[d[mask]].long()
    n = mask.sum().item()
    log.info(f"    {label}: {n:,} arestas")
    return torch.stack([new_s, new_d]), ea[mask]


# ─── Split de um quadrante ───────────────────────────────────────────────────
def split_quadrant(src: HeteroData, q_name: str,
                   r0: int, r1: int, c0: int, c1: int,
                   city: str = "lins") -> HeteroData:
    t0 = time.time()
    H_q = r1 - r0
    W_q = c1 - c0
    N_dem_q = H_q * W_q
    log.info(f"\n{'='*60}")
    log.info(f"Quadrante {q_name}: rows[{r0}:{r1}] cols[{c0}:{c1}] "
             f"→ {H_q}×{W_q} = {N_dem_q:,} nos DEM")

    # ── DEM: mascara por posicao na grade full ────────────────────────────────
    dem_pix = src['dem'].pixel_idx          # [N_full] — idx na grade 7200x7200
    dem_row = (dem_pix // W_FULL)
    dem_col = (dem_pix %  W_FULL)
    dem_mask = (dem_row >= r0) & (dem_row < r1) & \
               (dem_col >= c0) & (dem_col < c1)   # [N_full] bool
    dem_glob = torch.where(dem_mask)[0]            # indices globais selecionados

    assert dem_glob.shape[0] == N_dem_q, \
        f"DEM mismatch: {dem_glob.shape[0]} != {N_dem_q}"

    # pixel_idx local (referencia a grade 3600x3600)
    local_row = dem_row[dem_glob] - r0
    local_col = dem_col[dem_glob] - c0
    local_pix = local_row * W_q + local_col

    dem_g2l = _make_g2l(dem_glob, H_FULL * W_FULL)  # 415 MB como int32
    log.info(f"  DEM: {N_dem_q:,} nos OK")

    # ── Sentinel: filtrar pelo parent DEM ────────────────────────────────────
    H_sent_q  = H_q * 3
    W_sent_q  = W_q * 3
    W_sent_full = W_FULL * 3
    N_sent_q  = H_sent_q * W_sent_q

    sent_parent_glob = src['sentinel'].parent_idx   # [N_sent_full] — DEM global idx
    sent_mask = dem_mask[sent_parent_glob]          # bool [N_sent_full]
    sent_glob = torch.where(sent_mask)[0]

    assert sent_glob.shape[0] == N_sent_q, \
        f"Sentinel mismatch: {sent_glob.shape[0]} != {N_sent_q}"

    # pixel_idx local para sentinel
    sent_pix_g = src['sentinel'].pixel_idx[sent_glob]
    s_row_g = sent_pix_g // W_sent_full
    s_col_g = sent_pix_g %  W_sent_full
    sent_pix_local = (s_row_g - r0 * 3) * W_sent_q + (s_col_g - c0 * 3)
    del s_row_g, s_col_g, sent_pix_g

    # g2l para sentinel (int32 = 1.86 GB)
    sent_g2l = _make_g2l(sent_glob, H_FULL * W_FULL * 9)
    log.info(f"  Sentinel: {N_sent_q:,} nos OK")

    # ── LiDAR: filtrar pelo parent DEM ───────────────────────────────────────
    N_lidar_full = src['lidar'].x.shape[0]
    lidar_g2l = None
    lidar_glob = None
    lidar_mask = None

    if N_lidar_full > 0:
        lidar_parent_glob = src['lidar'].parent_idx   # [N_lidar_full]
        lidar_mask = dem_mask[lidar_parent_glob]
        lidar_glob = torch.where(lidar_mask)[0]
        N_lidar_q  = lidar_glob.shape[0]
        lidar_g2l  = _make_g2l(lidar_glob, N_lidar_full)
        log.info(f"  LiDAR: {N_lidar_q:,} pontos OK")
    else:
        N_lidar_q = 0

    # ── Montar HeteroData do quadrante ───────────────────────────────────────
    q = HeteroData()

    # DEM
    q['dem'].x          = src['dem'].x[dem_glob]
    q['dem'].y          = src['dem'].y[dem_glob]
    q['dem'].pos        = src['dem'].pos[dem_glob]
    q['dem'].pixel_idx  = local_pix
    q['dem'].node_id    = torch.arange(N_dem_q, dtype=torch.long)
    q['dem'].grid_shape = torch.tensor([H_q, W_q], dtype=torch.long)

    # Sentinel
    q['sentinel'].x          = src['sentinel'].x[sent_glob]
    q['sentinel'].pos        = src['sentinel'].pos[sent_glob]
    q['sentinel'].parent_idx = dem_g2l[sent_parent_glob[sent_glob]].long()
    q['sentinel'].pixel_idx  = sent_pix_local
    q['sentinel'].node_id    = torch.arange(N_sent_q, dtype=torch.long)
    q['sentinel'].grid_shape = torch.tensor([H_sent_q, W_sent_q], dtype=torch.long)

    # LiDAR
    if N_lidar_q > 0:
        q['lidar'].x          = src['lidar'].x[lidar_glob]
        q['lidar'].pos        = src['lidar'].pos[lidar_glob]
        q['lidar'].parent_idx = dem_g2l[src['lidar'].parent_idx[lidar_glob]].long()
        q['lidar'].node_id    = torch.arange(N_lidar_q, dtype=torch.long)
    else:
        q['lidar'].x          = torch.zeros((0, 3),  dtype=torch.float32)
        q['lidar'].pos        = torch.zeros((0, 2),  dtype=torch.float32)
        q['lidar'].parent_idx = torch.zeros((0,),    dtype=torch.long)
        q['lidar'].node_id    = torch.zeros((0,),    dtype=torch.long)

    # ── Arestas ───────────────────────────────────────────────────────────────
    # 1. dem ↔ dem
    ei, ea = (src['dem', 'adjacent_to', 'dem'].edge_index,
              src['dem', 'adjacent_to', 'dem'].edge_attr)
    ei_q, ea_q = _filter_edges(ei, ea, dem_mask, dem_mask,
                                dem_g2l, dem_g2l, "dem↔dem")
    q['dem', 'adjacent_to', 'dem'].edge_index = ei_q
    q['dem', 'adjacent_to', 'dem'].edge_attr  = ea_q
    del ei_q, ea_q; gc.collect()

    # 2. sentinel → dem
    ei, ea = (src['sentinel', 'belongs_to', 'dem'].edge_index,
              src['sentinel', 'belongs_to', 'dem'].edge_attr)
    ei_q, ea_q = _filter_edges(ei, ea, sent_mask, dem_mask,
                                sent_g2l, dem_g2l, "sentinel→dem")
    q['sentinel', 'belongs_to', 'dem'].edge_index = ei_q
    q['sentinel', 'belongs_to', 'dem'].edge_attr  = ea_q
    del ei_q, ea_q; gc.collect()

    # 3. lidar → dem (belongs_to)
    if N_lidar_q > 0:
        ei, ea = (src['lidar', 'belongs_to', 'dem'].edge_index,
                  src['lidar', 'belongs_to', 'dem'].edge_attr)
        ei_q, ea_q = _filter_edges(ei, ea, lidar_mask, dem_mask,
                                   lidar_g2l, dem_g2l, "lidar→dem(belongs)")
        q['lidar', 'belongs_to', 'dem'].edge_index = ei_q
        q['lidar', 'belongs_to', 'dem'].edge_attr  = ea_q
        del ei_q, ea_q

        # 4. lidar → dem (near_to)
        ei, ea = (src['lidar', 'near_to', 'dem'].edge_index,
                  src['lidar', 'near_to', 'dem'].edge_attr)
        # near_to edges: src pode ser lidar de fora, dst DEM dentro
        # Usar apenas arestas onde o LiDAR src está no quadrante
        lidar_src_mask = torch.zeros(N_lidar_full, dtype=torch.bool)
        lidar_src_mask[lidar_glob] = True
        ei_q, ea_q = _filter_edges(ei, ea, lidar_src_mask, dem_mask,
                                   lidar_g2l, dem_g2l, "lidar→dem(near)")
        q['lidar', 'near_to', 'dem'].edge_index = ei_q
        q['lidar', 'near_to', 'dem'].edge_attr  = ea_q
        del ei_q, ea_q

        # 5. lidar ↔ lidar
        ei, ea = (src['lidar', 'near_to', 'lidar'].edge_index,
                  src['lidar', 'near_to', 'lidar'].edge_attr)
        ei_q, ea_q = _filter_edges(ei, ea, lidar_src_mask, lidar_src_mask,
                                   lidar_g2l, lidar_g2l, "lidar↔lidar")
        q['lidar', 'near_to', 'lidar'].edge_index = ei_q
        q['lidar', 'near_to', 'lidar'].edge_attr  = ea_q
        del ei_q, ea_q, lidar_src_mask; gc.collect()
    else:
        q['lidar', 'belongs_to', 'dem'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        q['lidar', 'belongs_to', 'dem'].edge_attr  = torch.zeros((0, 2), dtype=torch.float32)
        q['lidar', 'near_to', 'dem'].edge_index    = torch.zeros((2, 0), dtype=torch.long)
        q['lidar', 'near_to', 'dem'].edge_attr     = torch.zeros((0, 2), dtype=torch.float32)
        q['lidar', 'near_to', 'lidar'].edge_index  = torch.zeros((2, 0), dtype=torch.long)
        q['lidar', 'near_to', 'lidar'].edge_attr   = torch.zeros((0, 4), dtype=torch.float32)

    # ── Metadata ──────────────────────────────────────────────────────────────
    city_label_map = {
        "lins": "Lins SP", "campinas": "Campinas SP",
        "sorocaba": "Sorocaba SP", "bauru": "Bauru SP",
    }
    q.region       = f"{city}_{q_name.lower()}"
    q.city_label   = f"{city_label_map.get(city, city.title())} {q_name}"
    q.sentinel_src = src.sentinel_src
    q.normalization = src.normalization   # mesmos stats de normalizacao
    q.raster_meta   = src.raster_meta    # transform da grade full (aproximado)
    q.quadrant      = {"name": q_name, "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                       "full_grid": [H_FULL, W_FULL]}

    elapsed = time.time() - t0
    log.info(f"  Quadrante {q_name} processado em {elapsed:.0f}s")
    return q


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Split {city}_v19_gpu.pt em 4 quadrantes de 3600x3600")
    parser.add_argument("--city", type=str, default="lins",
                        choices=["lins", "campinas", "sorocaba", "bauru"],
                        help="Cidade a processar (padrao: lins)")
    parser.add_argument("--src", type=str, default="",
                        help="Caminho alternativo para o grafo completo .pt "
                             "(padrao: graph_data/{city}_v19_gpu.pt)")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Diretorio de saida dos quadrantes "
                             "(padrao: graph_data/)")
    parser.add_argument("--quadrants", nargs="+",
                        default=list(QUADRANT_DEFS.keys()),
                        choices=list(QUADRANT_DEFS.keys()),
                        help="Quadrantes a gerar (padrao: todos)")
    args = parser.parse_args()

    city    = args.city.lower()
    src_path = Path(args.src) if args.src else (_GRAPH_DATA / f"{city}_v19_gpu.pt")
    out_dir  = Path(args.output_dir) if args.output_dir else _GRAPH_DATA
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"SPLIT V19 — {city.upper()} → 4 QUADRANTES 3600x3600")
    log.info("=" * 60)
    log.info(f"Fonte : {src_path}")
    log.info(f"Saida : {out_dir}")
    log.info(f"Quadrantes: {args.quadrants}")

    if not src_path.exists():
        log.error(f"Arquivo fonte nao encontrado: {src_path}")
        log.error(f"Execute primeiro: python data_raw/generate_graph_v19.py --city {city}")
        raise SystemExit(1)

    t_total = time.time()

    log.info("\nCarregando grafo completo (pode demorar ~1-2 min)...")
    src = torch.load(src_path, map_location="cpu", weights_only=False)
    dem_gs = src['dem'].grid_shape
    H, W = (dem_gs.tolist() if torch.is_tensor(dem_gs) else list(dem_gs))
    log.info(f"  Carregado: dem {H}x{W}  sentinel {src['sentinel'].x.shape[0]:,}  "
             f"lidar {src['lidar'].x.shape[0]:,}")
    assert H == H_FULL and W == W_FULL, \
        f"Grade inesperada: {H}x{W}, esperado {H_FULL}x{W_FULL}"

    for q_name in args.quadrants:
        r0, r1, c0, c1 = QUADRANT_DEFS[q_name]
        q = split_quadrant(src, q_name, r0, r1, c0, c1, city=city)

        out_path = out_dir / f"{city}_v19_{q_name}_gpu.pt"
        log.info(f"  Salvando → {out_path.name} ...")
        torch.save(q, out_path)
        sz_gb = out_path.stat().st_size / (1024**3)
        log.info(f"  Salvo: {sz_gb:.1f} GB")

        del q; gc.collect()

    elapsed_total = (time.time() - t_total) / 60
    log.info(f"\n{'='*60}")
    log.info(f"CONCLUIDO em {elapsed_total:.1f} min")
    log.info("Arquivos gerados:")
    for q_name in args.quadrants:
        p = out_dir / f"{city}_v19_{q_name}_gpu.pt"
        if p.exists():
            log.info(f"  {p.name}  {p.stat().st_size/(1024**3):.1f} GB")
    log.info(f"\nProximo passo — extrair embeddings:")
    log.info(f"  python extract_topo_embeddings_v19.py --city {city} --all --biome cerrado")


if __name__ == "__main__":
    main()
