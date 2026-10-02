"""Etapa 1 da regeracao: grafo de terreno da cidade, com o MOSAICO CORRIGIDO.

ARQUIVO NOVO. Nao altera `GNN_RF_V2/data_raw/generate_graph_v19.py`: importa
aquele modulo e redireciona as constantes de caminho, no mesmo padrao que
`rodar_mc_dropout_q3q4.py` ja usa na casa.

O QUE REDIRECIONA E POR QUE:
  BASE_DIR   -> o original aponta para F:\\arpia_topo_refinado\\..., que NAO
                existe nesta maquina (os insumos vivem em D:\\_ARQUIVO_SSD_F).
  DEM_MOSAIC -> `interior_sp_dem_consolidated_v2.tif`, o mosaico refeito com
                rasterio.merge. O antigo devolve ZERO em 100% dos pontos de
                sorocaba Q3/Q4, 90% de campinas Q3/Q4 e 50% de sorocaba Q1/Q2
                (medido em refazer_mosaico_dem_v2.json). Bauru, Lins e
                campinas Q1 dao desvio IDENTICO nos dois mosaicos, entao a
                troca nao mexe em quem ja estava certo.
  OUTPUT_DIR -> pasta nova em F:, que tem o espaco (402 GB livres).

NAO redireciona mais nada: derivadas, KNN de LiDAR, leitura do Sentinel-2 e a
montagem do HeteroData continuam sendo o codigo original, sem uma linha
reescrita.

Uso:
  python gerar_grafo_v19_mosaico_corrigido.py --city campinas [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

V2 = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2")
EVID = Path(__file__).resolve().parents[1]
OUT_JSON = EVID / "dados" / "alvo"
DESTINO = Path(r"F:\TOPO_RF_DOWNLOAD_DRIVE\graph_data_v3")
MOSAICO_V2 = (V2 / "data_raw" / "satelites_raw" / "dem"
              / "interior_sp_dem_consolidated_v2.tif")

sys.path.insert(0, str(V2 / "data_raw"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="janela 1024x1024 do proprio gerador; valida a cadeia em minutos")
    args = ap.parse_args()

    if not MOSAICO_V2.exists():
        print(f"ABORTA: mosaico corrigido ausente: {MOSAICO_V2}")
        print("  rode antes: refazer_mosaico_dem_v2.py")
        return 1

    import generate_graph_v19 as G

    base = V2 / "data_raw" / "satelites_raw"
    G.BASE_DIR = base
    G.DEM_TILES_DIR = base / "dem" / "tiles"
    G.DEM_MOSAIC = MOSAICO_V2          # <- a correcao
    G.LIDAR_FILE = base / "lidar" / "lidar_interior_sp.csv"
    G.S2_BASE_DIR = base / "sentinel2"
    DESTINO.mkdir(parents=True, exist_ok=True)
    G.OUTPUT_DIR = DESTINO

    # --- Sentinel-2: segundo defeito, achado pelo piloto -------------------
    # O piloto de campinas registrou "TIFs S2 reais carregados: 0 | Cobertura
    # S2 real: 0.0%". A causa em generate_graph_v19.py:93-105: `REAL_S2_TIFS`
    # e calculado NO IMPORT, com o S2_BASE_DIR original apontando para
    # F:\arpia_topo_refinado, que nao existe nesta maquina. A lista nasce
    # vazia e nada depois a corrige. O layout em subpasta por cidade que a
    # funcao espera esta certo — o que falta e recalcular apos redirecionar.
    # Gerar sem isso produziria as 6 colunas espectrais zeradas — o mesmo tipo
    # de buraco que estamos consertando no terreno.
    #
    # Preferencia por `_2deg_v2.tif`: sao os mosaicos S2 REFEITOS em 02-03/09
    # para tapar o zero-fill de NDVI que a auditoria de eng-dados achou no
    # bauru_Q2 (86,8% dos nos sem banda espectral). Os `_2deg.tif` sem sufixo
    # sao os antigos, de marco. Usar o v2 e a mesma escolha que o E2 ja fez.
    cidades = ["lins", "campinas", "bauru", "sorocaba"]
    s2 = []
    escolha = {}
    for c in cidades:
        d = G.S2_BASE_DIR / c
        cands = [d / f"interior_sp_s2_{c}_10m_real_2deg_v2.tif",
                 d / f"interior_sp_s2_{c}_10m_real_2deg.tif",
                 d / f"interior_sp_s2_{c}_10m_real.tif"]
        achado = next((p for p in cands if p.exists()), None)
        if achado:
            s2.append(achado)
            escolha[c] = achado.name
    if not s2:
        print(f"ABORTA: nenhum TIF Sentinel-2 encontrado em {G.S2_BASE_DIR}")
        return 1
    G.REAL_S2_TIFS = s2
    if hasattr(G, "_build_real_s2_list"):
        G._build_real_s2_list = lambda: s2
    print("Sentinel-2 em uso:")
    for c, n in escolha.items():
        print(f"   {c:<10} {n}")

    faltando = [str(p) for p in (G.DEM_MOSAIC, G.DEM_TILES_DIR, G.LIDAR_FILE,
                                 G.S2_BASE_DIR) if not p.exists()]
    if faltando:
        print("ABORTA: insumos ausentes:")
        for f in faltando:
            print(f"  {f}")
        return 1

    print(f"cidade      : {args.city}")
    print(f"mosaico     : {G.DEM_MOSAIC}")
    print(f"saida       : {G.OUTPUT_DIR}")
    print(f"dry-run     : {args.dry_run}", flush=True)

    t0 = datetime.now(timezone.utc)
    out = G.generate_graph_v19(args.city, dry_run=args.dry_run)
    dt = (datetime.now(timezone.utc) - t0).total_seconds()

    src_py = V2 / "data_raw" / "generate_graph_v19.py"
    rec = {
        "timestamp_utc": t0.isoformat(),
        "cidade": args.city, "dry_run": args.dry_run,
        "segundos": round(dt, 1),
        "script_original": str(src_py),
        "script_original_sha256": hashlib.sha256(src_py.read_bytes()).hexdigest(),
        "mosaico_usado": str(G.DEM_MOSAIC),
        "sentinel2_usado": escolha,
        "mosaico_sha256_nao_calculado": "arquivo de 2,4 GB; hash sob demanda",
        "saida": str(out), "saida_bytes": out.stat().st_size,
        "nota": ("mosaico corrigido: o consolidado antigo devolvia 0.0 em 100% "
                 "dos pontos de sorocaba Q3/Q4 (ver refazer_mosaico_dem_v2.json)"),
    }
    OUT_JSON.mkdir(parents=True, exist_ok=True)
    tag = "_dryrun" if args.dry_run else ""
    dest = OUT_JSON / f"gerar_grafo_{args.city}{tag}.json"
    dest.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nconcluido em {dt/60:.1f} min -> {out}")
    print(f"registro: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
