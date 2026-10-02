"""Etapa 3 da regeracao: transfer_dataset por quadrante COM seed 42 no A_terrain.

ARQUIVO NOVO. Nao altera `01_data/prepare_transfer_dataset_v19.py`: importa o
modulo e chama `run_per_quadrant`, com uma unica coisa acrescentada antes —
`np.random.seed(42)`.

POR QUE ISSO E OBRIGATORIO: `prepare_transfer_dataset_v19.py:141` faz
`np.random.uniform(0.02, 0.15, n_nodes)` para o A_terrain SEM semente nenhuma.
Rodar sem semear produziria um alvo irreproduzivel — o revisor R1 pediu
exatamente reprodutibilidade, e o dono ratificou em 2026-09-03 que todo
rebuild v2 usa seed 42 nesse ponto. E a mesma semente, no mesmo lugar, que o
`e3_alvos_dataset_v2_multi.py` ja aplica.

O sorteio e consumido por `generate_realistic_coverage.py:109-116`, entao
semear aqui cobre a cadeia inteira do alvo.

Uso:
  python preparar_transfer_seed42.py --city campinas --graph-dir <dir> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

V2 = Path(r"D:\_ARQUIVO_SSD_F\TOPO_RF\GNN_RF_V2")
EVID = Path(__file__).resolve().parents[1]
OUT_JSON = EVID / "dados" / "alvo"
SEED_A_TERRAIN = 42

sys.path.insert(0, str(V2 / "01_data"))
sys.path.insert(0, str(V2 / "data_raw"))
sys.path.insert(0, str(V2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--struct-dir", default="")
    ap.add_argument("--quadrants", nargs="+", default=["Q1", "Q2", "Q3", "Q4"])
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    import prepare_transfer_dataset_v19 as P

    gd = Path(args.graph_dir)
    od = Path(args.output_dir)
    sd = Path(args.struct_dir) if args.struct_dir else gd
    od.mkdir(parents=True, exist_ok=True)
    P.GRAPH_DIR = gd
    P.OUTPUT_DIR = od
    P.STRUCT_DIR = sd

    src = V2 / "01_data" / "prepare_transfer_dataset_v19.py"
    print(f"cidade     : {args.city}")
    print(f"graph_dir  : {gd}")
    print(f"output_dir : {od}")
    print(f"seed A_terrain: {SEED_A_TERRAIN} (prepare_transfer_dataset_v19.py:141 "
          f"nao semeia sozinho)", flush=True)

    t0 = datetime.now(timezone.utc)
    resultados = {}
    for q in args.quadrants:
        # semeia IMEDIATAMENTE antes de cada quadrante, para que o alvo de cada
        # um dependa so da semente e nao da ordem em que os quadrantes rodaram
        np.random.seed(SEED_A_TERRAIN)
        print(f"\n--- {args.city} {q} (np.random.seed({SEED_A_TERRAIN}) aplicado) ---",
              flush=True)
        try:
            # BUG CORRIGIDO 06/09: `run_per_quadrant` recebe a cidade como
            # parametro com DEFAULT "lins" (prepare_transfer_dataset_v19.py:318).
            # A chamada anterior omitia `city=`, entao a fila pedia sorocaba e o
            # programa construia lins — o log registrou "Antenas Lins: 175" e
            # "Construindo transfer_dataset_lins_v19_Q4.pt" numa corrida de
            # sorocaba. Passar a cidade explicitamente e obrigatorio.
            P.run_per_quadrant([q], skip_existing=args.skip_existing,
                               city=args.city.lower())
            resultados[q] = "ok"
        except Exception as e:
            resultados[q] = f"FALHOU: {type(e).__name__}: {e}"
            print(f"[ERRO] {q}: {e}", flush=True)

    rec = {
        "timestamp_utc": t0.isoformat(),
        "cidade": args.city,
        "seed_a_terrain": SEED_A_TERRAIN,
        "onde_a_semente_entra": ("np.random.seed(42) antes de run_per_quadrant; "
                                 "consumida por prepare_transfer_dataset_v19.py:141 "
                                 "(np.random.uniform 0.02-0.15) -> "
                                 "generate_realistic_coverage.py:109-116"),
        "ratificada_pelo_dono_em": "2026-09-03 (rota tudo-v2)",
        "script_original": str(src),
        "script_original_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "graph_dir": str(gd), "output_dir": str(od), "struct_dir": str(sd),
        "quadrantes": resultados,
        "segundos": round((datetime.now(timezone.utc) - t0).total_seconds(), 1),
    }
    OUT_JSON.mkdir(parents=True, exist_ok=True)
    dest = OUT_JSON / f"preparar_transfer_{args.city}.json"
    dest.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nregistro: {dest}")
    ruins = [q for q, v in resultados.items() if v != "ok"]
    return 1 if ruins else 0


if __name__ == "__main__":
    sys.exit(main())
