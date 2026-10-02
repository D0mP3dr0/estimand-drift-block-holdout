"""D5 (forum-eng-dados, responde R5 do parecer fisico_frequencia_gerador_baselines.md):
estende a checagem F2 (que confirmou Lins Q1) as outras 15 celulas.

Para cada celula (cidade x Q), amostra 20 000 nos de terreno (mesma semente do F2,
20260926) e recalcula a Haversine ate:
  (a) todas as antenas do inventario da cidade (prepare_transfer_dataset_v19.load_antennas,
      leitura direta do CSV ANATEL filtrado so por municipio, SEM filtro de frequencia) --
      equivalente a antenna.pos do proprio tensor (conferido pela chave extra
      'antenna_pos_do_tensor' quando presente);
  (b) so as antenas de 1600-2000 MHz (1800+-200) do mesmo subconjunto de cidade.
Reporta a fracao de nos em que dist_nearest_m do tensor bate (|dif|<1 m) com (a) e com (b).

Roda UMA celula por chamada (argv: cidade Q), grava/atualiza o JSON agregado.
Python: /trabalho/ambientes/s33_amb_virtual/.venv/bin/python. torch.load com
weights_only=False (HeteroData) e mmap=True (arquivos de ~28 GB); so as chaves
terrain.pos, terrain.dist_nearest_m, antenna.pos/x sao materializadas."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
GEN = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2")
sys.path.insert(0, str(GEN / "data_raw"))
sys.path.insert(0, str(GEN / "01_data"))
import enrich_rf_targets as E  # noqa: E402
import prepare_transfer_dataset_v19 as P  # noqa: E402

OUT_JSON = AQUI / "3.5_dist_nearest_definicao_16celulas.json"
DADOS = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")
CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QS = ["Q1", "Q2", "Q3", "Q4"]
SEMENTE = 20260926
N_AMOSTRA = 20_000


def sha256_arquivo(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def processa(cidade: str, q: str) -> dict:
    caminho = DADOS / f"transfer_dataset_{cidade}_v19_{q}_enriched_cftudo.pt"
    assert caminho.exists(), caminho
    t0 = time.time()
    rf = torch.load(str(caminho), map_location="cpu", weights_only=False, mmap=True)
    pos = torch.as_tensor(rf["terrain"].pos).float().clone()
    dist = torch.as_tensor(rf["terrain"].dist_nearest_m).float().clone()
    apos_t = torch.as_tensor(rf["antenna"].pos).float().clone() if hasattr(rf["antenna"], "pos") else None
    n_antenas_tensor = int(apos_t.shape[0]) if apos_t is not None else None
    del rf

    rng = np.random.default_rng(SEMENTE)
    idx = torch.from_numpy(np.sort(rng.choice(pos.shape[0], N_AMOSTRA, replace=False)))

    df = P.load_antennas(cidade)
    cand_a = torch.tensor(df[["lon", "lat"]].values, dtype=torch.float32)
    f = df["freq_mhz"].values
    cand_b = torch.tensor(df.loc[np.abs(f - 1800) <= 200, ["lon", "lat"]].values, dtype=torch.float32)

    out = {"caminho": str(caminho), "n_nos_terreno": int(pos.shape[0]),
           "n_antenas_tensor": n_antenas_tensor,
           "hipotese_a_todas_antenas_cidade": {
               "n_torres": int(cand_a.shape[0]),
           },
           "hipotese_b_1800pm200": {
               "n_torres": int(cand_b.shape[0]),
           }}
    for ap, rot in ((cand_a, "hipotese_a_todas_antenas_cidade"), (cand_b, "hipotese_b_1800pm200")):
        if ap.shape[0] == 0:
            out[rot]["frac_dif_lt_1m"] = None
            out[rot]["max_abs_dif_m"] = None
            continue
        d = E._dist_nearest_chunked(pos[idx], ap, chunk=20_000, device=torch.device("cpu"))
        diff = (d - dist[idx]).abs().numpy()
        out[rot]["max_abs_dif_m"] = float(diff.max())
        out[rot]["frac_dif_lt_1m"] = float(np.mean(diff < 1.0))
        out[rot]["mediana_d_km"] = float(np.median(d.numpy()) / 1e3)
    out["mediana_dist_nearest_tensor_km"] = float(np.median(dist[idx].numpy()) / 1e3)
    out["tempo_s"] = round(time.time() - t0, 1)
    out["sha256_tensor"] = sha256_arquivo(caminho)
    return out


def main():
    if len(sys.argv) != 3:
        print("uso: 3.5_dist_nearest_definicao_16celulas.py <cidade> <Qn>", file=sys.stderr)
        sys.exit(2)
    cidade, q = sys.argv[1], sys.argv[2]
    assert cidade in CIDADES and q in QS, (cidade, q)
    resultado_celula = processa(cidade, q)

    agregado = {}
    if OUT_JSON.exists():
        agregado = json.loads(OUT_JSON.read_text())
    agregado.setdefault("script_sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    agregado.setdefault("semente_amostra", SEMENTE)
    agregado.setdefault("n_amostra", N_AMOSTRA)
    agregado.setdefault("celulas", {})
    agregado["celulas"][f"{cidade}_{q}"] = resultado_celula
    agregado["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT_JSON.write_text(json.dumps(agregado, indent=1, ensure_ascii=False))
    print(json.dumps(resultado_celula, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
