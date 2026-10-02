#!/trabalho/ambientes/s33_amb_virtual/.venv/bin/python
"""Suplemento POST HOC ao lote A4 (criterio_A4_suplemento_validos.json, emenda 09:50).

Pergunta: nos mesmos 20 sorteios (Bauru Q1-Q4 x split seeds 101-105), em quantos o
MAE_validos do modelo treinado (GNN, MLP) e menor que o MAE_validos da baseline
analitica calibrada como no script congelado (P_tx = mediana sobre todos os nos de
treino, lido de baselines_analiticos.test.p_tx_eff_dbm do run JSON)?

Fontes: gpu/A4/<run>/predicoes_*.npz (idx_global, target[:,3], sentinela),
run_*.json (p_tx_eff_dbm, freq_mhz, baseline_<b>_mae para autoverificacao),
tensor da celula (dist_nearest_m), formulas do script congelado (sem copia).
Autoverificacao: o MAE da baseline sobre TODOS os nos de teste deve reproduzir o
valor do run JSON em 20/20 sorteios; senao a saida e marcada NAO citavel.
Saida: gpu/A4/suplemento_inversao_validos_A4.json
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import numpy as np
import torch

RAIZ = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
A4 = RAIZ / "gpu" / "A4"
CRIT = RAIZ / "criterios" / "criterio_A4_suplemento_validos.json"
DADOS = Path("/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3")
BASELINES_CONGELADO = Path(
    "/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/FIRST_RESPONSE_REVIEW_IEEE_ACESSES/"
    "EVIDENCIA_RESUBMISSAO/dados/scripts_congelados/baselines_v2_por_particao.py")
OUT = A4 / "suplemento_inversao_validos_A4.json"
CELULAS = [("bauru", q) for q in ("Q1", "Q2", "Q3", "Q4")]
SEEDS = [101, 102, 103, 104, 105]
BASELINES = ("fspl", "hata_rural", "cost231_sub")
TOL_AUTOVERIF_DB = 1e-3


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def carregar_formulas():
    spec = importlib.util.spec_from_file_location("baselines_congelado", BASELINES_CONGELADO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {
        "fspl": lambda d, f: mod.free_space_path_loss(d, f),
        "hata_rural": lambda d, f: mod.okumura_hata_rural(d, f),
        "cost231_sub": lambda d, f: mod.cost231_hata(d, f, environment="suburban"),
    }


def dist_da_celula(cidade: str, q: str) -> np.ndarray:
    caminho = DADOS / f"transfer_dataset_{cidade}_v19_{q}_enriched_cftudo.pt"
    rf = torch.load(str(caminho), map_location="cpu", weights_only=False, mmap=True)
    return torch.as_tensor(rf["terrain"].dist_nearest_m).float().clone().numpy(), caminho


def mae(a: np.ndarray, b: np.ndarray) -> float | None:
    return float(np.mean(np.abs(a - b))) if a.size else None


def main() -> int:
    formulas = carregar_formulas()
    saida = {
        "artefato": "suplemento POST HOC ao A4 (nao pre-registrado no criterio_A4)",
        "criterio": str(CRIT.relative_to(RAIZ)), "criterio_sha256": sha(CRIT),
        "baselines_congelado_sha256": sha(BASELINES_CONGELADO),
        "calibracao": "(a) P_tx_eff do run JSON (mediana sobre todos os nos de treino), unica computavel",
        "autoverificacao_ok_todas": True, "celulas": {}, "contagens_20": {},
    }
    contagens = {m: {b: 0 for b in BASELINES} for m in ("gnn", "mlp")}
    n_total = 0
    for cidade, q in CELULAS:
        dist_all, tpath = dist_da_celula(cidade, q)
        cel = {"tensor": str(tpath), "por_sorteio": {}}
        for s in SEEDS:
            reg = {}
            npz_gnn = A4 / f"gnn_v3_a4_{cidade}_{q}_ss{s}" / f"predicoes_gnn_v3_a4_{cidade}_{q}_ss{s}.npz"
            run_gnn = A4 / f"gnn_v3_a4_{cidade}_{q}_ss{s}" / f"run_gnn_v3_a4_{cidade}_{q}_ss{s}.json"
            z = np.load(npz_gnn, allow_pickle=False)
            idx = z["idx_global"].astype(np.int64)
            tgt = z["target"]
            val = tgt[:, 0] < 299.0
            rssi = tgt[:, 3].astype(np.float64)
            rj = json.loads(run_gnn.read_text(encoding="utf-8"))
            bt = rj["baselines_analiticos"]["test"]
            ptx, freq = bt["p_tx_eff_dbm"], float(bt["freq_mhz"])
            d = dist_all[idx].astype(np.float64)
            reg["n"], reg["n_validos"], reg["freq_mhz"] = int(idx.size), int(val.sum()), freq
            reg["baselines"] = {}
            for b in BASELINES:
                pl = np.asarray(formulas[b](d, freq), dtype=np.float64)
                pred = float(ptx[b]) - pl
                e_all = mae(pred, rssi)
                reg["baselines"][b] = {
                    "mae_todos_db": e_all, "mae_todos_run_json_db": bt[f"baseline_{b}_mae"],
                    "autoverificacao_ok": abs(e_all - bt[f"baseline_{b}_mae"]) < TOL_AUTOVERIF_DB,
                    "mae_validos_db": mae(pred[val], rssi[val]), "mae_sentinela_db": mae(pred[~val], rssi[~val]),
                }
                if not reg["baselines"][b]["autoverificacao_ok"]:
                    saida["autoverificacao_ok_todas"] = False
            # MAE_validos dos modelos, recalculado dos .npz (mesma regra do A4_agregar)
            reg["modelos"] = {}
            for m in ("gnn", "mlp"):
                zp = np.load(A4 / f"{m}_v3_a4_{cidade}_{q}_ss{s}" / f"predicoes_{m}_v3_a4_{cidade}_{q}_ss{s}.npz", allow_pickle=False)
                assert np.array_equal(zp["idx_global"], idx), "particao diferente GNN x MLP"
                vm = zp["target"][:, 0] < 299.0
                reg["modelos"][m] = {"mae_validos_db": mae(zp["pred"][vm, 3].astype(np.float64), zp["target"][vm, 3].astype(np.float64))}
                reg["modelos"][m]["melhor_que"] = {}
                for b in BASELINES:
                    ok = reg["modelos"][m]["mae_validos_db"] < reg["baselines"][b]["mae_validos_db"]
                    reg["modelos"][m]["melhor_que"][b] = bool(ok)
                    contagens[m][b] += int(ok)
            n_total += 1
            cel["por_sorteio"][str(s)] = reg
        # distribuicoes por celula
        cel["distribuicao_mae_validos_baseline_db"] = {
            b: {"mediana": statistics.median(r["baselines"][b]["mae_validos_db"] for r in cel["por_sorteio"].values()),
                "min": min(r["baselines"][b]["mae_validos_db"] for r in cel["por_sorteio"].values()),
                "max": max(r["baselines"][b]["mae_validos_db"] for r in cel["por_sorteio"].values())}
            for b in BASELINES}
        cel["inversao_validos_por_modelo"] = {
            m: {b: sum(int(r["modelos"][m]["melhor_que"][b]) for r in cel["por_sorteio"].values()) for b in BASELINES}
            for m in ("gnn", "mlp")}
        saida["celulas"][f"{cidade}_{q}"] = cel
        print(f"{cidade}_{q}: baseline FSPL validos mediana {cel['distribuicao_mae_validos_baseline_db']['fspl']['mediana']:.2f} dB | "
              f"inversao gnn {cel['inversao_validos_por_modelo']['gnn']} mlp {cel['inversao_validos_por_modelo']['mlp']}")
    saida["contagens_20"] = {m: {b: {"n_sorteios_modelo_melhor": contagens[m][b], "n": n_total} for b in BASELINES} for m in contagens}
    saida["citavel_apos_votos"] = bool(saida["autoverificacao_ok_todas"]) and n_total == 20
    OUT.write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")
    print("autoverificacao (MAE todos os nos == run JSON, 20/20):", saida["autoverificacao_ok_todas"])
    print("contagens sobre 20:", json.dumps(saida["contagens_20"]))
    print("gravado:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
