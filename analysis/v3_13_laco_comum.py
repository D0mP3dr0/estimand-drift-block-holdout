#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v3_13_laco_comum.py -- laco compartilhado por R5 (variancia de um sorteio) e R6 (nivel do
sentinela). Criterios: _v3_2026-09-25/criterios/criterio_R5_variancia_um_sorteio.json e
criterio_R6_nivel_do_sentinela.json (fixados em 2026-10-01T22:47:16-03:00, antes deste script).

REUSA POR IMPORT, sem copia e sem edicao, a particao e a calibracao de
scripts/v3_2.1_3.1_deriva_calibracao.py (sha256 conferido em execucao; aborta se divergir),
como fez scripts/v3_12_R1_b0_deriva.py: split_espacial_3vias (g = 10 km, b = 2 km, fracs
0.70/0.15/0.15), modelos FSPL do script congelado, `orig.mae`, `orig.carregar_tensor`,
`orig.latlon_graus_para_metros`, `orig.SpatialKFold`. As linhas de carga do tensor e de
calibracao (b) (offset p_tx_eff = mediana(rssi + FSPL(dist)) so nos validos do treino) sao a
mesma sequencia de operacoes de `processar_celula` (que nao expoe os arrays); a validacao
(--validar) confere que o MAE resultante e igual, campo a campo, ao parcial gravado.

Blocos: a regra de blocos da particao e SpatialKFold._assign_groups(pos_km) com
grid_size_km = 10 (id = grid_x * max_y + grid_y); independe da semente. Cada no de teste
herda o id do seu bloco; S_B = soma das perdas |pred - rssi| e M_B = numero de nos pontuados
da populacao no bloco B, entre os nos de teste (apos a erosao pelo buffer).

Para R6, por sorteio, o MAE nos validos do teste dos preditores constantes
c in {-100, -110, -120, -130} dBm, da mediana dos validos do treino (recalculada por
sorteio) e, a titulo de referencia, da mediana do treino inteiro (a "constante" do 2.1).

Modos:
  --validar          lins_Q1 e bauru_Q1, 5 primeiros sorteios, contra
                     fase2/_v3_2.1_3.1_parcial_16x60rnd.json -> fase5/R5R6_validacao_laco.json
  --celula bauru_Q1  60 sorteios -> fase5/_parcial_laco/<celula>.json (retomavel)

Sem GPU (CUDA_VISIBLE_DEVICES vazio), sem treino. Nao interpreta resultado.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU apenas (a GPU esta com outro lote)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
SCRIPT_SHA256 = hashlib.sha256(HERE.read_bytes()).hexdigest()
BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics")
ORIG_PATH = BASE / "scripts" / "v3_2.1_3.1_deriva_calibracao.py"
ORIG_SHA256 = "9b9bcf91bab51621107a3d7978964d46dd16c5ec1a76a3789842ae1bce77f3bc"
V3 = BASE / "_v3_2026-09-25"
F2_JSON = V3 / "fase2" / "2.1_deriva_erro_baselines_16x60rnd.json"
PARCIAL_B2 = V3 / "fase2" / "_v3_2.1_3.1_parcial_16x60rnd.json"
OUT = V3 / "fase5"
PARCIAL_DIR = OUT / "_parcial_laco"
OUT_VALID = OUT / "R5R6_validacao_laco.json"
CONSTANTES_DBM = (-100.0, -110.0, -120.0, -130.0)
POPS = ("validos", "todos")

_orig_sha_real = hashlib.sha256(ORIG_PATH.read_bytes()).hexdigest()
if _orig_sha_real != ORIG_SHA256:
    raise SystemExit(f"ABORTA: sha do script original diverge ({_orig_sha_real})")

_spec = importlib.util.spec_from_file_location("deriva_orig", ORIG_PATH)
orig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(orig)  # main() protegido por __name__
assert orig.GRID_KM == 10.0 and orig.BUFFER_KM == 2.0 and orig.FRACS == (0.70, 0.15, 0.15) \
    and orig.FREQ_MHZ == 900.0, "configuracao g10b2 do original alterada"

import torch  # noqa: E402

torch.set_num_threads(1)


def agora() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[{agora()}] {msg}", flush=True)


def carregar_seeds() -> list:
    a = json.loads(F2_JSON.read_text(encoding="utf-8"))
    seeds = a["nota_divergencia_seeds"]["seeds_usados_nesta_rodada"]
    p = json.loads(PARCIAL_B2.read_text(encoding="utf-8"))
    assert len(seeds) == 60 and len(set(seeds)) == 60, "esperados 60 sorteios distintos"
    assert seeds == p["seeds_usados"], "sementes do JSON final != parcial de b=2"
    return [int(s) for s in seeds]


def carregar_celula(chave: str) -> dict:
    """Mesma sequencia de leitura de orig.processar_celula."""
    cidade, quad = chave.split("_")
    basename = f"transfer_dataset_{cidade}_v19_{quad}_enriched_cftudo.pt"
    tensor_path = orig.TENSOR_DIR / basename
    rf, modo = orig.carregar_tensor(tensor_path)
    ty = torch.as_tensor(rf["terrain"].y).float().clone().numpy()
    dist_all = torch.as_tensor(rf["terrain"].dist_nearest_m).float().clone().numpy()
    pos_deg = torch.as_tensor(rf["terrain"].pos).float().clone()
    del rf
    pos_m = orig.latlon_graus_para_metros(pos_deg)
    rssi = ty[:, 3].astype(np.float64)
    pl = ty[:, 0].astype(np.float64)
    sent = pl >= orig.PL_TARGET_MAX_VALID
    # blocos da particao (mesma regra de SpatialKFold._assign_groups, grid de 10 km)
    skf = orig.SpatialKFold(n_splits=3, buffer_km=orig.BUFFER_KM, grid_size_km=orig.GRID_KM, random_state=0)
    pos_km = pos_m / 1000.0
    gid = skf._assign_groups(pos_km)
    gx = (pos_km[:, 0] / orig.GRID_KM).astype(int)
    gy = (pos_km[:, 1] / orig.GRID_KM).astype(int)
    n_ocupados = int(np.unique(gid).size)
    n_bbox = int((gx.max() + 1) * (gy.max() + 1))
    return {"chave": chave, "modo_carga": modo, "tensor_path": str(tensor_path),
            "sha256_manifest": orig.carregar_manifest_sha(basename),
            "n_total": int(rssi.shape[0]), "pos_m": pos_m, "rssi": rssi, "dist": dist_all,
            "sent": sent, "gid": gid, "N_blocos_ocupados": n_ocupados, "N_blocos_bbox": n_bbox}


def somas_por_bloco(inv: np.ndarray, nb: int, ub: np.ndarray, loss: np.ndarray, mask: np.ndarray) -> dict:
    M = np.bincount(inv[mask], minlength=nb)
    S = np.bincount(inv[mask], weights=loss[mask], minlength=nb)
    ok = M > 0
    return {"blocos": [int(x) for x in ub[ok]], "S": [float(x) for x in S[ok]], "M": [int(x) for x in M[ok]]}


def sorteio(c: dict, seed: int) -> dict:
    parts = orig.split_espacial_3vias(c["pos_m"], orig.GRID_KM, orig.BUFFER_KM, orig.FRACS, seed)
    tr, te = parts["train"], parts["test"]
    if tr.size == 0 or te.size == 0:
        return {"split_seed": seed, "status": "erro_particao_vazia", "n_train": int(tr.size), "n_test": int(te.size)}
    rssi, dist, sent = c["rssi"], c["dist"], c["sent"]
    rssi_tr, rssi_te = rssi[tr], rssi[te]
    dist_tr, dist_te = dist[tr], dist[te]
    sent_tr, sent_te = sent[tr], sent[te]
    val_te = ~sent_te
    pop_te = {"validos": val_te, "todos": np.ones_like(sent_te, dtype=bool)}

    # --- preditores (mesmas operacoes de orig.processar_celula) ---
    constante = float(np.median(rssi_tr))
    fspl = orig.MODELOS["fspl"]
    mask_tr_valido = ~sent_tr
    if mask_tr_valido.sum() > 0:
        pl_pred_tr_v = fspl(dist_tr[mask_tr_valido])
        p_tx_b = float(np.median(rssi_tr[mask_tr_valido] + pl_pred_tr_v))
    else:
        p_tx_b = None
    loss = {"constante": np.abs(constante - rssi_te)}
    if p_tx_b is not None:
        rssi_bl_te = p_tx_b - fspl(dist_te)
        loss["fspl_b"] = np.abs(rssi_bl_te - rssi_te)

    ub, inv = np.unique(c["gid"][te], return_inverse=True)
    nb = int(ub.size)
    r5 = {}
    for pred, lv in loss.items():
        r5[pred] = {}
        for pop, m in pop_te.items():
            blk = somas_por_bloco(inv, nb, ub, lv, m)
            # MAE por orig.mae (identico ao do parcial gravado)
            if pred == "constante":
                mae_o = orig.mae(constante, rssi_te[m])
            else:
                mae_o = orig.mae(rssi_bl_te[m], rssi_te[m])
            r5[pred][pop] = {**blk, "mae_orig": mae_o}

    # --- R6: constantes no MAE dos validos do teste ---
    rv = rssi_te[val_te]
    r6 = {}
    for cval in CONSTANTES_DBM:
        r6[f"c{int(cval)}"] = {"c": cval, "mae_validos": orig.mae(cval, rv)}
    if mask_tr_valido.sum() > 0:
        cmv = float(np.median(rssi_tr[mask_tr_valido]))
        r6["mediana_validos_treino"] = {"c": cmv, "mae_validos": orig.mae(cmv, rv)}
    else:
        r6["mediana_validos_treino"] = {"c": None, "mae_validos": None}
    r6["mediana_treino_inteiro_ref"] = {"c": constante, "mae_validos": orig.mae(constante, rv)}

    return {"split_seed": seed, "status": "ok", "n_train": int(tr.size), "n_test": int(te.size),
            "n_pop_teste": {p: int(m.sum()) for p, m in pop_te.items()},
            "n_blocos_teste_com_no": nb, "constante_treino_mediana_rssi": constante,
            "offset_p_tx_eff_b_fspl": p_tx_b, "R5": r5, "R6": r6}


def rodar_celula(chave: str, seeds: list) -> dict:
    t0 = time.perf_counter()
    c = carregar_celula(chave)
    rec = {"celula": chave, "sha256_manifest": c["sha256_manifest"], "modo_carga": c["modo_carga"],
           "n_nodes_total": c["n_total"], "N_blocos_ocupados": c["N_blocos_ocupados"],
           "N_blocos_bbox": c["N_blocos_bbox"], "tempo_load_s": time.perf_counter() - t0}
    rec["por_sorteio"] = [sorteio(c, s) for s in seeds]
    rec["status"] = "ok"
    rec["tempo_total_s"] = time.perf_counter() - t0
    return rec


def deep_equal_nums(a, b, path, out):
    if a is None or b is None:
        if a != b:
            out["divergentes"].append((path, a, b))
        return
    out["n"] += 1
    d = abs(float(a) - float(b))
    out["max_abs"] = max(out["max_abs"], d)
    if a != b:
        out["divergentes"].append((path, a, b))


def validar():
    seeds = carregar_seeds()[:5]
    p = json.loads(PARCIAL_B2.read_text(encoding="utf-8"))
    saida_cel = {}
    ok_global = True
    for chave in ("lins_Q1", "bauru_Q1"):
        log(f"VALIDACAO {chave}, sorteios {seeds}")
        c = carregar_celula(chave)
        ref = p["celulas"][chave]["por_sorteio"][:5]
        ex = {"n": 0, "max_abs": 0.0, "divergentes": []}   # igualdade exata dos MAE de orig.mae
        soma = {"n": 0, "max_abs": 0.0}                    # Err = sum(S_B)/sum(M_B) vs gravado
        tabela = []
        for seed, r in zip(seeds, ref):
            s = sorteio(c, seed)
            assert s["split_seed"] == r["split_seed"] and s["status"] == "ok"
            assert s["n_test"] == r["n_test"] and s["n_train"] == r["n_train"]
            assert s["n_pop_teste"]["validos"] == r["n_pop_teste"]["validos"]
            assert s["n_pop_teste"]["todos"] == r["n_pop_teste"]["todos"]
            gravado = {
                ("constante", "validos"): r["mae_constante_teste"]["validos"],
                ("constante", "todos"): r["mae_constante_teste"]["todos"],
                ("fspl_b", "validos"): r["mae_modelo_b_validos_teste"]["fspl"]["validos"],
                ("fspl_b", "todos"): r["mae_modelo_b_validos_teste"]["fspl"]["todos"],
            }
            lin = {"split_seed": seed}
            for (pred, pop), g in gravado.items():
                x = s["R5"][pred][pop]
                deep_equal_nums(x["mae_orig"], g, f"{chave}/{seed}/{pred}/{pop}/mae_orig", ex)
                err_soma = sum(x["S"]) / sum(x["M"])
                soma["n"] += 1
                soma["max_abs"] = max(soma["max_abs"], abs(err_soma - g))
                lin[f"{pred}_{pop}"] = {"gravado": g, "laco_mae_orig": x["mae_orig"], "laco_soma_S_sobre_M": err_soma,
                                         "k_blocos": len(x["M"]), "M": int(sum(x["M"]))}
            # R6: c = -110 tem de igualar o constante gravado (mediana do treino = -110)
            deep_equal_nums(s["R6"]["c-110"]["mae_validos"], r["mae_constante_teste"]["validos"],
                            f"{chave}/{seed}/R6_c-110", ex)
            lin["constante_treino_mediana_rssi"] = s["constante_treino_mediana_rssi"]
            lin["R6_c-110_mae_validos"] = s["R6"]["c-110"]["mae_validos"]
            tabela.append(lin)
        reproduz = (len(ex["divergentes"]) == 0 and soma["max_abs"] < 1e-9)
        ok_global &= reproduz
        saida_cel[chave] = {"reproduz": bool(reproduz), "n_numeros_comparados_exato": ex["n"],
                            "max_abs_diferenca_mae_orig": ex["max_abs"], "divergentes": ex["divergentes"][:20],
                            "max_abs_diferenca_Err_soma_vs_gravado": soma["max_abs"],
                            "tolerancia_soma": 1e-9, "linhas": tabela}
    saida = {"id": "R5R6_validacao_laco", "sementes": seeds, "referencia": str(PARCIAL_B2),
             "referencia_sha256": orig.sha256_file(PARCIAL_B2),
             "criterio_de_aceite": "MAE do constante e do FSPL (b) nos validos e em todos os nos: igualdade exata "
                                   "(==) de orig.mae contra o parcial gravado; Err = sum S_B / sum M_B dentro de 1e-9; "
                                   "R6 c=-110 igual ao MAE do constante gravado",
             "reproduz": bool(ok_global), "celulas": saida_cel,
             "script_sha256": SCRIPT_SHA256, "script_original_sha256": ORIG_SHA256,
             "comando": " ".join(sys.argv), "data_utc": agora(), "venv": sys.executable}
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_VALID.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"validacao reproduz={ok_global} -> {OUT_VALID}")
    sys.exit(0 if ok_global else 2)


def rodar_uma(chave: str):
    PARCIAL_DIR.mkdir(parents=True, exist_ok=True)
    arq = PARCIAL_DIR / f"{chave}.json"
    if arq.exists() and json.loads(arq.read_text(encoding="utf-8")).get("status") == "ok":
        log(f"{chave}: parcial ok, pulando")
        return
    seeds = carregar_seeds()
    log(f"=== {chave}: {len(seeds)} sorteios ===")
    try:
        rec = rodar_celula(chave, seeds)
    except Exception as e:  # noqa: BLE001
        rec = {"celula": chave, "status": "erro_excecao", "erro": repr(e), "traceback": traceback.format_exc()}
    rec["script_sha256"] = SCRIPT_SHA256
    rec["data_utc"] = agora()
    arq.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"{chave}: status={rec['status']} -> {arq}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validar", action="store_true")
    g.add_argument("--celula", default="")
    a = ap.parse_args()
    if a.validar:
        validar()
    else:
        rodar_uma(a.celula)


if __name__ == "__main__":
    main()
