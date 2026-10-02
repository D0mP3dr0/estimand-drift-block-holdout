#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
matematica-estatistica-do-claim -- teste 3.2 (bordas reais da banda de P7,
prop:median, alegacao A5), roadmap v3, fio 2026-09-25_gnn_rf_artigo2_v3_fase1_cpu.

Pergunta (criterio_3.2.json): para as 16 celulas (seed 42), quais sao as
bordas Q1-(1-1/(2*pi_bar)) e Q1+(1/(2*pi_bar)) da banda de P7 calculadas
sobre os nos SENTINELA REAIS de treino (nao sobre distancia simulada
U[1,140] km), e onde cai a mediana calibrada (p_tx_eff) dentro dessa banda?

W = RSSI observado (terrain.y[:,3]) + PL_modelo(dist_nearest_m), no a no.
F1 = fda empirica de W restrita aos nos sentinela (y[:,0] >= 299) da
particao de TREINO de cada celula (split_espacial_3vias, split_seed=42,
grid_km=10, buffer_km=2, fracs 0.70/0.15/0.15 -- mesma logica congelada de
train_gnn_c0_spatial.py:412-472, reusada por IMPORT do script irmao
v3_2.1_3.1_deriva_calibracao.py, sem copia nem edicao).
pi_bar = fracao sentinela REAL do treino (recalculada do dado bruto, nao
o campo publicado no run JSON).

Mediana calibrada de referencia: baselines_analiticos.train.p_tx_eff_dbm
do run JSON congelado (definicao: train_gnn_c0_spatial.py:711,
ptx = median(rssi_tgt + pl) SO no treino) -- e TAMBEM recomputada aqui de
forma independente (mediana de W sobre TODOS os nos de treino, sentinela+
validos) como checagem cruzada contra o numero publicado.

Dado bruto: tensores leves *_enriched_cftudo.pt (~1,7 GB cada, mmap) em
/trabalho/TOPO_RF_DOWNLOAD_DRIVE/graph_data_v3/ (16/16 celulas presentes
localmente -- ao contrario do que a varredura_split_geometria.py registrou
para outra pasta, aqui a pasta v3 tem as 16).

Uso:
  .venv/bin/python v3_3.2_banda_p7_bordas_reais.py --out <json>
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
              "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/dados/treinos_c1")
SIBLING_SCRIPT = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/scripts/"
                       "v3_2.1_3.1_deriva_calibracao.py")

OUT_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
OUT_JSON = OUT_DIR / "fase3" / "3.2_banda_p7_bordas_reais.json"
CRIT_PATH = OUT_DIR / "criterios" / "criterio_3.2.json"

CIDADES = ["bauru", "campinas", "lins", "sorocaba"]
QUADRANTES = ["Q1", "Q2", "Q3", "Q4"]
PL_TARGET_MAX_VALID = 299.0
FREQ_MHZ = 900.0
GRID_KM = 10.0
BUFFER_KM = 2.0
FRACS = (0.70, 0.15, 0.15)
SPLIT_SEED = 42
LARGURA_MIN_DB = 1.0
N_CELULAS_MIN_LARGURA = 12

# offsets publicados de Lins Q1 (Folha/parecer) e bordas simuladas do fisico
# (comparacao final, nao entram no calculo)
OFFSETS_PUBLICADOS_LINS_Q1 = {"fspl": 20.402631103093142,
                              "hata_rural": 55.22485026374653,
                              "cost231_sub": 83.33030450186465}
BORDAS_SIMULADAS_FISICO_DBM = {
    "pi_0.8": {"inferior": 16.03, "superior": 20.40},
    "pi_0.9146": {"inferior": 17.65, "superior": 19.25},
}


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


# --- reuso por import do script irmao (sem copia, sem edicao) -------------
_spec = importlib.util.spec_from_file_location("v3_2_1_3_1", SIBLING_SCRIPT)
_sib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sib)
split_espacial_3vias = _sib.split_espacial_3vias
latlon_graus_para_metros = _sib.latlon_graus_para_metros
MODELOS = _sib.MODELOS
carregar_tensor = _sib.carregar_tensor
SIBLING_SHA256 = hashlib.sha256(SIBLING_SCRIPT.read_bytes()).hexdigest()


def processar_celula(cidade: str, quad: str) -> dict:
    basename = f"transfer_dataset_{cidade}_v19_{quad}_enriched_cftudo.pt"
    tensor_path = TENSOR_DIR / basename
    run_path = RUNDIR / f"run_c0c1cf_{cidade}_s42_{quad}_g10b2.json"

    rec = {"cidade": cidade, "quadrante": quad, "celula": f"{cidade}_{quad}",
           "tensor_path": str(tensor_path), "run_path": str(run_path),
           "timestamp_utc_inicio": datetime.now(timezone.utc).isoformat()}

    if not tensor_path.exists():
        rec["status"] = "nao_verificado"
        rec["motivo"] = "tensor _enriched_cftudo.pt ausente localmente"
        return rec
    if not run_path.exists():
        rec["status"] = "nao_verificado"
        rec["motivo"] = "run JSON ausente"
        return rec

    run_rec = json.loads(run_path.read_text(encoding="utf-8"))
    offsets_publicados = run_rec.get("baselines_analiticos", {}).get("train", {}).get("p_tx_eff_dbm", {})
    pi_publicado = 1.0 - run_rec.get("particoes", {}).get("train", {}).get("frac_pl_alvo_valido", None) \
        if run_rec.get("particoes", {}).get("train", {}).get("frac_pl_alvo_valido") is not None else None

    t1 = time.perf_counter()
    rf, modo_carga = carregar_tensor(tensor_path)
    ty = torch.as_tensor(rf["terrain"].y).float().clone().numpy()
    dist_all = torch.as_tensor(rf["terrain"].dist_nearest_m).float().clone().numpy()
    pos_deg = torch.as_tensor(rf["terrain"].pos).float().clone()
    n_total = int(ty.shape[0])
    del rf
    gc.collect()
    rec["modo_carga"] = modo_carga
    rec["tempo_load_s"] = time.perf_counter() - t1
    rec["n_nodes_total"] = n_total

    pos_m = latlon_graus_para_metros(pos_deg)
    rssi = ty[:, 3].astype(np.float64)
    pl = ty[:, 0].astype(np.float64)
    sentinela_all = pl >= PL_TARGET_MAX_VALID

    parts = split_espacial_3vias(pos_m, GRID_KM, BUFFER_KM, FRACS, SPLIT_SEED)
    tr = parts["train"]
    if tr.size == 0:
        rec["status"] = "nao_verificado"
        rec["motivo"] = "particao de treino vazia neste split_seed"
        return rec

    rssi_tr = rssi[tr]
    dist_tr = dist_all[tr]
    sent_tr = sentinela_all[tr]
    n_tr = int(tr.size)
    n_sent_tr = int(sent_tr.sum())
    pi_bar_real = n_sent_tr / n_tr

    rec["n_train"] = n_tr
    rec["n_sentinela_train"] = n_sent_tr
    rec["pi_bar_real_treino"] = pi_bar_real
    rec["pi_bar_publicado_1_menos_frac_valido"] = pi_publicado

    banda_nao_trivial = pi_bar_real > 0.5
    rec["banda_nao_trivial_pi_bar_gt_0_5"] = bool(banda_nao_trivial)

    modelos_out = {}
    if banda_nao_trivial:
        alpha = 1.0 - 1.0 / (2.0 * pi_bar_real)
        beta = 1.0 / (2.0 * pi_bar_real)
        rec["nivel_alpha_Q1_menos"] = alpha
        rec["nivel_beta_Q1_mais"] = beta
        rec["largura_em_nivel_1_menos_pi_sobre_pi"] = (1.0 - pi_bar_real) / pi_bar_real

        for nome, fmodel in MODELOS.items():
            pl_pred_tr_all = fmodel(dist_tr)
            w_all = rssi_tr + pl_pred_tr_all
            mediana_recomputada = float(np.median(w_all))

            w_sent = w_all[sent_tr]
            q_minus = float(np.quantile(w_sent, alpha, method="lower"))
            q_plus = float(np.quantile(w_sent, beta, method="higher"))
            largura_db = q_plus - q_minus
            dentro = q_minus <= mediana_recomputada <= q_plus
            posicao_relativa = ((mediana_recomputada - q_minus) / largura_db
                                if largura_db > 0 else None)
            offset_pub = offsets_publicados.get(nome)
            divergencia_recomputo_vs_publicado = (
                abs(mediana_recomputada - offset_pub) if offset_pub is not None else None)

            modelos_out[nome] = {
                "Q1_menos_dBm": q_minus,
                "Q1_mais_dBm": q_plus,
                "largura_banda_dB": largura_db,
                "largura_ge_1dB": bool(largura_db >= LARGURA_MIN_DB),
                "mediana_calibrada_recomputada_dBm": mediana_recomputada,
                "mediana_calibrada_publicada_dBm": offset_pub,
                "divergencia_recomputo_vs_publicado_dB": divergencia_recomputo_vs_publicado,
                "mediana_dentro_da_banda": bool(dentro),
                "posicao_relativa_0_a_1": posicao_relativa,
                "n_sentinela_usados_F1": int(w_sent.size),
            }
    else:
        rec["nota"] = "pi_bar_real <= 0.5: banda teorica vazia de conteudo (Q1- = -inf ou Q1+ = +inf)"

    rec["modelos"] = modelos_out
    rec["status"] = "ok"
    del ty, rssi, pl, dist_all, pos_deg, pos_m, sentinela_all
    gc.collect()
    rec["timestamp_utc_fim"] = datetime.now(timezone.utc).isoformat()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_inicio = time.perf_counter()
    celulas = {}
    for cidade in CIDADES:
        for quad in QUADRANTES:
            chave = f"{cidade}_{quad}"
            log(f"processando {chave}...")
            try:
                celulas[chave] = processar_celula(cidade, quad)
            except Exception as e:
                celulas[chave] = {"cidade": cidade, "quadrante": quad, "status": "erro",
                                   "erro": repr(e), "traceback": traceback.format_exc()}
            log(f"  {chave}: status={celulas[chave].get('status')}")

    # ---- resumo e veredito ----
    ok_cells = {k: v for k, v in celulas.items() if v.get("status") == "ok" and v.get("modelos")}
    todas_medianas_dentro = True
    violacoes = []
    larguras_ge_1db_por_modelo = {"fspl": 0, "hata_rural": 0, "cost231_sub": 0}
    n_celulas_avaliadas_por_modelo = {"fspl": 0, "hata_rural": 0, "cost231_sub": 0}
    largura_media_por_modelo = {"fspl": [], "hata_rural": [], "cost231_sub": []}
    divergencia_max_recomputo = 0.0

    for chave, rec in ok_cells.items():
        for nome, m in rec["modelos"].items():
            n_celulas_avaliadas_por_modelo[nome] += 1
            largura_media_por_modelo[nome].append(m["largura_banda_dB"])
            if m["largura_ge_1dB"]:
                larguras_ge_1db_por_modelo[nome] += 1
            if not m["mediana_dentro_da_banda"]:
                todas_medianas_dentro = False
                violacoes.append({"celula": chave, "modelo": nome})
            if m["divergencia_recomputo_vs_publicado_dB"] is not None:
                divergencia_max_recomputo = max(divergencia_max_recomputo,
                                                m["divergencia_recomputo_vs_publicado_dB"])

    largura_media_por_modelo = {k: (float(np.mean(v)) if v else None)
                                for k, v in largura_media_por_modelo.items()}

    n_cells_ge12_algum_modelo = any(v >= N_CELULAS_MIN_LARGURA
                                    for v in larguras_ge_1db_por_modelo.values())

    resumo = {
        "n_celulas_ok": len(ok_cells),
        "n_celulas_nao_verificadas": sum(1 for v in celulas.values() if v.get("status") == "nao_verificado"),
        "celulas_nao_verificadas": [k for k, v in celulas.items() if v.get("status") == "nao_verificado"],
        "todas_medianas_dentro_da_banda": todas_medianas_dentro,
        "violacoes_teorema": violacoes,
        "divergencia_maxima_recomputo_vs_publicado_dB": divergencia_max_recomputo,
        "n_celulas_com_largura_ge_1dB_por_modelo": larguras_ge_1db_por_modelo,
        "largura_media_banda_dB_por_modelo": largura_media_por_modelo,
        "criterio_alegacao_publicavel_atingido": bool(n_cells_ge12_algum_modelo),
        "lins_Q1_offsets_publicados_dBm": OFFSETS_PUBLICADOS_LINS_Q1,
        "lins_Q1_bordas_reais": ok_cells.get("lins_Q1", {}).get("modelos"),
        "bordas_simuladas_fisico_dBm_comparacao": BORDAS_SIMULADAS_FISICO_DBM,
    }

    if not todas_medianas_dentro:
        veredito = ("BUG: mediana calibrada fora da banda teorica em >=1 celula/modelo -- "
                     "parar e corrigir a implementacao antes de qualquer alegacao.")
        gravidade = "bloqueia"
    elif not n_cells_ge12_algum_modelo:
        veredito = ("Teorema confirmado (100% das medianas dentro da banda) mas o criterio de "
                    "publicabilidade da alegacao 'nao identificavel' (largura real >= 1 dB em "
                    f">= {N_CELULAS_MIN_LARGURA} celulas) NAO foi atingido para nenhum modelo "
                    "com os dados disponiveis -- rebaixar a alegacao A5 ou restringir aos modelos/"
                    "celulas que atingem o limiar.")
        gravidade = "alerta"
    else:
        veredito = ("Teorema confirmado e alegacao A5 publicavel com numero: largura real >= 1 dB "
                    f"em >= {N_CELULAS_MIN_LARGURA} celulas para pelo menos um modelo.")
        gravidade = "info"

    saida = {
        "pergunta": "Bordas Q1- e Q1+ reais da banda de P7 sobre sentinelas de treino, 16 celulas, seed 42",
        "alegacao": "A5 (P7): calibracao pela mediana nao identifica o offset",
        "criterio": json.loads(CRIT_PATH.read_text(encoding="utf-8")),
        "metodo": ("W = RSSI(y[:,3]) + PL_modelo(dist_nearest_m) no-a-no; F1 = fda empirica de W "
                  "sobre sentinelas (y[:,0]>=299) do treino (split_espacial_3vias, split_seed=42, "
                  "grid_km=10, buffer_km=2, fracs 0.70/0.15/0.15, logica congelada de "
                  "train_gnn_c0_spatial.py:412-472, reusada por import de v3_2.1_3.1_deriva_calibracao.py "
                  f"sha256={SIBLING_SHA256}); pi_bar = fracao sentinela real do treino; "
                  "Q1- = quantil(F1, 1-1/(2*pi_bar), lado inferior), Q1+ = quantil(F1, 1/(2*pi_bar), "
                  "lado superior); mediana calibrada recomputada de forma independente = "
                  "median(RSSI+PL_modelo) sobre TODO o treino (sentinela+validos), comparada com "
                  "baselines_analiticos.train.p_tx_eff_dbm do run JSON (definicao em "
                  "train_gnn_c0_spatial.py:711)."),
        "insumos": {
            "tensores": f"{TENSOR_DIR} (mmap, *_enriched_cftudo.pt, campos terrain.y, terrain.pos, "
                       "terrain.dist_nearest_m)",
            "run_json_dir": str(RUNDIR),
            "script_reusado": {"caminho": str(SIBLING_SCRIPT), "sha256": SIBLING_SHA256},
        },
        "por_celula": celulas,
        "resumo": resumo,
        "veredito_vs_criterio": veredito,
        "gravidade": gravidade,
        "nao_verificado": [k for k, v in celulas.items() if v.get("status") == "nao_verificado"],
        "comando_rodado": f"{sys.executable} {SCRIPT_PATH} --out {out_path}",
        "tempo_total_s": time.perf_counter() - t_inicio,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=1)
    log(f"gravado {out_path} ({time.perf_counter()-t_inicio:.1f}s total)")
    log(f"veredito: {veredito}")


if __name__ == "__main__":
    main()
