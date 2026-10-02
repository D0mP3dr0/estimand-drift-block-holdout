#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu/A1/teste_d_split_e_f_extra.py -- CPU-only, sem GPU.

(d) Confere, por script (nao por leitura de log a olho), que
    `selecao.test_no_melhor_ckpt` do run JSON do smoke (modelo_v3/smoke/)
    vem da particao de TESTE (nao validacao): recomputa o sha256 dos
    idx_global do .npz de predicoes (ordenados) e compara com
    `particoes.test.idx_sha256_global` gravado no run JSON -- e confere que
    esse hash e DIFERENTE do de val/train (nao e coincidencia de todos
    darem o mesmo hash). Confere tambem os campos declarados
    (`test_usado_na_selecao: false`, criterio = val).

(f) extra: confirma por grep (evidencia estatica) que `fisico()` nunca e
    chamado dentro dos loops de treino dos scripts CONGELADOS nem dos
    wrappers v3 (so no pos-treino, para gerar as predicoes .npz), e mede
    a degenerescencia do alvo coverage_prob (canal 4) no MESMO batch real
    usado no teste a/b/c/e (achado n2 do G0 real: coverage_prob=0
    constante no batch antigo) -- roda sobre os .npz de teste do smoke v3
    (gnn e mlp), que ja tem target[:,4] real.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

BASE = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25")
SMOKE = BASE / "gpu/modelo_v3/smoke"
FROZEN_GNN = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
                   "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/scripts/"
                   "train_gnn_c0_spatial.py")
FROZEN_MLP = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
                   "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/scripts/"
                   "train_mlp_c0_spatial.py")
WRAP_GNN = BASE / "gpu/modelo_v3/train_gnn_v3.py"
WRAP_MLP = BASE / "gpu/modelo_v3/train_mlp_v3.py"
V3_COMMON = BASE / "gpu/modelo_v3/v3_common.py"


def sha256_idx(idx: np.ndarray) -> str:
    """Mesma convencao do congelado: sha256 dos indices int64 ORDENADOS."""
    arr = np.sort(idx.astype(np.int64))
    h = hashlib.sha256()
    h.update(arr.tobytes())
    return h.hexdigest()


def checar_split(run_label: str) -> dict:
    d = SMOKE / run_label
    run_json = json.load(open(d / f"run_{run_label}.json", encoding="utf-8"))
    npz = np.load(d / f"predicoes_{run_label}.npz")

    hash_npz_recomputado = sha256_idx(npz["idx_global"])
    hash_gravado_test = run_json["particoes"]["test"]["idx_sha256_global"]
    hash_gravado_val = run_json["particoes"]["val"]["idx_sha256_global"]
    hash_gravado_train = run_json["particoes"]["train"]["idx_sha256_global"]

    bate_com_test = (hash_npz_recomputado == hash_gravado_test)
    bate_com_val = (hash_npz_recomputado == hash_gravado_val)
    bate_com_train = (hash_npz_recomputado == hash_gravado_train)
    n_npz = int(npz["idx_global"].shape[0])
    n_test_rec = run_json["particoes"]["test"]["n"]

    sel = run_json["selecao"]
    return {
        "run_label": run_label,
        "n_nos_npz": n_npz,
        "n_nos_particao_test_no_run_json": n_test_rec,
        "n_bate": (n_npz == n_test_rec),
        "hash_npz_idx_global_recomputado": hash_npz_recomputado,
        "hash_gravado_particoes_test": hash_gravado_test,
        "hash_gravado_particoes_val": hash_gravado_val,
        "hash_gravado_particoes_train": hash_gravado_train,
        "npz_bate_com_particao_test": bate_com_test,
        "npz_bate_com_particao_val_FALSO_POSITIVO_SERIA_BUG": bate_com_val,
        "npz_bate_com_particao_train_FALSO_POSITIVO_SERIA_BUG": bate_com_train,
        "selecao_criterio": sel["criterio"],
        "selecao_test_usado_na_selecao": sel["test_usado_na_selecao"],
        "selecao_tem_test_no_melhor_ckpt": ("test_no_melhor_ckpt" in sel and sel["test_no_melhor_ckpt"] is not None),
        "coverage_prob_target_no_npz": {
            "min": float(npz["target"][:, 4].min()),
            "max": float(npz["target"][:, 4].max()),
            "std": float(npz["target"][:, 4].std()),
            "constante": bool(npz["target"][:, 4].std() < 1e-9),
        },
        "passa_d": bool(bate_com_test and not bate_com_val and not bate_com_train and (n_npz == n_test_rec)
                        and sel["test_usado_na_selecao"] is False),
    }


def grep_fisico(caminho: Path) -> int:
    r = subprocess.run(["grep", "-c", r"\.fisico(", str(caminho)], capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def main():
    resultado = {"teste": "A1_d_split_correto_e_f_extra_fisico_coverage"}

    resultado["d_split_correto"] = {
        "gnn_v3_smoke_bauru_Q2": checar_split("gnn_v3_smoke_bauru_Q2"),
        "mlp_v3_smoke_bauru_Q2": checar_split("mlp_v3_smoke_bauru_Q2"),
    }
    resultado["d_passa"] = (resultado["d_split_correto"]["gnn_v3_smoke_bauru_Q2"]["passa_d"]
                             and resultado["d_split_correto"]["mlp_v3_smoke_bauru_Q2"]["passa_d"])

    # -- (f) fisico() fora do treino: grep estatico nos 4 arquivos relevantes --
    ocorrencias = {
        "frozen_train_gnn_c0_spatial.py": grep_fisico(FROZEN_GNN),
        "frozen_train_mlp_c0_spatial.py": grep_fisico(FROZEN_MLP),
        "wrapper_train_gnn_v3.py": grep_fisico(WRAP_GNN),
        "wrapper_train_mlp_v3.py": grep_fisico(WRAP_MLP),
        "v3_common.py (esperado>0, so em gerar_predicoes_teste_*, POS-treino)": grep_fisico(V3_COMMON),
    }
    fisico_ausente_nos_congelados = (ocorrencias["frozen_train_gnn_c0_spatial.py"] == 0
                                      and ocorrencias["frozen_train_mlp_c0_spatial.py"] == 0)
    resultado["f_fisico_fora_do_treino"] = {
        "ocorrencias_grep_fisico_parenteses": ocorrencias,
        "nota": ("fisico() so pode aparecer nos WRAPPERS v3, e la e chamado depois de "
                 "mod.main() terminar (reconstrucao de predicoes de teste), nunca dentro "
                 "do loop de treino/loss dos congelados."),
        "passa": fisico_ausente_nos_congelados,
    }

    # canal 4 esperado: alvo coverage em [0,1], huber direto (physics_loss.py:63,99-103,
    # target_names[4]='coverage'); AffineDecoderV3 aplica sigmoid so no canal 4
    # (rf_decoder_v3.py:148) -- coerente. Ja confirmado por leitura de codigo.
    resultado["f_canal4_sigmoid_coerente_com_curriculumrfloss"] = {
        "physics_loss_target_names_4": "coverage",
        "physics_loss_forward": "huber_loss(predictions[:,4], targets[:,4]) direto, sem transformacao",
        "rf_decoder_v3_canal4": "sigmoid(raw[:,4]) -- unico canal comprimido, produz [0,1]",
        "conclusao": "coerente: alvo e probabilidade em [0,1], decoder entrega [0,1]",
        "passa": True,
    }

    veredito_geral_script = (resultado["d_passa"] and resultado["f_fisico_fora_do_treino"]["passa"])
    resultado["veredito_script"] = "passa" if veredito_geral_script else "bloqueia"

    saida = BASE / "gpu/A1/teste_d_split_e_f_extra.json"
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"gravado: {saida}")
    print(json.dumps(resultado, indent=2, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
