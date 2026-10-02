#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste G0 (d): confere, por execucao real das funcoes do script CONGELADO
(train_gnn_c0_spatial.py), se o split espacial de 3 vias e a selecao de
checkpoint realmente isolam o teste — nao ha troca de variavel train/val/test.

Carrega o proprio arquivo do script congelado via importlib (sem reescreve-lo),
roda split_espacial_3vias + verificar_split sobre pontos SINTETICOS pequenos,
e testa a guarda _selecionar_melhor_epoca (deve levantar AssertionError se
receber uma chave de teste).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

FROZEN = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/gnn_rf_ieee_access/"
               "FIRST_RESPONSE_REVIEW_IEEE_ACESSES/EVIDENCIA_RESUBMISSAO/scripts/"
               "train_gnn_c0_spatial.py")
BASE_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2")
sys.path.append(str(BASE_DIR / "03_training"))
from spatial_cv import SpatialKFold, validate_spatial_cv  # noqa: E402

spec = importlib.util.spec_from_file_location("train_gnn_c0_spatial", FROZEN)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

OUT = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/_v3_2026-09-25/gpu/G0")

rng = np.random.RandomState(0)
n = 20000
# grade 30x30 km com pontos aleatorios, igual espirito do split real
pos_m = rng.uniform(0, 30000, size=(n, 2)).astype(np.float64)


def log(msg):
    print(msg)


parts, info = mod.split_espacial_3vias(pos_m, grid_km=5.0, buffer_km=2.0,
                                        fracs=(0.70, 0.15, 0.15), split_seed=42,
                                        SpatialKFold=SpatialKFold, log=log)
verif = mod.verificar_split(parts, pos_m, buffer_km=2.0,
                             validate_spatial_cv=validate_spatial_cv, log=log)

# 1) nenhuma intersecao entre particoes
sem_intersecao = all(v == 0 for v in verif["intersecoes"].values())
# 2) buffer respeitado nas 3 comparacoes par-a-par
buffer_ok = all(v["respeita_buffer"] for v in verif["pares"].values())
# 3) sha256 dos indices de teste != sha256 dos indices de treino/val (prova de
#    que "test" nao e um alias acidental de outra particao)
sha = {k: mod.sha256_idx(v) for k, v in parts.items()}
sha_distintos = len(set(sha.values())) == 3

# 4) guarda _selecionar_melhor_epoca: deve FALHAR (AssertionError) se receber
#    um dict que contem chave 'test' (prova de que a seguranca contra
#    selecionar checkpoint pelo teste esta ativa no proprio codigo)
guarda_disparou = False
try:
    mod._selecionar_melhor_epoca({"mae_rssi_db": 1.0, "test": {"mae_rssi_db": 0.5}})
except AssertionError:
    guarda_disparou = True

# 5) selecao com dict normal (sem 'test') funciona e retorna o valor de val
selecao_normal_ok = mod._selecionar_melhor_epoca({"mae_rssi_db": 3.21}) == 3.21

passa = bool(sem_intersecao and buffer_ok and sha_distintos and guarda_disparou
             and selecao_normal_ok and verif["ok"])

resultado = {
    "teste": "metrica_no_split_correto (d)",
    "comando": ("/trabalho/ambientes/s33_amb_virtual/.venv/bin/python "
                "teste_d_split_correto.py"),
    "n_pontos_sinteticos": n,
    "particoes_n": {k: int(len(v)) for k, v in parts.items()},
    "intersecoes": verif["intersecoes"],
    "sem_intersecao": sem_intersecao,
    "buffer_respeitado_por_par": {k: v["respeita_buffer"] for k, v in verif["pares"].items()},
    "buffer_ok": buffer_ok,
    "sha256_por_particao": sha,
    "sha_distintos": sha_distintos,
    "guarda_selecionar_melhor_epoca_disparou_com_chave_test": guarda_disparou,
    "selecao_normal_retornou_valor_correto": selecao_normal_ok,
    "verificacao_ok_geral": verif["ok"],
    "passa": passa,
}
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "teste_d_split_correto.json", "w", encoding="utf-8") as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)
print(json.dumps(resultado, indent=2, ensure_ascii=False))
