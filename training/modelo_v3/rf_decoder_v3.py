#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu/modelo_v3/rf_decoder_v3.py

Decodificador V3 -- PLANO A da fila de GPU (decisao do dono D7, 25/09/2026,
`/trabalho/HERMES/AGENTES/_DECISOES/DECISOES-2026-09-25-artigo2-v3-independencia-e-roadmap.md`):
saida AFIM, sem softplus/sigmoid/clamp no caminho da PERDA, para eliminar a
saturacao de ~100% das amostras (canais 0-2 e 4) encontrada com dado real
em `gpu/G0/ia-bug-silencioso_real.json` (achado a2b), que explica o plateau
do overfit de 1 batch (razao loss_final/loss_inicial = 0.132, criterio de
passagem <= 0.10).

NAO edita `GNN_RF_V2/02_models/rf_decoder.py` (READ-ONLY, lei inviolavel).
Importa so a classe BASE `RFDecoder` de la; nunca importa nem reusa
`PhysicsConstrainedDecoder`.

Unidades e colunas -- confirmadas por leitura de codigo, nao de memoria:
`GNN_RF_V2/02_models/physics_loss.py:58-64` (`PhysicsRFLoss.target_names`)
e `:71-139` (`PhysicsRFLoss.forward`, herdado sem alteracao por
`CurriculumRFLoss`):
    0: path_loss_total      (dB)   clamp de producao [0, 200]
    1: path_loss_vegetation (dB)   clamp de producao [0, 50]
    2: path_loss_terrain    (dB)   clamp de producao [0, 30]
    3: rssi                 (dBm)  clamp de producao [-150, 0]
    4: coverage_prob        (adimensional, 0-1)

`PhysicsRFLoss.forward` (physics_loss.py:93-103) faz
`huber_loss(predictions[:, i], targets[:, i])` COLUNA A COLUNA, direto,
sem nenhuma renormalizacao interna -- o decodificador tem que entregar
o valor JA na mesma unidade do alvo (dB/dBm), exatamente como o
`PhysicsConstrainedDecoder` original fazia (mas sem a compressao que
satura). O MAE de relato do script congelado
(`train_gnn_c0_spatial.py`, `metricas_particao()`) usa a coluna 3 (rssi,
`mae_rssi_db`) e a coluna 0 (path_loss_total, `mae_pl_db`) -- nada disso
muda aqui; a perda em si (CurriculumRFLoss) nao e tocada por este arquivo.

Mapeamento AFIM -- escalas/offsets FIXOS e declarados, centrados no meio
de cada clamp de producao, span = metade do clamp (sem nenhuma
nao-linearidade de compressao: a funcao e linear em toda a reta real,
entao o gradiente nunca satura por construcao):
    canal 0 (path_loss_total)      = raw*100 + 100   (centro do clamp [0,200])
    canal 1 (path_loss_vegetation) = raw*25  + 25     (centro do clamp [0,50])
    canal 2 (path_loss_terrain)    = raw*15  + 15     (centro do clamp [0,30])
    canal 3 (rssi)                 = raw*75  - 75     (centro do clamp [-150,0])
    canal 4 (coverage_prob)        = sigmoid(raw)     -- ver nota abaixo

Nota sobre o canal 4 (UNICA excecao afim deste decodificador, documentada
aqui, nao e regra geral): a perda de producao compara
`predictions[:, 4]` com `targets[:, 4]` por huber_loss DIRETO, e o alvo de
cobertura e uma probabilidade em [0, 1] (na amostra real auditada em
`ia-bug-silencioso_real.json`, achado `n2`, `coverage_prob` e constante
0.0 no batch amostrado -- ja sinalizado para a frente de dados). Deixar o
canal 4 puramente afim e ilimitado compararia um logit contra um alvo em
[0, 1], o que e mecanicamente computavel mas fisicamente sem sentido; por
isso este e o UNICO canal em que se aplica sigmoid -- e o MESMO sigmoid
que `RFDecoder.forward` original ja aplicava no canal 4
(`rf_decoder.py:85-89`) antes de qualquer constraint fisico, decisao
preservada, nao nova. Os canais 0-3 nunca passam por nenhuma nao-linearidade
de compressao.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

# --------------------------------------------------------------------------
# Importa a classe BASE do arquivo congelado (READ-ONLY). O proprio
# rf_decoder.py insere `sys.path.insert(0, parent_de_02_models)` para achar
# `config`; replicamos so o `sys.path` necessario para importar o MODULO
# `rf_decoder` por nome (sem copiar nenhuma linha de codigo dele).
# --------------------------------------------------------------------------
_GNN_RF_V2_02_MODELS = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF_V2/02_models")
if str(_GNN_RF_V2_02_MODELS) not in sys.path:
    sys.path.insert(0, str(_GNN_RF_V2_02_MODELS))

from rf_decoder import RFDecoder  # noqa: E402 (import tardio, sys.path acima)


# Escalas/offsets FIXOS (nao aprendidos), documentados no docstring do modulo.
_ESCALA_OFFSET_PL = (
    (100.0, 100.0),   # canal 0: path_loss_total,      clamp producao [0, 200]
    (25.0, 25.0),     # canal 1: path_loss_vegetation, clamp producao [0, 50]
    (15.0, 15.0),     # canal 2: path_loss_terrain,    clamp producao [0, 30]
)
_ESCALA_RSSI, _OFFSET_RSSI = 75.0, -75.0   # canal 3, clamp producao [-150, 0]

CLAMP_PL = ((0.0, 200.0), (0.0, 50.0), (0.0, 30.0))
CLAMP_RSSI = (-150.0, 0.0)
CLAMP_COVERAGE = (0.0, 1.0)


class AffineDecoderV3(RFDecoder):
    """
    Decodificador afim (PLANO A / D7): substitui softplus+clamp (canais
    0-2) e sigmoid escalado (canal 3) do `PhysicsConstrainedDecoder` por
    uma transformacao AFIM sem compressao. Os clamps de producao SO entram
    em `fisico()` (inferencia/relato); o caminho de treino/perda nunca ve
    um clamp ou uma saturacao.

    Herda de `RFDecoder` (nunca de `PhysicsConstrainedDecoder`) -- reusa
    so a espinha dorsal `self.layers` (MLP compartilhado) e os 3 heads
    mortos (`path_loss_head`/`rssi_head`/`coverage_head`), preservados SEM
    uso (nunca entram no forward), exatamente como no original -- ver
    achado `a2_gradiente_cabecas_mortas_real` do gate `ia-bug-silencioso`
    (`gpu/G0/ia-bug-silencioso_real.json`). Nao os removemos aqui: nao e
    escopo do decodificador mudar a contagem de parametros reportada, e a
    paridade arquitetural com o original importa para A2/A3 compararem
    GNN e MLP sob a mesma regua.
    """

    def __init__(self, *args, **kwargs):
        # min_path_loss/max_path_loss/min_rssi/max_rssi: aceitos so por
        # compatibilidade de assinatura com PhysicsConstrainedDecoder (o
        # monkeypatch em train_*_v3.py substitui a CLASSE inteira; algumas
        # chamadas podem repassar esses kwargs). Usados so para popular os
        # limites de `fisico()` quando informados explicitamente; o default
        # e sempre o clamp de producao documentado acima.
        min_path_loss = kwargs.pop("min_path_loss", CLAMP_PL[0][0])
        max_path_loss = kwargs.pop("max_path_loss", CLAMP_PL[0][1])
        min_rssi = kwargs.pop("min_rssi", CLAMP_RSSI[0])
        max_rssi = kwargs.pop("max_rssi", CLAMP_RSSI[1])
        super().__init__(*args, **kwargs)
        self.min_path_loss = min_path_loss
        self.max_path_loss = max_path_loss
        self.min_rssi = min_rssi
        self.max_rssi = max_rssi

    def forward(self, terrain_embeddings: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward de TREINO: saida afim nos canais 0-3 (sem nenhuma
        nao-linearidade de compressao), sigmoid so no canal 4 (ver nota do
        modulo). Chama `self.layers` DIRETO -- nao chama
        `RFDecoder.forward` nem `PhysicsConstrainedDecoder.forward` -- para
        nao herdar o sigmoid que `RFDecoder.forward` ja aplicava no canal 4
        antes de qualquer constraint (rf_decoder.py:85-89); aqui o sigmoid
        do canal 4 e aplicado UMA unica vez, explicitamente, abaixo.
        """
        raw = self.layers(terrain_embeddings)  # [N, 5], pre-ativacao, sem compressao

        out = torch.empty_like(raw)
        for canal, (escala, offset) in enumerate(_ESCALA_OFFSET_PL):
            out[:, canal] = raw[:, canal] * escala + offset
        out[:, 3] = raw[:, 3] * _ESCALA_RSSI + _OFFSET_RSSI
        out[:, 4] = torch.sigmoid(raw[:, 4])  # unico canal comprimido; ver nota do modulo

        return out

    def fisico(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Aplica os clamps de PRODUCAO -- so para inferencia/relato, NUNCA no
        caminho da perda de treino: [0,200] dB, [0,50] dB, [0,30] dB,
        [-150,0] dBm, [0,1] (cobertura -- ja e probabilidade por causa do
        sigmoid do forward; o clamp aqui e so uma guarda de seguranca
        numerica, nao deveria alterar nenhum valor na pratica).
        """
        out = pred.clone()
        out[:, 0] = torch.clamp(out[:, 0], *CLAMP_PL[0])
        out[:, 1] = torch.clamp(out[:, 1], *CLAMP_PL[1])
        out[:, 2] = torch.clamp(out[:, 2], *CLAMP_PL[2])
        out[:, 3] = torch.clamp(out[:, 3], *CLAMP_RSSI)
        out[:, 4] = torch.clamp(out[:, 4], *CLAMP_COVERAGE)
        return out


def sha256_do_arquivo(caminho: Path, bloco: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while True:
            chunk = f.read(bloco)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    # Teste de fumaca minimo, SEM GPU: shape, ausencia de NaN, saturacao
    # medida (deve ser 0 por construcao, exceto no unico canal com sigmoid),
    # e checagem de que fisico() respeita os clamps de producao.
    torch.manual_seed(0)
    dec = AffineDecoderV3(in_channels=64, hidden_channels=32, out_channels=5)
    emb = torch.randn(2000, 64) * 8.0  # embeddings "grandes" de proposito, tenta forcar saturacao
    pred = dec(emb)
    assert pred.shape == (2000, 5), pred.shape
    assert not torch.isnan(pred).any(), "NaN na saida do decodificador V3"

    # saturacao: fracao de |pre-ativacao| > 4 nos canais 0-3 (softplus/sigmoid
    # antigos saturavam ali); aqui os canais 0-3 sao afins, entao a fracao
    # "saturada" no sentido do gate antigo e SEMPRE 0 (nao existe patamar).
    raw = dec.layers(emb)
    sat_0a3 = float((raw[:, :4].abs() > 4.0).float().mean())
    print(f"fracao |pre-ativacao|>4 nos canais 0-3 (nao usada para saturar mais): {sat_0a3:.4f}")

    fis = dec.fisico(pred)
    assert fis[:, 0].min() >= 0.0 and fis[:, 0].max() <= 200.0
    assert fis[:, 1].min() >= 0.0 and fis[:, 1].max() <= 50.0
    assert fis[:, 2].min() >= 0.0 and fis[:, 2].max() <= 30.0
    assert fis[:, 3].min() >= -150.0 and fis[:, 3].max() <= 0.0
    assert fis[:, 4].min() >= 0.0 and fis[:, 4].max() <= 1.0

    print("OK: AffineDecoderV3 smoke de CPU passou.", pred.shape,
          "| sha256(este arquivo)=", sha256_do_arquivo(Path(__file__)))
