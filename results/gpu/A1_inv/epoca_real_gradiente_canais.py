#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpu/A1_inv/epoca_real_gradiente_canais.py -- itens (b) passo 2 e (c), PLANO A
gate A1, investigacao (nao e o gate, nao decide veredito -- so mede).

Roda `train_gnn_v3.py` (o WRAPPER de A0, sem editar nada nele) por
importlib, chamando `main(argv)` diretamente no MESMO processo, com
--epochs 1 --mmap --max-nodes 100000 sobre Bauru Q2 (dado real). Antes de
chamar `main()`, registra DOIS ganchos de diagnostico (monkeypatch de
METODO em objetos JA existentes, nao edita nenhum arquivo):

  (A) `gnn_rf_model.GNNRFModel.__init__` (patch de metodo na CLASSE --
      afeta a instancia real criada dentro do congelado, mesmo que o
      congelado tenha feito `from gnn_rf_model import GNNRFModel` antes
      do patch, porque e o MESMO objeto de classe): so acrescenta
      `self` a uma lista global logo apos o __init__ original rodar.
  (B) `torch.optim.AdamW.step` (patch de metodo na CLASSE, idem): a cada
      chamada (uma por batch/passo real de treino, feita pelo
      `GradScaler.step(optimizer)` do congelado), le a norma do
      gradiente JA CALCULADO (`p.grad`, apos backward + unscale + clip,
      ANTES do update) dos parametros-alvo do encoder (lin_l.weight,
      lin_l.bias, lin_r.weight das camadas 0-3 do edge type
      terrain->antenna, in_range_of) e acumula por passo.

Depois que `train_gnn_v3.main()` termina (treino + checkpoint +
predicoes .npz FISICAS de teste, que ja ficam gravadas por A0), este
script recarrega o MESMO checkpoint, reconstroi a particao (mesma
funcao `v3_common.carregar_base_e_particoes`, mesmos seeds/geometria) e
roda o modelo em eval() sobre VAL e TESTE guardando as predicoes AFINS
(antes de `fisico()`) para medir, por canal 0-3: fracao fora da faixa
fisica, mediana/P95 do excesso (distancia a borda mais proxima), e a
fracao de ALVOS exatamente 0 no canal 1 (path_loss_vegetation).

Nao editar modelo_v3/ nem os congelados. So leitura + monkeypatch de
metodo em memoria (instrumentacao, nao mudanca de comportamento nem de
resultado numerico do treino).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODELO_V3_DIR = Path("/trabalho/ARPIA_RF/TOPO_RF_PROJETO/GNN_RF/MDPI_Mathematics/"
                      "_v3_2026-09-25/gpu/modelo_v3")
A1_INV_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODELO_V3_DIR))
import v3_common as v3  # noqa: E402

T0 = time.perf_counter()


def log(msg):
    print(f"[epoca_real +{time.perf_counter()-T0:.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# Gancho (A): captura a instancia do GNNRFModel assim que e construida.
# --------------------------------------------------------------------------
MODELOS_CRIADOS = []


def instalar_gancho_captura_modelo():
    v3.patch_decoder_gnn()
    import gnn_rf_model
    orig_init = gnn_rf_model.GNNRFModel.__init__

    def init_com_captura(self, *a, **kw):
        orig_init(self, *a, **kw)
        MODELOS_CRIADOS.append(self)
        log(f"[gancho-A] GNNRFModel instanciado (id={id(self)}), "
            f"n_params={sum(p.numel() for p in self.parameters())}")

    gnn_rf_model.GNNRFModel.__init__ = init_com_captura
    return gnn_rf_model


# --------------------------------------------------------------------------
# Gancho (B): a cada AdamW.step(), le p.grad dos parametros-alvo do
# encoder (edge type terrain->antenna, in_range_of, camadas 0-3).
# --------------------------------------------------------------------------
GRAD_LOG = []  # lista de dicts por passo
ALVO_SUBSTR = "in_range_of___antenna"
PASSO_COUNTER = {"n": 0}


def instalar_gancho_gradiente():
    orig_step = torch.optim.AdamW.step

    def step_com_gancho(self, *a, **kw):
        if MODELOS_CRIADOS:
            modelo = MODELOS_CRIADOS[-1]
            registro = {"passo": PASSO_COUNTER["n"]}
            for nome, p in modelo.named_parameters():
                if ALVO_SUBSTR in nome and (".lin_l." in nome or ".lin_r." in nome):
                    registro[nome] = (None if p.grad is None
                                       else float(p.grad.detach().norm().item()))
            GRAD_LOG.append(registro)
            PASSO_COUNTER["n"] += 1
        return orig_step(self, *a, **kw)

    torch.optim.AdamW.step = step_com_gancho


CLAMPS_0A3 = {
    0: (0.0, 200.0, "path_loss_total"),
    1: (0.0, 50.0, "path_loss_vegetation"),
    2: (0.0, 30.0, "path_loss_terrain"),
    3: (-150.0, 0.0, "rssi"),
}


def diagnostico_canais(afim: np.ndarray, alvo: np.ndarray) -> dict:
    d = {}
    for c, (lo, hi, nome) in CLAMPS_0A3.items():
        v = afim[:, c]
        fora_baixo = np.maximum(lo - v, 0.0)
        fora_alto = np.maximum(v - hi, 0.0)
        excesso = np.maximum(fora_baixo, fora_alto)
        fora = excesso > 0.0
        d[nome] = {
            "n": int(v.shape[0]),
            "fracao_fora_da_faixa": float(fora.mean()),
            "excesso_mediana_dB_dBm": float(np.median(excesso[fora])) if fora.any() else 0.0,
            "excesso_p95_dB_dBm": float(np.percentile(excesso[fora], 95)) if fora.any() else 0.0,
            "excesso_max_dB_dBm": float(excesso.max()) if fora.any() else 0.0,
        }
        if c == 1:
            alvo_c = alvo[:, c]
            d[nome]["fracao_alvos_exatamente_zero"] = float((alvo_c == 0.0).mean())
    return d


def gerar_afim_particao(mod, ctx, model, device, particao: str, k_antenna: int, k_terrain: int,
                         eval_batch_size: int, seed: int) -> dict:
    """Mesma logica de v3_common.gerar_predicoes_teste_gnn, mas devolve a
    predicao AFIM (antes de fisico()), para qualquer particao (val/test)."""
    torch.manual_seed(seed)
    graph_p, _info = mod.induzir_particao(
        ctx.base, torch.from_numpy(ctx.parts_local[particao]), ctx.n_antenna, log, particao)
    nn_kw = {mod.ET_AT: [k_antenna], mod.ET_TT: [k_terrain], mod.ET_TA: [k_antenna]}
    from torch_geometric.loader import NeighborLoader
    loader = NeighborLoader(data=graph_p, num_neighbors=nn_kw,
                             input_nodes=("terrain", None), batch_size=eval_batch_size,
                             shuffle=False, num_workers=0)
    n_p = int(graph_p["terrain"].x.shape[0])
    model.eval()
    P, T, I = [], [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            bs = b["terrain"].batch_size
            out = model(b)
            P.append(out["predictions"][:bs].float().detach().cpu())
            T.append(b["terrain"].rf_targets[:bs].float().detach().cpu())
            I.append(b["terrain"].n_id[:bs].cpu())
    po, to, visto = mod._scatter_por_semente(torch.cat(I), torch.cat(P), torch.cat(T), n_p)
    assert bool(visto.all()), f"algum no da particao '{particao}' nao recebeu predicao"
    return {
        "afim": po.numpy().astype(np.float32),
        "alvo": to.numpy().astype(np.float32),
        "n": n_p,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponivel -- abortando (item b/c pedem GPU real)")
    livre0 = torch.cuda.mem_get_info()[0] / 1024**2
    log(f"VRAM livre antes: {livre0:.0f} MiB")
    if livre0 < 1024:
        raise RuntimeError(f"VRAM livre {livre0:.0f} MiB < 1024 MiB -- abortando por regra do protocolo")

    instalar_gancho_captura_modelo()
    instalar_gancho_gradiente()

    # importa o WRAPPER train_gnn_v3.py por importlib (nao edita, so chama main()).
    spec = importlib.util.spec_from_file_location("train_gnn_v3_diag", MODELO_V3_DIR / "train_gnn_v3.py")
    train_gnn_v3 = importlib.util.module_from_spec(spec)
    sys.modules["train_gnn_v3_diag"] = train_gnn_v3
    spec.loader.exec_module(train_gnn_v3)

    evid_dir = A1_INV_DIR / "epoca_real"
    run_label = "A1_inv_epoca_real_gnn_bauru_Q2"
    argv = [
        "--cidade", "bauru", "--quadrante", "Q2",
        "--seed-treino", "42", "--split-seed", "42",
        "--epochs", "1", "--max-nodes", "100000", "--mmap", "--smoke",
        "--evid-dir", str(evid_dir), "--run-label", run_label,
        "--no-baselines",
    ]
    log(f"chamando train_gnn_v3.main({argv})")
    rc = train_gnn_v3.main(argv)
    if rc != 0:
        raise RuntimeError(f"train_gnn_v3.main retornou {rc}")

    livre1 = torch.cuda.mem_get_info()[0] / 1024**2
    log(f"VRAM livre depois do treino de 1 epoca: {livre1:.0f} MiB")

    # -------- (b) sumario do gradiente por passo --------
    dead_esperadas_layer3 = True  # achado A1-b-1: camada 3 nunca participa (p.grad=None sempre)
    nomes_medidos = sorted({k for reg in GRAD_LOG for k in reg if k != "passo"})
    zero_em_todos_os_passos = []
    algum_none_em_todos = []
    for nome in nomes_medidos:
        vals = [reg.get(nome) for reg in GRAD_LOG]
        if all(v is not None and v < 1e-12 for v in vals):
            zero_em_todos_os_passos.append(nome)
        if all(v is None for v in vals):
            algum_none_em_todos.append(nome)

    achado_b = {
        "n_passos_medidos": len(GRAD_LOG),
        "nomes_parametros_medidos": nomes_medidos,
        "grad_norm_por_passo": GRAD_LOG,
        "parametros_com_gradiente_zero_em_TODOS_os_passos_reais": zero_em_todos_os_passos,
        "parametros_com_grad_None_em_TODOS_os_passos_reais_camada_nunca_usada": algum_none_em_todos,
        "achado_b_confirmado_em_producao": any(
            ".lin_l.weight" in n for n in zero_em_todos_os_passos),
        "nota": ("mesma configuracao de fanout do gate (num_neighbors com lista de 1 elemento por "
                 "tipo de aresta, {ET_AT:[k_antenna], ET_TT:[k_terrain], ET_TA:[k_antenna]}) e usada "
                 "TAMBEM pelo script de PRODUCAO train_gnn_c0_spatial.py (linha nn_kw, identica); "
                 "isto NAO e um artefato exclusivo do teste de overfit do gate."),
    }

    # -------- (c) predicoes afins em val e teste, reconstruindo particao --------
    rf_data_file = "transfer_dataset_bauru_v19_Q2_enriched_v2.pt"
    graph_file = "bauru_v19_Q2_gpu.pt"
    run_json_path = evid_dir / run_label / f"run_{run_label}.json"
    with open(run_json_path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    grid_km_usado = rec.get("geometria", {}).get("grid_km_usado", 5.0)
    buffer_km_usado = rec.get("geometria", {}).get("buffer_km_usado", 2.0)

    mod = v3.carregar_modulo_congelado(v3.FROZEN_GNN_SCRIPT, "train_gnn_c0_spatial_diag_c")
    ctx = v3.carregar_base_e_particoes(
        mod=mod, graph_dir=v3.GRAPH_DIR_DEFAULT, rf_data_file=rf_data_file, graph_file=graph_file,
        max_nodes=100000, window_anchor="cobertura",
        grid_km=grid_km_usado, buffer_km=buffer_km_usado, split_frac=(0.70, 0.15, 0.15),
        split_seed=42, smoke_geometria=False, mmap=True, precisa_arestas_ter_ter=True, log=log,
    )

    device = "cuda"
    from gnn_rf_model import GNNRFModel as GNNRFModel_eval
    model_eval = GNNRFModel_eval(
        terrain_dim=int(ctx.x_full.shape[1]), antenna_dim=int(ctx.ant_x.shape[1]),
        hidden_dim=256, num_layers=4, heads=4, edge_dim=2,
        output_dim=5, dropout=0.1, use_physics_constraints=True).to(device)
    ckpt_path = evid_dir / run_label / "checkpoints" / "checkpoint_best.pt"
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_eval.load_state_dict(st["model_state_dict"])
    log(f"checkpoint carregado: {ckpt_path} (melhor_epoca={st.get('epoch')})")

    achado_c = {}
    for particao in ("val", "test"):
        r = gerar_afim_particao(mod, ctx, model_eval, device, particao,
                                 k_antenna=20, k_terrain=8, eval_batch_size=24576, seed=42)
        diag = diagnostico_canais(r["afim"], r["alvo"])
        achado_c[particao] = {"n_nos": r["n"], "por_canal": diag}
        log(f"[{particao}] canal1(path_loss_vegetation) fracao_fora={diag['path_loss_vegetation']['fracao_fora_da_faixa']:.4f} "
            f"mediana_excesso={diag['path_loss_vegetation']['excesso_mediana_dB_dBm']:.3f} dB "
            f"p95={diag['path_loss_vegetation']['excesso_p95_dB_dBm']:.3f} dB "
            f"frac_alvo_zero={diag['path_loss_vegetation']['fracao_alvos_exatamente_zero']:.4f}")

    livre2 = torch.cuda.mem_get_info()[0] / 1024**2

    resultado = {
        "item": "A1-b passo 2 (gradiente em epoca real) e A1-c (canais em epoca real, val+teste)",
        "comando_reproduzido": ("importlib de train_gnn_v3.py, main(['--cidade','bauru','--quadrante','Q2',"
                                 "'--seed-treino','42','--split-seed','42','--epochs','1','--max-nodes','100000',"
                                 "'--mmap','--evid-dir',...,'--run-label','A1_inv_epoca_real_gnn_bauru_Q2'])"),
        "run_json": str(run_json_path),
        "checkpoint": str(ckpt_path),
        "vram_livre_mib": {"antes": livre0, "apos_treino": livre1, "apos_avaliacao": livre2},
        "b_gradiente_epoca_real": achado_b,
        "c_canais_epoca_real": achado_c,
        "comparacao_com_gate_overfit_300passos": {
            "gate_gnn_clamp_final_path_loss_vegetation": 0.3125219941139221,
            "gate_mlp_clamp_final_path_loss_vegetation": 0.3299560546875,
            "nota": ("o gate mede fracao fora do clamp APOS 300 passos de Adam OVERFITANDO 1 SO "
                     "batch fixo (lr=3e-3, 300 atualizacoes no MESMO batch); aqui medimos APOS "
                     "1 epoca real (poucos batches, lr=1e-3 default do wrapper, SEM overfit "
                     "deliberado) sobre val/teste retidos (nunca vistos no treino)."),
        },
        "tempo_total_s": time.perf_counter() - T0,
    }
    saida = A1_INV_DIR / "epoca_real_gradiente_canais.json"
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    log(f"gravado: {saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
